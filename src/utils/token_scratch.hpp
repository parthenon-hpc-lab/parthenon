//========================================================================================
// (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001 for Los
// Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
// for the U.S. Department of Energy/National Nuclear Security Administration. All rights
// in the program are reserved by Triad National Security, LLC, and the U.S. Department
// of Energy/National Nuclear Security Administration. The Government is granted for
// itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do so.
//========================================================================================
#ifndef UTILS_TOKEN_SCRATCH_HPP_
#define UTILS_TOKEN_SCRATCH_HPP_

// This file was created part with generative AI

#include <cassert>
#include <type_traits>

#include <Kokkos_Core.hpp>

#include "cleantypes.hpp"

namespace parthenon {

//========================================================================================
//! \class ScratchAllocator
//! \brief RAII wrapper for per-token scratch allocation with type-safe view carving
//!
//! Provides stack-like allocation of typed views from a pre-allocated token-specific
//! buffer. Handles alignment automatically and validates bounds. Token is automatically
//! released when the allocator goes out of scope.
//========================================================================================
template <Kokkos::Experimental::UniqueTokenScope TokenScope =
              Kokkos::Experimental::UniqueTokenScope::Global,
          typename ExecutionSpace = Kokkos::DefaultExecutionSpace,
          typename MemorySpace = ExecutionSpace::memory_space>
class ScratchAllocator {
 private:
  using PoolView = Kokkos::View<char **, MemorySpace>;
  using TokenType = Kokkos::Experimental::UniqueToken<ExecutionSpace, TokenScope>;

  const PoolView &pool_;
  TokenType const *tokens_;
  int token_id_;
  std::size_t capacity_;
  std::size_t offset_;

  //! Align offset to satisfy alignment requirements of type T
  KOKKOS_INLINE_FUNCTION
  static std::size_t AlignOffset(std::size_t offset, std::size_t alignment) {
    return (offset + alignment - 1) & ~(alignment - 1);
  }

 public:
  //! Construct allocator for a specific token
  KOKKOS_INLINE_FUNCTION
  ScratchAllocator(const PoolView &pool, TokenType const *tokens, int token,
                   std::size_t capacity)
      : pool_(pool), tokens_(tokens), token_id_(token), capacity_(capacity), offset_(0) {}

  // Disable copy to ensure single ownership of token
  ScratchAllocator(const ScratchAllocator &) = delete;
  ScratchAllocator &operator=(const ScratchAllocator &) = delete;

  // Enable move for flexibility
  KOKKOS_INLINE_FUNCTION
  ScratchAllocator(ScratchAllocator &&other)
      : pool_(other.pool_), tokens_(other.tokens_), token_id_(other.token_id_),
        capacity_(other.capacity_), offset_(other.offset_) {
    other.token_id_ = -1; // Mark as moved-from
  }

  //! Destructor releases the token
  KOKKOS_INLINE_FUNCTION
  ~ScratchAllocator() {
    if (token_id_ >= 0 && tokens_ != nullptr) {
      tokens_->release(token_id_);
    }
  }

  //! Allocate an unmanaged view of type T with arbitrary rank
  //! \param dims Variadic pack of dimension sizes
  //! \return Unmanaged Kokkos::View with rank = sizeof...(dims)
  //!
  //! Uses fold expressions to compute total elements and constructs
  //! the appropriate View type using pointer_depth helper.
  //!
  //! Examples:
  //!   allocate_view<double>(100)       -> View<double*>
  //!   allocate_view<int>(10, 20)       -> View<int**>
  //!   allocate_view<float>(5, 10, 15)  -> View<float***>
  template <typename T, typename... Dims>
  KOKKOS_INLINE_FUNCTION auto allocate_view(Dims... dims) {
    static_assert(sizeof...(Dims) > 0, "At least one dimension required");
    static_assert((std::is_convertible_v<Dims, std::size_t> && ...),
                  "All dimensions must be convertible to std::size_t");

    constexpr std::size_t rank = sizeof...(Dims);
    using DataType = typename cleantypes::pointer_depth<T, rank>::type;

    // Use fold expression to compute total number of elements
    const std::size_t total_elements = (dims * ...);
    const std::size_t bytes_needed = total_elements * sizeof(T);
    const std::size_t aligned_offset = AlignOffset(offset_, alignof(T));

#ifndef NDEBUG
    // In debug builds, check for overflow
    if (aligned_offset + bytes_needed > capacity_) {
      Kokkos::abort("TokenScratch: allocation exceeded capacity");
    }
#endif

    char *base = pool_.data() + token_id_ * capacity_ + aligned_offset;
    offset_ = aligned_offset + bytes_needed;

    return Kokkos::View<DataType, MemorySpace, Kokkos::MemoryUnmanaged>(
        reinterpret_cast<T *>(base), dims...);
  }

  //! Get current offset (useful for debugging)
  KOKKOS_INLINE_FUNCTION
  std::size_t current_offset() const { return offset_; }

  //! Get remaining capacity
  KOKKOS_INLINE_FUNCTION
  std::size_t remaining() const { return capacity_ - offset_; }
};

//========================================================================================
//! \class TokenScratchPool
//! \brief Manages a pool of per-token scratch memory using Kokkos UniqueToken
//!
//! Allocates a 2D pool where each row corresponds to a unique token's scratch buffer.
//! Threads acquire tokens, get a ScratchAllocator, and carve typed views from their
//! token's buffer.
//!
//! Example:
//! \code
//!   TokenScratchPool<ExecSpace, MemorySpace> pool(64*1024); // 64KB per token
//!   Kokkos::parallel_for(policy, KOKKOS_LAMBDA(int i) {
//!     auto scratch = pool.acquire();
//!     auto doubles = scratch.template allocate_view<double>(100);
//!     auto ints = scratch.template allocate_view<int>(50);
//!     // Use views...
//!   });
//! \endcode
//========================================================================================
template <typename ExecutionSpace = Kokkos::DefaultExecutionSpace,
          typename MemorySpace = ExecutionSpace::memory_space,
          Kokkos::Experimental::UniqueTokenScope TokenScope =
              Kokkos::Experimental::UniqueTokenScope::Global>
class TokenScratchPool {
 private:
  using PoolView = Kokkos::View<char **, MemorySpace>;
  using TokenType = Kokkos::Experimental::UniqueToken<ExecutionSpace, TokenScope>;

  TokenType tokens_;
  PoolView pool_;
  std::size_t bytes_per_token_;

 public:
  //! Construct pool with specified bytes per token
  //! \param bytes_per_token Amount of scratch memory for each unique token
  explicit TokenScratchPool(std::size_t bytes_per_token)
      : tokens_(), bytes_per_token_(bytes_per_token),
        pool_("token_scratch_pool", tokens_.size(), bytes_per_token) {}

  //! Acquire a token and return a scratch allocator for this thread
  //! The allocator automatically manages token lifetime via RAII
  KOKKOS_INLINE_FUNCTION
  auto acquire() const {
    const int token_id = tokens_.acquire();
    return ScratchAllocator<TokenScope, ExecutionSpace, MemorySpace>(
        pool_, &tokens_, token_id, bytes_per_token_);
  }

  //! Get the number of tokens in the pool
  KOKKOS_INLINE_FUNCTION
  std::size_t size() const { return tokens_.size(); }

  //! Get bytes per token
  KOKKOS_INLINE_FUNCTION
  std::size_t bytes_per_token() const { return bytes_per_token_; }

  //! Get total pool size in bytes
  KOKKOS_INLINE_FUNCTION
  std::size_t total_bytes() const { return tokens_.size() * bytes_per_token_; }
};
} // namespace parthenon

#endif // UTILS_TOKEN_SCRATCH_HPP_
