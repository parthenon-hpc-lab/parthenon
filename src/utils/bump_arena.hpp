//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020-2024 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================

#ifndef UTILS_BUMP_ARENA_HPP_
#define UTILS_BUMP_ARENA_HPP_

#include <algorithm>
#include <cstddef>
#include <vector>

namespace parthenon {

// A per-thread bump (arena) allocator for transient host scratch.
//
// Motivation: on the raw (host==device) loop backend, per-point scratch used to
// be a fresh std::vector<T>(N, T{}) constructed inside every outer-loop iteration
// -- i.e. a malloc + zero-fill per (block, plane) for each of the ~10 scratch
// arrays a kernel allocates. This arena replaces that with a single persistent
// buffer per thread: allocations are pointer bumps returning *uninitialized*
// memory, and the bump pointer is reset once per outer iteration. The buffer only
// ever grows to the high-water mark, so steady state performs zero allocations.
//
// Contract: because allocate() returns uninitialized memory, any storage that is
// read-modify-written (e.g. += accumulators) must be explicitly zeroed by the
// caller -- exactly as the Kokkos team_scratch path already requires.
//
// This is a plain host allocator with no dependence on Kokkos or the loop
// abstraction; it is never used on the Kokkos/GPU backend (which allocates through
// team_scratch instead). Access it through GetThreadLocalBumpArena(), whose
// thread_local instance makes it correct for a threaded host backend at no cost on
// serial.
class ThreadLocalBumpArena {
 public:
  // Reserve `bytes` of raw storage from the arena, aligned for any scalar type.
  //
  // The primary buffer only ever grows at reset() (to the high-water mark of the
  // previous iteration), never here -- so bump allocations never reallocate it and
  // pointers handed out earlier in the same iteration stay valid. If a request
  // does not fit the primary buffer (only possible while the high-water mark is
  // still being discovered, i.e. the first iteration), it is satisfied from an
  // individually-heap-allocated overflow block that also stays put until reset().
  void *allocate(std::size_t bytes) {
    constexpr std::size_t align = alignof(std::max_align_t);
    const std::size_t aligned = (offset_ + align - 1) & ~(align - 1);
    const std::size_t end = aligned + bytes;
    // Always advance the bump pointer so every allocation reserves a distinct slot
    // in the eventual contiguous buffer; high_water_ then records the footprint the
    // whole iteration needs, and reset() sizes the primary buffer to it.
    offset_ = end;
    high_water_ = std::max(high_water_, end);
    if (end <= buffer_.size()) {
      return buffer_.data() + aligned;
    }
    // Doesn't fit the primary buffer yet (only until the high-water mark is
    // discovered): serve from a stable, individually-heap-allocated overflow block.
    overflow_.emplace_back(bytes);
    return overflow_.back().data();
  }

  // Reclaim all outstanding allocations. Called once per outer iteration. Grows the
  // primary buffer to the high-water mark (so subsequent identical iterations are
  // served entirely from it with no allocation) and drops any overflow blocks.
  void reset() {
    if (high_water_ > buffer_.size()) buffer_.resize(high_water_);
    if (!overflow_.empty()) overflow_.clear();
    offset_ = 0;
    high_water_ = 0;
  }

 private:
  std::vector<std::byte> buffer_;
  std::vector<std::vector<std::byte>> overflow_;
  std::size_t offset_ = 0;
  std::size_t high_water_ = 0;
};

inline ThreadLocalBumpArena &GetThreadLocalBumpArena() {
  static thread_local ThreadLocalBumpArena arena;
  return arena;
}

} // namespace parthenon

#endif // UTILS_BUMP_ARENA_HPP_
