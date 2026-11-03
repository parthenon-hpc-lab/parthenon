//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2026 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================

#ifndef UTILS_VARIABLE_POOL_HPP_
#define UTILS_VARIABLE_POOL_HPP_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

#include <Kokkos_Core.hpp>

#include "kokkos_types.hpp"
#include "utils/object_pool.hpp"

namespace parthenon {

// A complete variable shape is required: two variables with the same total number of
// elements but different extents cannot generally share a rank-7 Kokkos view.
struct VariablePoolKey {
  std::array<int, MAX_VARIABLE_DIMENSION> dims;

  bool operator==(const VariablePoolKey &) const = default;
};

struct VariablePoolKeyHash {
  std::size_t operator()(const VariablePoolKey &key) const {
    std::size_t hash = 0;
    for (const auto dim : key.dims) {
      hash ^= std::hash<int>{}(dim) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }
    return hash;
  }
};

// This handle owns a pool lease when storage comes from a mesh pool. Direct allocations
// deliberately have no lease and are owned by the public ParArrayND held by Variable.
template <typename T>
class VariableStorage {
 public:
  using raw_t = ParArrayND<T>;
  using pool_t = ObjectPool<raw_t>;
  using owner_t = typename pool_t::owner_t;

  VariableStorage() = default;
  explicit VariableStorage(const raw_t &direct) : direct_(direct) {}
  explicit VariableStorage(const owner_t &owner) : owner_(owner), pooled_(true) {}

  raw_t view() const { return pooled_ ? static_cast<raw_t>(owner_) : direct_; }
  bool pooled() const { return pooled_; }
  bool allocated() const {
    return pooled_ ? owner_.is_allocated() : direct_.is_allocated();
  }

  void Reset() {
    direct_.Reset();
    owner_ = owner_t{};
    pooled_ = false;
  }

 private:
  raw_t direct_;
  owner_t owner_;
  bool pooled_ = false;
};

// A rank-7 facade over ChunkedObjectPoolMap. It only defines the rank-8 backing chunk
// and rank-7 subdivision; keyed pool bookkeeping is shared with communication buffers.
template <typename T>
class VariablePoolMap {
 public:
  using raw_t = ParArrayND<T>;
  using storage_t = VariableStorage<T>;
  using pool_t = typename storage_t::pool_t;
  using owner_t = typename storage_t::owner_t;
  using impl_t = ChunkedObjectPoolMap<VariablePoolKey, raw_t, VariablePoolKeyHash>;

  explicit VariablePoolMap(std::size_t chunk_size = 1)
      : impl_(AllocateChunk), chunk_size_(chunk_size) {}

  storage_t Acquire(const std::array<int, MAX_VARIABLE_DIMENSION> &dims) {
    const VariablePoolKey key{dims};
    if (!Contains(key)) AddPool(key);
    return storage_t(impl_.GetOwning(key));
  }

  bool Contains(const VariablePoolKey &key) const { return impl_.Contains(key); }

  void SetChunkSize(const std::size_t chunk_size) {
    chunk_size_ = std::max(chunk_size, std::size_t{1});
  }
  std::size_t GetChunkSize() const { return chunk_size_; }

  std::uint64_t SizeInBytes() const {
    std::uint64_t size = 0;
    for (const auto &[key, pool] : impl_.GetMap())
      size += pool.SizeInBytes();
    return size;
  }

  std::size_t NumObjectsInUse() const {
    std::size_t size = 0;
    for (const auto &[key, pool] : impl_.GetMap())
      size += pool.NumBuffersInUse();
    return size;
  }

 private:
  void AddPool(const VariablePoolKey &key) { impl_.AddPool(key, chunk_size_); }

  static raw_t AllocateChunk(pool_t *pool, const VariablePoolKey &key,
                             const std::size_t chunk_size) {
    static std::size_t counter = 0; // NOT thread safe.
    const auto &dims = key.dims;
    const auto label = "variable pool chunk " + std::to_string(counter++);
    ParArray8DRaw<T> chunk(label, chunk_size, dims[6], dims[5], dims[4], dims[3], dims[2],
                           dims[1], dims[0]);
    for (std::size_t i = 1; i < chunk_size; ++i) {
      pool->AddFreeObjectToPool(raw_t(
          Kokkos::subview(chunk, i, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(),
                          Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL())));
    }
    return raw_t(Kokkos::subview(chunk, 0, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(),
                                 Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(),
                                 Kokkos::ALL()));
  }

  impl_t impl_;
  std::size_t chunk_size_;
};

} // namespace parthenon

#endif // UTILS_VARIABLE_POOL_HPP_
