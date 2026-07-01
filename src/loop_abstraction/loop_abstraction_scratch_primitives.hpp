#pragma once

#include <algorithm>
#include <array>
#include <concepts>
#include <optional>
#include <typeindex>
#include <unordered_map>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "basic_types.hpp"
#include "interface/mesh_data.hpp"
#include "kokkos_types.hpp"
#include "mesh/mesh.hpp"
#include "utils/indexer.hpp"

namespace loop_abstraction {

template <class Halo>
struct HaloBox {
  static constexpr int min_k = [] {
    int out = Halo::dk(0);
    for (int n = 1; n < Halo::npoints; ++n) out = std::min(out, Halo::dk(n));
    return out;
  }();
  static constexpr int max_k = [] {
    int out = Halo::dk(0);
    for (int n = 1; n < Halo::npoints; ++n) out = std::max(out, Halo::dk(n));
    return out;
  }();
  static constexpr int min_j = [] {
    int out = Halo::dj(0);
    for (int n = 1; n < Halo::npoints; ++n) out = std::min(out, Halo::dj(n));
    return out;
  }();
  static constexpr int max_j = [] {
    int out = Halo::dj(0);
    for (int n = 1; n < Halo::npoints; ++n) out = std::max(out, Halo::dj(n));
    return out;
  }();
  static constexpr int min_i = [] {
    int out = Halo::di(0);
    for (int n = 1; n < Halo::npoints; ++n) out = std::min(out, Halo::di(n));
    return out;
  }();
  static constexpr int max_i = [] {
    int out = Halo::di(0);
    for (int n = 1; n < Halo::npoints; ++n) out = std::max(out, Halo::di(n));
    return out;
  }();
  static constexpr int nk = max_k - min_k + 1;
  static constexpr int nj = max_j - min_j + 1;
  static constexpr int ni = max_i - min_i + 1;
  static constexpr int size = nk * nj * ni;
};

template <class T, class IndexRange>
struct StackScratch1D {
  using halo_t = typename IndexRange::halo_t;
  using box_t = HaloBox<halo_t>;

  mutable std::array<T, box_t::size> data{};
  int ks = 0;
  int js = 0;
  int is = 0;

  KOKKOS_INLINE_FUNCTION
  explicit StackScratch1D(const IndexRange &idx_range)
      : ks(idx_range.ks), js(idx_range.js), is(idx_range.is) {}

  KOKKOS_FORCEINLINE_FUNCTION T &operator()(Index3 idx) const {
    return data[DenseIndex(idx.k - ks, idx.j - js, idx.i - is)];
  }

  KOKKOS_FORCEINLINE_FUNCTION T &operator()(MemoryOffset idx) const {
    return data[DenseIndex(idx.dk, idx.dj, idx.di)];
  }

  KOKKOS_FORCEINLINE_FUNCTION
  T &operator()(int k, int j, int i) const {
    return (*this)(Index3{k, j, i});
  }

  KOKKOS_FORCEINLINE_FUNCTION
  constexpr std::size_t size() const {
    return box_t::size;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  constexpr std::size_t shmem_size() const {
    return 0;
  }

 private:
  KOKKOS_FORCEINLINE_FUNCTION static bool ContainsDeclaredOffset(const int dk,
                                                                 const int dj,
                                                                 const int di) {
    for (int n = 0; n < halo_t::npoints; ++n) {
      if (dk == halo_t::dk(n) && dj == halo_t::dj(n) && di == halo_t::di(n)) {
        return true;
      }
    }
    return false;
  }

  KOKKOS_FORCEINLINE_FUNCTION static int DenseIndex(const int dk, const int dj,
                                                    const int di) {
    KOKKOS_ASSERT(dk >= box_t::min_k && dk <= box_t::max_k);
    KOKKOS_ASSERT(dj >= box_t::min_j && dj <= box_t::max_j);
    KOKKOS_ASSERT(di >= box_t::min_i && di <= box_t::max_i);
    KOKKOS_ASSERT(ContainsDeclaredOffset(dk, dj, di));
    return ((dk - box_t::min_k) * box_t::nj + (dj - box_t::min_j)) * box_t::ni +
           (di - box_t::min_i);
  }
};

template <class IndexRange, class T>
struct HostScratch1D {
  IndexRange idx_range;
  mutable std::vector<T> data;

  HostScratch1D(const IndexRange &idx_range)
      : idx_range(idx_range), data(idx_range.size(), T{}) {}

  template <class IDXT>
  KOKKOS_FORCEINLINE_FUNCTION T &operator()(IDXT idx) const {
    return data[idx_range.CompactIndex(idx)];
  }

  KOKKOS_FORCEINLINE_FUNCTION T &operator()(int k, int j, int i) const {
    return data[idx_range.CompactIndex(k, j, i)];
  }

  KOKKOS_FORCEINLINE_FUNCTION
  std::size_t size() const {
    return data.size();
  }

  KOKKOS_FORCEINLINE_FUNCTION
  constexpr std::size_t shmem_size() const {
    return 0;
  }
};

template <class IndexRange, class T>
struct TeamScratch1D {
  static constexpr int scratch_level = 1;
  IndexRange idx_range;
  parthenon::ScratchPad1D<T> data;
  TeamScratch1D(const IndexRange &idx_range) 
      : idx_range(idx_range),
        data(idx_range.team_member->team_scratch(scratch_level), idx_range.size()) {} 

  template<class IDXT>
  KOKKOS_FORCEINLINE_FUNCTION
  T &operator()(IDXT idx_in) const { return data(idx_range.CompactIndex(idx_in)); }
  
  KOKKOS_FORCEINLINE_FUNCTION T &operator()(int k, int j, int i) const {
    return data(idx_range.CompactIndex(k, j, i));
  }
  KOKKOS_FORCEINLINE_FUNCTION
  std::size_t size() const {
    return idx_range.size();
  }

  KOKKOS_FORCEINLINE_FUNCTION
  std::size_t shmem_size() const {
    return parthenon::ScratchPad1D<T>::shmem_size(size());
  }
};

template <class T, class IndexRange>
KOKKOS_INLINE_FUNCTION
auto GetPerPointScratch(const IndexRange &idx_range) {
  if constexpr (IndexRange::index_space_t::loop_tag_v == loop_tag::boiv) {
    return StackScratch1D<T, IndexRange>(idx_range);
  } else if constexpr (IndexRange::index_space_t::backend_v == loop_backend::raw) {
    return HostScratch1D<IndexRange, T>(idx_range);
  } else if constexpr (IndexRange::index_space_t::backend_v == loop_backend::kokkos) {
    return TeamScratch1D<IndexRange, T>(idx_range);
  } else {
    static_assert(always_false<IndexRange>,
                  "Unsupported loop backend for per-point scratch.");
  }
}

template <class T, class IndexRange>
KOKKOS_INLINE_FUNCTION std::size_t GetPerPointScratchSize(const IndexRange &idx_range) {
  if constexpr (IndexRange::index_space_t::loop_tag_v == loop_tag::boiv ||
                IndexRange::index_space_t::backend_v == loop_backend::raw) {
    return 0;
  } else if constexpr (IndexRange::index_space_t::backend_v == loop_backend::kokkos) {
    return parthenon::ScratchPad1D<T>::shmem_size(idx_range.size());
  } else {
    static_assert(always_false<IndexRange>,
                  "Unsupported loop backend for per-point scratch size.");
  }
}

template <class T, class Halo, class IndexSpaceType>
inline std::size_t GetPerTeamScratchSize(const IndexSpaceType &idx_space) {
  if constexpr (IndexSpaceType::loop_tag_v == loop_tag::boiv ||
                IndexSpaceType::backend_v == loop_backend::raw) {
    return 0;
  } else if constexpr (IndexSpaceType::backend_v == loop_backend::kokkos) {
    const std::size_t key = reinterpret_cast<std::size_t>(&idx_space) ^
                            (std::type_index(typeid(T)).hash_code() << 1) ^
                            (std::type_index(typeid(Halo)).hash_code() << 2);
    static thread_local std::unordered_map<std::size_t, std::size_t> cache;
    if (const auto it = cache.find(key); it != cache.end()) {
      return it->second;
    }
    std::size_t scratch_size = 0;
    using BaseRangeType = InnerIndexRange<IndexSpaceType>;
    const auto &logical_kji = idx_space.GetLogicalIndexer();
    
    // Lambda for calculating the amount of scratch required for a given inner IndexRange
    auto update_scratch_size = [&](const auto &base_range) {
      const auto halo_range = base_range.template AddHalo<Halo>();
      scratch_size =
          std::max(scratch_size,
                   parthenon::ScratchPad1D<T>::shmem_size(halo_range.size()));
    };

    if constexpr (IndexSpaceType::loop_tag_v == loop_tag::bvoi) {
      for (int b = 0; b < idx_space.GetNBlocks(); ++b) {
        const BaseRangeType idx_range(idx_space, logical_kji, b);
        update_scratch_size(idx_range);
      }
    } else {
      const int nouter = GetNOuter(idx_space);
      for (int b = 0; b < idx_space.GetNBlocks(); ++b) {
        for (int o = 0; o < nouter; ++o) {
          const int logical_start = o * idx_space.GetNInner();
          const int logical_end =
              std::min((o + 1) * idx_space.GetNInner() - 1,
                       static_cast<int>(logical_kji.size()) - 1);
          const BaseRangeType idx_range(idx_space, logical_kji, b, logical_start,
                                        logical_end);
          update_scratch_size(idx_range);
        }
      }
    }
    cache.emplace(key, scratch_size);
    return scratch_size;
  } else {
    static_assert(always_false<IndexSpaceType>,
                  "Unsupported loop backend for per-team scratch size.");
  }
}

template <class T, class IndexSpaceType>
inline std::size_t GetPerTeamScratchSize(const IndexSpaceType &idx_space) {
  return GetPerTeamScratchSize<T, halo::none_t>(idx_space);
}

} // namespace loop_abstraction
