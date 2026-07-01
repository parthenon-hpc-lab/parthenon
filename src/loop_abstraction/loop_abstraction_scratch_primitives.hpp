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

template <class T, int N>
struct StackScratch1D {
  mutable std::array<T, N> data{};
  // TODO: Make this work for Halos
  template <class IDXT>
  KOKKOS_FORCEINLINE_FUNCTION
  T &operator()(IDXT) const {
    return data[0];
  }
  
  KOKKOS_FORCEINLINE_FUNCTION
  T &operator()(int k, int j, int i) const {
    return data[0];
  }

  KOKKOS_FORCEINLINE_FUNCTION
  constexpr std::size_t size() const {
    return N;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  constexpr std::size_t shmem_size() const {
    return 0;
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
    return StackScratch1D<T, IndexRange::halo_t::npoints>{};
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
