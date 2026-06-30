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

} // namespace loop_abstraction
