//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020-2024 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2024. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001
// for Los Alamos National Laboratory (LANL), which is operated by Triad
// National Security, LLC for the U.S. Department of Energy/National Nuclear
// Security Administration. All rights in the program are reserved by Triad
// National Security, LLC, and the U.S. Department of Energy/National Nuclear
// Security Administration. The Government is granted for itself and others
// acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license
// in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do
// so.
//========================================================================================

#ifndef LOOP_BOUND_TRANSLATOR_HPP_
#define LOOP_BOUND_TRANSLATOR_HPP_

#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

#include <Kokkos_Core.hpp>

#include "basic_types.hpp"
#include "kokkos_types.hpp"
#include "utils/indexer.hpp"
#include "utils/type_list.hpp"

namespace parthenon {
// Struct for translating between loop bounds given in terms of IndexRanges and loop
// bounds given in terms of raw integers
template <class... Bound_ts>
struct LoopBoundTranslator {
  using Bound_tl = TypeList<Bound_ts...>;
  static constexpr bool are_integers = std::is_integral_v<
      typename std::remove_reference<typename Bound_tl::template type<0>>::type>;
  static constexpr uint rank = sizeof...(Bound_ts) / (1 + are_integers);

  std::array<IndexRange, rank> bounds;

  KOKKOS_INLINE_FUNCTION
  IndexRange &operator[](int i) { return bounds[i]; }

  KOKKOS_INLINE_FUNCTION
  const IndexRange &operator[](int i) const { return bounds[i]; }

  KOKKOS_INLINE_FUNCTION
  explicit LoopBoundTranslator(Bound_ts... bounds_in) {
    if constexpr (are_integers) {
      std::array<int64_t, 2 * rank> bounds_arr{static_cast<int64_t>(bounds_in)...};
      for (int r = 0; r < rank; ++r) {
        bounds[r].s = static_cast<int64_t>(bounds_arr[2 * r]);
        bounds[r].e = static_cast<int64_t>(bounds_arr[2 * r + 1]);
      }
    } else {
      bounds = std::array<IndexRange, rank>{bounds_in...};
    }
  }

  template <int RankStart, int RankStop>
  auto GetKokkosFlatRangePolicy(DevExecSpace exec_space) const {
    constexpr int ndim = RankStop - RankStart;
    static_assert(ndim > 0, "Need a valid range of ranks");
    static_assert(RankStart >= 0, "Need a valid range of ranks");
    static_assert(RankStop <= rank, "Need a valid range of ranks");
    int64_t npoints = 1;
    for (int d = RankStart; d < RankStop; ++d)
      npoints *= (bounds[d].e + 1 - bounds[d].s);
    return Kokkos::Experimental::require(
        Kokkos::RangePolicy<>(exec_space, 0, npoints),
        Kokkos::Experimental::WorkItemProperty::HintLightWeight);
  }

  template <int RankStart, int RankStop>
  auto GetKokkosMDRangePolicy(DevExecSpace exec_space) const {
    constexpr int ndim = RankStop - RankStart;
    static_assert(ndim > 1, "Need a valid range of ranks");
    static_assert(RankStart >= 0, "Need a valid range of ranks");
    static_assert(RankStop <= rank, "Need a valid range of ranks");
    Kokkos::Array<int64_t, ndim> start, end, tile;
    for (int d = 0; d < ndim; ++d) {
      start[d] = bounds[d + RankStart].s;
      end[d] = bounds[d + RankStart].e + 1;
      tile[d] = 1;
    }
    tile[ndim - 1] = end[ndim - 1] - start[ndim - 1];
    return Kokkos::Experimental::require(
        Kokkos::MDRangePolicy<Kokkos::Rank<ndim>>(exec_space, start, end, tile),
        Kokkos::Experimental::WorkItemProperty::HintLightWeight);
  }

  template <int RankStart, std::size_t... Is>
  KOKKOS_INLINE_FUNCTION auto GetIndexer(std::index_sequence<Is...>) const {
    return MakeIndexer(
        std::pair<int, int>(bounds[Is + RankStart].s, bounds[Is + RankStart].e)...);
  }

  template <int RankStart, int RankStop>
  KOKKOS_INLINE_FUNCTION auto GetIndexer() const {
    constexpr int ndim = RankStop - RankStart;
    static_assert(ndim > 0, "Need a valid range of ranks");
    static_assert(RankStart >= 0, "Need a valid range of ranks");
    static_assert(RankStop <= rank, "Need a valid range of ranks");
    return GetIndexer<RankStart>(std::make_index_sequence<ndim>());
  }
};

template <class... Bound_ts>
struct LoopBoundTranslator<TypeList<Bound_ts...>>
    : public LoopBoundTranslator<Bound_ts...> {};

} // namespace parthenon

#endif // LOOP_BOUND_TRANSLATOR_HPP_
