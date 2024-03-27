//========================================================================================
// (C) (or copyright) 2023. Triad National Security, LLC. All rights reserved.
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
#ifndef UTILS_INDEXER_HPP_
#define UTILS_INDEXER_HPP_

#include <array>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

#include "mesh/forest/block_ownership.hpp"
#include "utils/concepts_lite.hpp"
#include "utils/utils.hpp"

namespace parthenon {

template <class... Ts>
struct Indexer {
  KOKKOS_INLINE_FUNCTION
  Indexer() : N{}, start{}, _size{} {};

  std::string GetRangesString() const {
    std::string out;
    for (int i = 0; i < sizeof...(Ts); ++i) {
      out += "[ " + std::to_string(start[i]) + ", " + std::to_string(end[i]) + "]";
    }
    return out;
  }

  KOKKOS_INLINE_FUNCTION
  explicit Indexer(std::pair<Ts, Ts>... Ns)
      : N{GetFactors(std::make_tuple((Ns.second - Ns.first + 1)...),
                     std::make_index_sequence<sizeof...(Ts)>())},
        start{Ns.first...}, end{Ns.second...}, _size(((Ns.second - Ns.first + 1) * ...)) {
  }

  KOKKOS_FORCEINLINE_FUNCTION
  std::size_t size() const { return _size; }

  KOKKOS_FORCEINLINE_FUNCTION
  std::tuple<Ts...> operator()(int idx) const {
    return GetIndicesImpl(idx, std::make_index_sequence<sizeof...(Ts)>());
  }

  KOKKOS_FORCEINLINE_FUNCTION
  auto GetIdxArray(int idx) const {
    return get_array_from_tuple(
        GetIndicesImpl(idx, std::make_index_sequence<sizeof...(Ts)>()));
  }

  template <std::size_t I>
  KOKKOS_FORCEINLINE_FUNCTION auto StartIdx() const {
    return std::get<I>(start);
  }

  template <std::size_t I>
  KOKKOS_FORCEINLINE_FUNCTION auto EndIdx() const {
    return std::get<I>(end);
  }

  static const constexpr std::size_t rank = sizeof...(Ts);

 protected:
  template <std::size_t... Is>
  KOKKOS_FORCEINLINE_FUNCTION std::tuple<Ts...>
  GetIndicesImpl(int idx, std::index_sequence<Is...>) const {
    std::tuple<Ts...> idxs;
    (
        [&] {
          std::get<Is>(idxs) = idx / std::get<Is>(N);
          idx -= std::get<Is>(idxs) * std::get<Is>(N);
          std::get<Is>(idxs) += std::get<Is>(start);
        }(),
        ...);
    return idxs;
  }

  template <std::size_t... Is>
  KOKKOS_FORCEINLINE_FUNCTION static std::array<int, sizeof...(Ts)>
  GetFactors(std::tuple<Ts...> Nt, std::index_sequence<Is...>) {
    std::array<int, sizeof...(Ts)> N;
    int cur = 1;
    (
        [&] {
          constexpr std::size_t idx = sizeof...(Ts) - (Is + 1);
          std::get<idx>(N) = cur;
          cur *= std::get<idx>(Nt);
        }(),
        ...);
    return N;
  }

  std::array<int, sizeof...(Ts)> N;
  std::array<int, sizeof...(Ts)> start;
  std::array<int, sizeof...(Ts)> end;
  std::size_t _size;
};

template <class... Ts>
class SpatiallyMaskedIndexer : public Indexer<Ts...> {
 public:
  KOKKOS_INLINE_FUNCTION
  SpatiallyMaskedIndexer() : Indexer<Ts...>(), active_() {}

  template <class... Args>
  KOKKOS_INLINE_FUNCTION SpatiallyMaskedIndexer(const block_ownership_t &active,
                                                std::pair<Ts, Ts>... Ns)
      : active_(active), Indexer<Ts...>(Ns...) {}

  KOKKOS_INLINE_FUNCTION
  bool IsActive(int k, int j, int i) const {
    const int istart = Indexer<Ts...>::start[sizeof...(Ts) - 1];
    const int iend = Indexer<Ts...>::end[sizeof...(Ts) - 1];
    const int jstart = Indexer<Ts...>::start[sizeof...(Ts) - 2];
    const int jend = Indexer<Ts...>::end[sizeof...(Ts) - 2];
    const int kstart = Indexer<Ts...>::start[sizeof...(Ts) - 3];
    const int kend = Indexer<Ts...>::end[sizeof...(Ts) - 3];
    const int iidx = (i == iend) - (i == istart);
    const int jidx = (j == jend) - (j == jstart);
    const int kidx = (k == kend) - (k == kstart);
    return active_(iidx, jidx, kidx);
  }

 private:
  block_ownership_t active_;
};

using Indexer1D = Indexer<int>;
using Indexer2D = Indexer<int, int>;
using Indexer3D = Indexer<int, int, int>;
using Indexer4D = Indexer<int, int, int, int>;
using Indexer5D = Indexer<int, int, int, int, int>;
using Indexer6D = Indexer<int, int, int, int, int, int>;
using Indexer7D = Indexer<int, int, int, int, int, int, int>;
using Indexer8D = Indexer<int, int, int, int, int, int, int, int>;

using SpatiallyMaskedIndexer6D = SpatiallyMaskedIndexer<int, int, int, int, int, int>;

} // namespace parthenon
#endif // UTILS_INDEXER_HPP_
