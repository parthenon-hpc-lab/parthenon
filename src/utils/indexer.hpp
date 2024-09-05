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

#include <Kokkos_Core.hpp>

#include "utils/concepts_lite.hpp"
#include "utils/type_list.hpp"
#include "utils/utils.hpp"

namespace parthenon {

struct block_ownership_t {
 public:
  KOKKOS_FORCEINLINE_FUNCTION
  const bool &operator()(int ox1, int ox2, int ox3) const {
    return ownership[ox1 + 1][ox2 + 1][ox3 + 1];
  }
  KOKKOS_FORCEINLINE_FUNCTION
  bool &operator()(int ox1, int ox2, int ox3) {
    return ownership[ox1 + 1][ox2 + 1][ox3 + 1];
  }

  KOKKOS_FORCEINLINE_FUNCTION
  block_ownership_t() : block_ownership_t(false) {}

  KOKKOS_FORCEINLINE_FUNCTION
  explicit block_ownership_t(bool value) : initialized(false) {
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 3; ++k) {
          ownership[i][j][k] = value;
        }
      }
    }
  }

  bool initialized;

  bool operator==(const block_ownership_t &rhs) const {
    bool same = initialized == rhs.initialized;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 3; ++k) {
          same = same && (ownership[i][j][k] == rhs.ownership[i][j][k]);
        }
      }
    }
    return same;
  }

 private:
  bool ownership[3][3][3];
};
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
      : N{GetFactors({(Ns.second - Ns.first + 1)...},
                     std::make_index_sequence<sizeof...(Ts)>())},
        start{Ns.first...}, end{Ns.second...}, _size(((Ns.second - Ns.first + 1) * ...)) {
  }

  template <class... IndRngs>
  KOKKOS_INLINE_FUNCTION explicit Indexer(IndRngs... Ns)
      : N{GetFactors({(Ns.e - Ns.s + 1)...}, std::make_index_sequence<sizeof...(Ts)>())},
        start{Ns.s...}, end{Ns.e...}, _size(((Ns.e - Ns.s + 1) * ...)) {}

  KOKKOS_FORCEINLINE_FUNCTION std::size_t size() const { return _size; }

  KOKKOS_FORCEINLINE_FUNCTION
  std::tuple<Ts...> operator()(int idx) const {
    return GetIndicesImpl(idx, std::make_index_sequence<sizeof...(Ts)>());
  }

  KOKKOS_FORCEINLINE_FUNCTION
  auto GetIdxArray(int idx) const {
    return GetIndicesKArrayImpl(idx, std::make_index_sequence<sizeof...(Ts)>());
  }

  template <std::size_t I>
  KOKKOS_FORCEINLINE_FUNCTION auto StartIdx() const {
    return start[I];
  }

  template <std::size_t I>
  KOKKOS_FORCEINLINE_FUNCTION auto EndIdx() const {
    return end[I];
  }

  static const constexpr std::size_t rank = sizeof...(Ts);

 protected:
  template <std::size_t... Is>
  KOKKOS_FORCEINLINE_FUNCTION std::tuple<Ts...>
  GetIndicesImpl(int idx, std::index_sequence<Is...>) const {
    std::tuple<Ts...> idxs;
    (
        [&] {
          std::get<Is>(idxs) = idx / N[Is];
          idx -= std::get<Is>(idxs) * N[Is];
          std::get<Is>(idxs) += start[Is];
        }(),
        ...);
    return idxs;
  }

  template <std::size_t... Is>
  KOKKOS_FORCEINLINE_FUNCTION Kokkos::Array<int, sizeof...(Ts)>
  GetIndicesKArrayImpl(int idx, std::index_sequence<Is...>) const {
    Kokkos::Array<int, sizeof...(Ts)> indices;
    (
        [&] {
          indices[Is] = idx / N[Is];
          idx -= indices[Is] * N[Is];
          indices[Is] += start[Is];
        }(),
        ...);
    return indices;
  }

  template <std::size_t... Is>
  KOKKOS_FORCEINLINE_FUNCTION static Kokkos::Array<int, sizeof...(Ts)>
  GetFactors(Kokkos::Array<int, sizeof...(Ts)> Nt, std::index_sequence<Is...>) {
    Kokkos::Array<int, sizeof...(Ts)> N;
    int cur = 1;
    (
        [&] {
          constexpr std::size_t idx = sizeof...(Ts) - (Is + 1);
          N[idx] = cur;
          cur *= Nt[idx];
        }(),
        ...);
    return N;
  }

  Kokkos::Array<int, sizeof...(Ts)> N;
  Kokkos::Array<int, sizeof...(Ts)> start;
  Kokkos::Array<int, sizeof...(Ts)> end;
  std::size_t _size;
};

template <class... Ts>
struct IndexRanger {
  KOKKOS_INLINE_FUNCTION
  IndexRanger() : N{}, _size{} {};

  KOKKOS_INLINE_FUNCTION
  explicit IndexRanger(Ts... IdrsA) {}

  Kokkos::Array<IndexRange, sizeof...(Ts)> N;
  std::size_t _size;
};

template <>
struct Indexer<> {
  // this is a dummy and shouldn't ever actually get used to index an array
  KOKKOS_FORCEINLINE_FUNCTION
  Kokkos::Array<int, 1> GetIdxArray(int idx) const { return {-1}; }
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

template <class... Ts>
KOKKOS_FORCEINLINE_FUNCTION auto MakeIndexer(const std::pair<Ts, Ts> &...ranges) {
  return Indexer<Ts...>(ranges...);
}

template <std::size_t NIdx, class... Ts, std::size_t... Is>
KOKKOS_FORCEINLINE_FUNCTION auto MakeIndexer(TypeList<Ts...>,
                                             Kokkos::Array<IndexRange, NIdx> bounds_arr,
                                             std::integer_sequence<std::size_t, Is...>) {
  return Indexer<Ts...>(bounds_arr[Is]...);
}

template <std::size_t NIdx>
KOKKOS_FORCEINLINE_FUNCTION auto MakeIndexer(Kokkos::Array<IndexRange, NIdx> bounds_arr) {
  return MakeIndexer(list_of_type_t<NIdx, IndexRange>(), bounds_arr,
                     std::make_index_sequence<NIdx>());
}

} // namespace parthenon
#endif // UTILS_INDEXER_HPP_
