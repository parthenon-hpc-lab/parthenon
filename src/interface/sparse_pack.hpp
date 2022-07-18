//========================================================================================
// (C) (or copyright) 2020-2022. Triad National Security, LLC. All rights reserved.
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
#ifndef INTERFACE_SPARSE_PACK_HPP_
#define INTERFACE_SPARSE_PACK_HPP_

#include <algorithm>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <regex>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "coordinates/coordinates.hpp"
#include "interface/sparse_pack_base.hpp"
#include "interface/variable.hpp"
#include "utils/utils.hpp"

namespace parthenon {

namespace impl {
template <int... IN>
struct multiply;

template <>
struct multiply<> : std::integral_constant<std::size_t, 1> {};

template <int I0, int... IN>
struct multiply<I0, IN...> : std::integral_constant<int, I0 * multiply<IN...>::value> {};

// GetTypeIdx is taken from Stack Overflow 26169198, should cause compile time failure if
// type is not in list
template <typename T, typename... Ts>
struct GetTypeIdx;

template <typename T, typename... Ts>
struct GetTypeIdx<T, T, Ts...> : std::integral_constant<std::size_t, 0> {
  using type = void;
};

template <typename T, typename U, typename... Ts>
struct GetTypeIdx<T, U, Ts...>
    : std::integral_constant<std::size_t, 1 + GetTypeIdx<T, Ts...>::value> {
  using type = void;
};

template <class T, class... Ts>
struct IncludesType;

template <typename T>
struct IncludesType<T, T> : std::true_type {};

template <typename T, typename... Ts>
struct IncludesType<T, T, Ts...> : std::true_type {};

template <typename T, typename U>
struct IncludesType<T, U> : std::false_type {};

template <typename T, typename U, typename... Ts>
struct IncludesType<T, U, Ts...> : IncludesType<T, Ts...> {};

} // namespace impl

using namespace impl;

namespace variables {
// Struct that all variables types should inherit from
template <bool REGEX, int... NCOMP>
struct base_t {
  KOKKOS_INLINE_FUNCTION
  base_t() : idx(0) {}

  KOKKOS_INLINE_FUNCTION
  explicit base_t(int idx1) : idx(idx1) {}

  virtual ~base_t() = default;

  // All of these are just static methods so that there is no
  // extra storage in the struct
  static std::string name() {
    PARTHENON_FAIL("Need to implement your own name method.");
    return "error";
  }
  KOKKOS_INLINE_FUNCTION
  static bool regex() { return REGEX; }
  KOKKOS_INLINE_FUNCTION
  static int ndim() { return sizeof...(NCOMP); }
  KOKKOS_INLINE_FUNCTION
  static int size() { return multiply<NCOMP...>::value; }

  const int idx;
};

struct any : public base_t<true> {
  template <class... Ts>
  KOKKOS_INLINE_FUNCTION any(Ts &&...args)
      : parthenon::variables::base_t<true>(std::forward<Ts>(args)...) {}
  static std::string name() { return ".*"; }
};
} // namespace variables

template <class... Ts>
class SparsePack : public SparsePackBase {
 public:
  SparsePack() = default;

  explicit SparsePack(const SparsePackBase &spb) : SparsePackBase(spb) {}

  template <class T>
  static SparsePack Make(T *pmd, const std::vector<MetadataFlag> &flags = {},
                         bool fluxes = false, bool coarse = false) {
    auto &cache = pmd->GetSparsePackCache();
    return SparsePack(cache.Get(
        pmd, PackDescriptor(std::vector<std::string>{Ts::name()...},
                            std::vector<bool>{Ts::regex()...}, flags, fluxes, coarse)));
  }

  template <class T, class VAR_VEC>
  static std::tuple<SparsePack, SparsePackIdxMap>
  Make(T *pmd, const VAR_VEC &vars, const std::vector<MetadataFlag> &flags = {},
       bool fluxes = false, bool coarse = false) {
    static_assert(sizeof...(Ts) == 0);
    auto out = SparsePackBase::Make(pmd, vars, flags, fluxes, coarse);
    return {SparsePack(std::get<0>(out)), std::get<1>(out)};
  }

  template <class T>
  static SparsePack MakeWithFluxes(T *pmd, const std::vector<MetadataFlag> &flags = {}) {
    const bool coarse = false;
    const bool fluxes = true;
    return Make(pmd, flags, fluxes, coarse);
  }

  template <class T, class VAR_VEC>
  static std::tuple<SparsePack, SparsePackIdxMap>
  MakeWithFluxes(T *pmd, const VAR_VEC &vars,
                 const std::vector<MetadataFlag> &flags = {}) {
    const bool coarse = false;
    const bool fluxes = true;
    Make(pmd, vars, flags, fluxes, coarse);
  }

  template <class T>
  static SparsePack MakeWithCoarse(T *pmd, const std::vector<MetadataFlag> &flags = {}) {
    const bool coarse = true;
    const bool fluxes = false;
    return Make(pmd, flags, fluxes, coarse);
  }

  template <class T, class VAR_VEC>
  static std::tuple<SparsePack, SparsePackIdxMap>
  MakeWithCoarse(T *pmd, const VAR_VEC &vars,
                 const std::vector<MetadataFlag> &flags = {}) {
    const bool coarse = true;
    const bool fluxes = false;
    Make(pmd, vars, flags, fluxes, coarse);
  }

  KOKKOS_FORCEINLINE_FUNCTION
  int GetNBlocks() const { return nblocks_; }

  KOKKOS_FORCEINLINE_FUNCTION
  int GetNDim() const { return ndim_; }

  KOKKOS_FORCEINLINE_FUNCTION
  int GetDim(const int i) const {
    assert(i > 0 && i < 6);
    PARTHENON_REQUIRE(
        i != 2, "Should not ask for the second dimension since it is logically ragged");
    return dims_[i];
  }

  KOKKOS_INLINE_FUNCTION
  const Coordinates_t &GetCoordinates(const int b) const { return coords_(b)(); }

  // Bound overloads
  KOKKOS_INLINE_FUNCTION int GetLowerBound(const int b) const { return bounds_(0, b, 0); }

  KOKKOS_INLINE_FUNCTION int GetUpperBound(const int b) const {
    return bounds_(1, b, nvar_ - 1);
  }

  KOKKOS_INLINE_FUNCTION int GetLowerBound(const int b, PackIdx idx) const {
    static_assert(sizeof...(Ts) == 0);
    return bounds_(0, b, idx.Vidx());
  }

  KOKKOS_INLINE_FUNCTION int GetUpperBound(const int b, PackIdx idx) const {
    static_assert(sizeof...(Ts) == 0);
    return bounds_(1, b, idx.Vidx());
  }

  template <class TIn,
            class = typename std::enable_if<IncludesType<TIn, Ts...>::value>::type>
  KOKKOS_INLINE_FUNCTION int GetLowerBound(const int b, const TIn &) const {
    const int vidx = GetTypeIdx<TIn, Ts...>::value;
    return bounds_(0, b, vidx);
  }

  template <class TIn,
            class = typename std::enable_if<IncludesType<TIn, Ts...>::value>::type>
  KOKKOS_INLINE_FUNCTION int GetUpperBound(const int b, const TIn &) const {
    const int vidx = GetTypeIdx<TIn, Ts...>::value;
    return bounds_(1, b, vidx);
  }

  // operator() overloads
  KOKKOS_INLINE_FUNCTION
  auto &operator()(const int b, const int idx) const { return pack_(0, b, idx); }

  KOKKOS_INLINE_FUNCTION
  Real &operator()(const int b, const int idx, const int k, const int j,
                   const int i) const {
    return pack_(0, b, idx)(k, j, i);
  }

  KOKKOS_INLINE_FUNCTION
  Real &operator()(const int b, PackIdx idx, const int k, const int j,
                   const int i) const {
    static_assert(sizeof...(Ts) == 0);
    const int n = bounds_(0, b, idx.Vidx()) + idx.Off();
    return pack_(0, b, n)(k, j, i);
  }

  template <class TIn,
            class = typename std::enable_if<IncludesType<TIn, Ts...>::value>::type>
  KOKKOS_INLINE_FUNCTION Real &operator()(const int b, const TIn &t, const int k,
                                          const int j, const int i) const {
    const int vidx = GetLowerBound(b, t) + t.idx;
    return pack_(0, b, vidx)(k, j, i);
  }

  KOKKOS_INLINE_FUNCTION
  Real &operator()(const int dir, const int b, const int idx, const int k, const int j,
                   const int i) const {
    return pack_(dir, b, idx)(k, j, i);
  }

  KOKKOS_INLINE_FUNCTION
  Real &operator()(const int dir, const int b, PackIdx idx, const int k, const int j,
                   const int i) const {
    static_assert(sizeof...(Ts) == 0);
    const int n = bounds_(0, b, idx.Vidx()) + idx.Off();
    return pack_(dir, b, n)(k, j, i);
  }

  template <class TIn,
            class = typename std::enable_if<IncludesType<TIn, Ts...>::value>::type>
  KOKKOS_INLINE_FUNCTION Real &operator()(const int dir, const int b, const TIn &t, const int k,
                                          const int j, const int i) const {
    const int vidx = GetLowerBound(b, t) + t.idx;
    return pack_(dir, b, vidx)(k, j, i);
  }


  // flux() overloads
  KOKKOS_INLINE_FUNCTION
  auto &flux(const int b, const int dir, const int idx) const {
    PARTHENON_DEBUG_REQUIRE(dir > 0 && dir < 4 && with_fluxes_, "Bad input to flux call");
    return pack_(dir, b, idx);
  }

  KOKKOS_INLINE_FUNCTION
  Real &flux(const int b, const int dir, const int idx, const int k, const int j,
             const int i) const {
    PARTHENON_DEBUG_REQUIRE(dir > 0 && dir < 4 && with_fluxes_, "Bad input to flux call");
    return pack_(dir, b, idx)(k, j, i);
  }

  KOKKOS_INLINE_FUNCTION
  Real &flux(const int b, const int dir, PackIdx idx, const int k, const int j,
             const int i) const {
    static_assert(sizeof...(Ts) == 0);
    PARTHENON_DEBUG_REQUIRE(dir > 0 && dir < 4 && with_fluxes_, "Bad input to flux call");
    const int n = bounds_(0, b, idx.Vidx()) + idx.Off();
    return pack_(dir, b, n)(k, j, i);
  }

  // edge() overloads
  KOKKOS_INLINE_FUNCTION
  auto &edge(const int b, const int dir, const int idx) const {
    PARTHENON_DEBUG_REQUIRE(dir > 0 && dir < 4 && with_fluxes_, "Bad input to edge call");
    return pack_(dir, b, idx);
  }

  KOKKOS_INLINE_FUNCTION
  Real &edge(const int b, const int dir, const int idx, const int k, const int j,
             const int i) const {
    PARTHENON_DEBUG_REQUIRE(dir > 0 && dir < 4 && with_fluxes_, "Bad input to edge call");
    return pack_(dir, b, idx)(k, j, i);
  }

  KOKKOS_INLINE_FUNCTION
  Real &edge(const int b, const int dir, PackIdx idx, const int k, const int j,
             const int i) const {
    static_assert(sizeof...(Ts) == 0);
    PARTHENON_DEBUG_REQUIRE(dir > 0 && dir < 4 && with_fluxes_, "Bad input to edge call");
    const int n = bounds_(0, b, idx.Vidx()) + idx.Off();
    return pack_(dir, b, n)(k, j, i);
  }

  template <class TIn,
            class = typename std::enable_if<IncludesType<TIn, Ts...>::value>::type>
  KOKKOS_INLINE_FUNCTION Real &flux(const int b, const int dir, const TIn &t, const int k,
                                    const int j, const int i) const {
    PARTHENON_DEBUG_REQUIRE(dir > 0 && dir < 4 && with_fluxes_, "Bad input to flux call");
    const int vidx = GetLowerBound(b, t) + t.idx;
    return pack_(dir, b, vidx)(k, j, i);
  }
};

} // namespace parthenon

#endif // INTERFACE_SPARSE_PACK_HPP_
