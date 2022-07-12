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
  KOKKOS_FUNCTION
  base_t() : idx(0) {}

  KOKKOS_FUNCTION
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
                         bool coarse = false, bool fluxes = false) {
    auto &cache = pmd->GetSparsePackCache();
    return SparsePack(cache.Get(
        pmd, PackDescriptor(std::vector<std::string>{Ts::name()...},
                            std::vector<bool>{Ts::regex()...}, flags, fluxes, coarse)));
  }

  template <class T>
  static SparsePack MakeWithFluxes(T *pmd, const std::vector<MetadataFlag> &flags = {}) {
    const bool coarse = false;
    const bool fluxes = true;
    return Make(pmd, flags, fluxes, coarse);
  }

  template <class T>
  static SparsePack MakeWithCoarse(T *pmd, const std::vector<MetadataFlag> &flags = {}) {
    const bool coarse = true;
    const bool fluxes = false;
    return Make(pmd, flags, fluxes, coarse);
  }

  template <class TIn>
  KOKKOS_INLINE_FUNCTION int GetLowerBound(const int b, const TIn &) const {
    const int vidx = GetTypeIdx<TIn, Ts...>::value;
    return bounds_(0, b, vidx);
  }

  template <class TIn>
  KOKKOS_INLINE_FUNCTION int GetUpperBound(const int b, const TIn &) const {
    const int vidx = GetTypeIdx<TIn, Ts...>::value;
    return bounds_(1, b, vidx);
  }

  template <class TIn,
            class = typename std::enable_if<IncludesType<TIn, Ts...>::value>::type>
  KOKKOS_INLINE_FUNCTION Real &operator()(const int b, const TIn &t, const int k,
                                          const int j, const int i) const {
    const int vidx = GetLowerBound(t, b) + t.idx;
    return pack_(0, b, vidx)(k, j, i);
  }

  template <class TIn,
            class = typename std::enable_if<IncludesType<TIn, Ts...>::value>::type>
  KOKKOS_INLINE_FUNCTION Real &flux(const int b, const int dir, const TIn &t, const int k,
                                    const int j, const int i) const {
    PARTHENON_DEBUG_REQUIRE(dir > 0 && dir < 4 && with_fluxes_, "Bad input to flux call");
    const int vidx = GetLowerBound(t, b) + t.idx;
    return pack_(dir, b, vidx)(k, j, i);
  }

  // This has to be defined here since the base class operator is apparently
  // covered by the template operator below even if std::enable_if fails
  KOKKOS_INLINE_FUNCTION
  Real &operator()(const int b, const int idx, const int k, const int j,
                   const int i) const {
    return pack_(0, b, idx)(k, j, i);
  }

  KOKKOS_INLINE_FUNCTION
  Real &flux(const int b, const int dir, const int idx, const int k, const int j,
             const int i) const {
    PARTHENON_DEBUG_REQUIRE(dir > 0 && dir < 4 && with_fluxes_, "Bad input to flux call");
    return pack_(dir, b, idx)(k, j, i);
  }
};

} // namespace parthenon

#endif // INTERFACE_SPARSE_PACK_HPP_
