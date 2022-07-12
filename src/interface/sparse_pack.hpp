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
struct GetTypeIdx<T, T, Ts...> : std::integral_constant<std::size_t, 0> {};

template <typename T, typename U, typename... Ts>
struct GetTypeIdx<T, U, Ts...>
    : std::integral_constant<std::size_t, 1 + GetTypeIdx<T, Ts...>::value> {};
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

  template <class T>
  explicit SparsePack(T *pmd, const std::vector<MetadataFlag> &flags = {},
                      bool with_fluxes = false)
      : SparsePackBase(Build(pmd, GetDescriptor(flags, with_fluxes))) {}

  template <class T>
  SparsePack(T *pmd, SparsePackCache *pcache, const std::vector<MetadataFlag> &flags = {},
             bool with_fluxes = false)
      : SparsePackBase(pcache->Get(pmd, GetDescriptor(flags, with_fluxes))) {}

  explicit SparsePack(const SparsePackBase &spb) : SparsePackBase(spb) {}

  static PackDescriptor GetDescriptor(const std::vector<MetadataFlag> &flags,
                                      bool with_fluxes) {
    return PackDescriptor{std::vector<std::string>{Ts::name()...},
                          std::vector<bool>{Ts::regex()...}, flags, with_fluxes};
  }

  template <class T>
  static SparsePack Make(T *pmd, const std::vector<MetadataFlag> &flags = {}) {
    return MakeImpl(pmd, flags, false, static_cast<int>(0));
  }

  template <class T>
  static SparsePack MakeWithFluxes(T *pmd, const std::vector<MetadataFlag> &flags = {}) {
    return MakeImpl(pmd, flags, true, static_cast<int>(0));
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

  // This has to be defined here since the base class operator is apparently
  // covered by the template operator below even if std::enable_if fails
  KOKKOS_INLINE_FUNCTION
  Real &operator()(const int b, const int idx, const int k, const int j,
                   const int i) const {
    return pack_(0, b, idx)(k, j, i);
  }

  template <class TIn,
            class = typename std::enable_if<!std::is_integral<TIn>::value>::type>
  KOKKOS_INLINE_FUNCTION Real &operator()(const int b, const TIn &t, const int k,
                                          const int j, const int i) const {
    const int vidx = GetLowerBound(t, b) + t.idx;
    return pack_(0, b, vidx)(k, j, i);
  }

  KOKKOS_INLINE_FUNCTION
  Real &flux(const int b, const int dir, const int idx, const int k, const int j,
             const int i) const {
    PARTHENON_DEBUG_REQUIRE(dir > 0 && dir < 4 && with_fluxes_, "Bad input to flux call");
    return pack_(dir, b, idx)(k, j, i);
  }

  template <class TIn,
            class = typename std::enable_if<!std::is_integral<TIn>::value>::type>
  KOKKOS_INLINE_FUNCTION Real &flux(const int b, const int dir, const TIn &t, const int k,
                                    const int j, const int i) const {
    PARTHENON_DEBUG_REQUIRE(dir > 0 && dir < 4 && with_fluxes_, "Bad input to flux call");
    const int vidx = GetLowerBound(t, b) + t.idx;
    return pack_(dir, b, vidx)(k, j, i);
  }

 protected:
  template <class T>
  static auto MakeImpl(T *pmd, const std::vector<MetadataFlag> &flags, bool fluxes, int)
      -> decltype(T().GetSparsePackCache(), SparsePack()) {
    return SparsePack(pmd, &(pmd->GetSparsePackCache()), flags, fluxes);
  }

  template <class T>
  static SparsePack MakeImpl(T *pmd, const std::vector<MetadataFlag> &flags, bool fluxes,
                             double) {
    return SparsePack(pmd, flags, fluxes);
  }
};

} // namespace parthenon

#endif // INTERFACE_SPARSE_PACK_HPP_
