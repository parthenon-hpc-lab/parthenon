//========================================================================================
// (C) (or copyright) 2020-2024. Triad National Security, LLC. All rights reserved.
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
#ifndef INTERFACE_SWARM_PACK_HPP_
#define INTERFACE_SWARM_PACK_HPP_

#include <algorithm>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <regex>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "coordinates/coordinates.hpp"
#include "interface/mesh_data.hpp"
#include "interface/meshblock_data.hpp"
#include "interface/swarm.hpp"
#include "interface/swarm_pack_base.hpp"
#include "interface/variable.hpp"
#include "utils/concepts_lite.hpp"
#include "utils/utils.hpp"

namespace parthenon {

class SwarmPackIdx {
 public:
  KOKKOS_INLINE_FUNCTION
  explicit SwarmPackIdx(std::size_t var_idx) : vidx(var_idx), offset(0) {}
  KOKKOS_INLINE_FUNCTION
  SwarmPackIdx(std::size_t var_idx, int off) : vidx(var_idx), offset(off) {}

  KOKKOS_INLINE_FUNCTION
  SwarmPackIdx &operator=(std::size_t var_idx) {
    vidx = var_idx;
    offset = 0;
    return *this;
  }

  KOKKOS_INLINE_FUNCTION
  std::size_t VariableIdx() { return vidx; }
  KOKKOS_INLINE_FUNCTION
  int Offset() { return offset; }

 private:
  std::size_t vidx;
  int offset;
};

// Operator overloads to make calls like `my_pack(b, my_pack_idx + 3, k, j, i)` work
template <class T, REQUIRES(std::is_integral<T>::value)>
KOKKOS_INLINE_FUNCTION SwarmPackIdx operator+(SwarmPackIdx idx, T offset) {
  return SwarmPackIdx(idx.VariableIdx(), idx.Offset() + offset);
}

template <class T, REQUIRES(std::is_integral<T>::value)>
KOKKOS_INLINE_FUNCTION SwarmPackIdx operator+(T offset, SwarmPackIdx idx) {
  return idx + offset;
}

// Namespace in which to put variable name types that are used for indexing into
// SwarmPack<[type list of variable name types]> on device
namespace swarm_variable_names {
// Struct that all variable_name types should inherit from
template <bool REGEX, typename T = Real, int... NCOMP>
struct base_t {
  using data_type = T;

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

} // namespace swarm_variable_names

template <typename TYPE, class... Ts>
class SwarmPack : public SwarmPackBase<TYPE> {
 public:
  using typename SwarmPackBase<TYPE>::pack_t;
  using typename SwarmPackBase<TYPE>::desc_t;
  using typename SwarmPackBase<TYPE>::bounds_t;
  using typename SwarmPackBase<TYPE>::contexts_t;
  using typename SwarmPackBase<TYPE>::max_active_indices_t;

  using SwarmPackBase<TYPE>::pack_;
  using SwarmPackBase<TYPE>::bounds_;
  using SwarmPackBase<TYPE>::contexts_;
  using SwarmPackBase<TYPE>::max_active_indices_;
  using SwarmPackBase<TYPE>::nvar_;
  using SwarmPackBase<TYPE>::nblocks_;

  explicit SwarmPack(const SwarmPackBase<TYPE> &spb) : SwarmPackBase<TYPE>(spb) {
    if constexpr (sizeof...(Ts) != 0) {
      static_assert(std::is_same<TYPE, typename GetDataType<Ts...>::value>::value,
                    "Type mismatch in SwarmPack! When passing type-based variables as "
                    "template argument to SwarmPack, ensure that the first template "
                    "parameter is a data type (e.g., Real or int) that matches the "
                    "data type of subsequent variable types!");
    }
  }

  class Descriptor : public impl::SwarmPackDescriptor<TYPE> {
   public:
    Descriptor() = default;
    explicit Descriptor(const impl::SwarmPackDescriptor<TYPE> &desc_in)
        : impl::SwarmPackDescriptor<TYPE>(desc_in) {}

    template <class T>
    SwarmPack GetPack(T *pmd) const {
      return SwarmPack(SwarmPackBase<TYPE>::GetPack(pmd, *this));
    }

    SparsePackIdxMap GetMap() const {
      PARTHENON_REQUIRE(sizeof...(Ts) == 0,
                        "Should not be getting an IdxMap for a type based pack");
      return SwarmPackBase<TYPE>::GetIdxMap(*this);
    }
  };

  const SwarmDeviceContext &GetContext(const int b = 0) const { return contexts_(b); }
  const int &GetMaxActiveIndex(const int b = 0) const { return max_active_indices_(b); }

  // Methods for getting parts of the shape of the pack
  KOKKOS_FORCEINLINE_FUNCTION
  int GetNBlocks() const { return nblocks_; }

  // Bound overloads
  KOKKOS_INLINE_FUNCTION int GetLowerBound(const int b) const { return bounds_(0, b, 0); }

  KOKKOS_INLINE_FUNCTION int GetUpperBound(const int b) const {
    return bounds_(1, b, nvar_ - 1);
  }

  KOKKOS_INLINE_FUNCTION int GetLowerBound(const int b, SwarmPackIdx idx) const {
    static_assert(sizeof...(Ts) == 0, "Cannot create a string/type hybrid pack");
    return bounds_(0, b, idx.VariableIdx());
  }

  KOKKOS_INLINE_FUNCTION int GetUpperBound(const int b, SwarmPackIdx idx) const {
    static_assert(sizeof...(Ts) == 0, "Cannot create a string/type hybrid pack");
    return bounds_(1, b, idx.VariableIdx());
  }

  template <class TIn, REQUIRES(IncludesType<TIn, Ts...>::value)>
  KOKKOS_INLINE_FUNCTION int GetLowerBound(const int b, const TIn &) const {
    const int vidx = GetTypeIdx<TIn, Ts...>::value;
    return bounds_(0, b, vidx);
  }

  template <class TIn, REQUIRES(IncludesType<TIn, Ts...>::value)>
  KOKKOS_INLINE_FUNCTION int GetUpperBound(const int b, const TIn &) const {
    const int vidx = GetTypeIdx<TIn, Ts...>::value;
    return bounds_(1, b, vidx);
  }

  // operator() overloads
  KOKKOS_INLINE_FUNCTION
  auto &operator()(const int b, const int idx) const { return pack_(0, b, idx); }

  KOKKOS_INLINE_FUNCTION
  auto &operator()(const int b, const int idx, const int n) const {
    return pack_(0, b, idx)(n);
  }

  template <class TIn, REQUIRES(IncludesType<TIn, Ts...>::value)>
  KOKKOS_INLINE_FUNCTION auto &operator()(const int b, const TIn &t, const int n) const {
    const int vidx = GetLowerBound(b, t) + t.idx;
    return pack_(0, b, vidx)(n);
  }
};

} // namespace parthenon

#endif // INTERFACE_SWARM_PACK_HPP_
