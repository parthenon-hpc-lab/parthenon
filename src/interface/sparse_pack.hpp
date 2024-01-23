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
#ifndef INTERFACE_SPARSE_PACK_HPP_
#define INTERFACE_SPARSE_PACK_HPP_

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
#include "interface/sparse_pack_base.hpp"
#include "interface/variable.hpp"
#include "utils/concepts_lite.hpp"
#include "utils/utils.hpp"

namespace parthenon {

KOKKOS_INLINE_FUNCTION
IndexShape GetIndexShape(const ParArray3D<Real, VariableState> &arr, int ng) {
  int extra_zone = std::max(TopologicalOffsetJ(arr.topological_element),
                            TopologicalOffsetK(arr.topological_element));
  extra_zone = std::max(extra_zone, TopologicalOffsetI(arr.topological_element));
  int nx1 = arr.GetDim(1) > 1 ? arr.GetDim(1) - extra_zone - 2 * ng : 0;
  int nx2 = arr.GetDim(2) > 1 ? arr.GetDim(2) - extra_zone - 2 * ng : 0;
  int nx3 = arr.GetDim(3) > 1 ? arr.GetDim(3) - extra_zone - 2 * ng : 0;
  return IndexShape::GetFromSeparateInts(nx3, nx2, nx1, ng);
}

// Sparse pack index type which allows for relatively simple indexing
// into non-variable name type based SparsePacks (i.e. objects of
// type SparsePack<> which are created with a vector of variable
// names and/or regexes)
class PackIdx {
 public:
  KOKKOS_INLINE_FUNCTION
  explicit PackIdx(std::size_t var_idx) : vidx(var_idx), offset(0) {}
  KOKKOS_INLINE_FUNCTION
  PackIdx(std::size_t var_idx, int off) : vidx(var_idx), offset(off) {}

  KOKKOS_INLINE_FUNCTION
  PackIdx &operator=(std::size_t var_idx) {
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
KOKKOS_INLINE_FUNCTION PackIdx operator+(PackIdx idx, T offset) {
  return PackIdx(idx.VariableIdx(), idx.Offset() + offset);
}

template <class T, REQUIRES(std::is_integral<T>::value)>
KOKKOS_INLINE_FUNCTION PackIdx operator+(T offset, PackIdx idx) {
  return idx + offset;
}

// Namespace in which to put variable name types that are used for
// indexing into SparsePack<[type list of variable name types]> on
// device
namespace variable_names {
// Struct that all variable_name types should inherit from
constexpr int ANYDIM = -1234;
template <bool REGEX, int... NCOMP>
struct base_t {
  KOKKOS_INLINE_FUNCTION
  base_t() : idx(0) {}

  KOKKOS_INLINE_FUNCTION
  explicit base_t(int idx1) : idx(idx1) {}

  template <typename... Args, REQUIRES(all_implement<integral(Args...)>::value)>
  /*
    for 2D:, (M, N),
    idx(m, n) = N*m + n
    for 3D: (L, M, N)
    idx(l, m, n) = (M*l + m)*N + n
                 = l*M*N + m*N + n
   */
  KOKKOS_INLINE_FUNCTION explicit base_t(Args... args)
      : idx(GetIndex_<NCOMP...>(std::forward<Args>(args)...)) {}

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

 private:
  template <int Head, int... Tail, typename... TailArgs,
            REQUIRES(all_implement<integral(TailArgs...)>::value)>
  KOKKOS_INLINE_FUNCTION static auto GetIndex_(int first, TailArgs... rest) {
    if constexpr (sizeof...(Tail) == 0)
      return first;
    else if constexpr (sizeof...(TailArgs) == 0)
      return first;
    else
      return (multiply<Tail...>::value * first +
              GetIndex_<Tail...>(std::forward<TailArgs>(rest)...));
  }
};

// An example variable name type that selects all variables available
// on Mesh*Data
struct any : public base_t<true> {
  template <class... Ts>
  KOKKOS_INLINE_FUNCTION any(Ts &&...args) : base_t<true>(std::forward<Ts>(args)...) {}
  static std::string name() { return ".*"; }
};
} // namespace variable_names

template <class... Ts>
class SparsePack : public SparsePackBase {
 public:
  SparsePack() = default;

  explicit SparsePack(const SparsePackBase &spb) : SparsePackBase(spb) {}

  class Descriptor : public impl::PackDescriptor {
   public:
    Descriptor() = default;
    explicit Descriptor(const impl::PackDescriptor &desc_in)
        : impl::PackDescriptor(desc_in) {}

    // Make a `SparsePack` from variable_name types in the type list Ts..., creating the
    // pack in `pmd->SparsePackCache` if it doesn't already exist. Variables can be
    // accessed on device via instance of types in the type list Ts...
    // The pack will be created and accessible on the device
    template <class T>
    SparsePack GetPack(T *pmd, std::vector<bool> include_block = {},
                       bool only_fine_two_level_composite_blocks = true) const {
      // If this is a composite grid MeshData object and if
      // only_fine_two_level_composite_blocks is true, only
      // include blocks on the finer level
      if constexpr (std::is_same<T, MeshData<Real>>::value) {
        if (pmd->grid.type == GridType::two_level_composite &&
            only_fine_two_level_composite_blocks) {
          if (include_block.size() != pmd->NumBlocks()) {
            include_block = std::vector<bool>(pmd->NumBlocks(), true);
          }
          int fine_level = pmd->grid.logical_level;
          for (int b = 0; b < pmd->NumBlocks(); ++b)
            include_block[b] =
                include_block[b] &&
                (fine_level == pmd->GetBlockData(b)->GetBlockPointer()->loc.level());
        }
      }
      return SparsePack(SparsePackBase::GetPack(pmd, *this, include_block));
    }

    SparsePackIdxMap GetMap() const {
      PARTHENON_REQUIRE(sizeof...(Ts) == 0,
                        "Should not be getting an IdxMap for a type based pack");
      return SparsePackBase::GetIdxMap(*this);
    }
  };

  // Methods for getting parts of the shape of the pack
  KOKKOS_FORCEINLINE_FUNCTION
  int GetNBlocks() const { return nblocks_; }

  KOKKOS_FORCEINLINE_FUNCTION
  int GetMaxNumberOfVars() const { return pack_.extent_int(2); }

  // Returns the total number of vars/components in pack
  KOKKOS_FORCEINLINE_FUNCTION
  int GetSize() const { return size_; }

  KOKKOS_INLINE_FUNCTION
  const Coordinates_t &GetCoordinates(const int b = 0) const { return coords_(b)(); }

  // Bound overloads
  KOKKOS_INLINE_FUNCTION int GetLowerBound(const int b) const {
    return (flat_ && (b > 0)) ? (bounds_(1, b - 1, nvar_) + 1) : 0;
  }

  KOKKOS_INLINE_FUNCTION int GetUpperBound(const int b) const {
    return bounds_(1, b, nvar_);
  }

  KOKKOS_INLINE_FUNCTION int GetLowerBound(const int b, PackIdx idx) const {
    static_assert(sizeof...(Ts) == 0, "Cannot create a string/type hybrid pack");
    return bounds_(0, b, idx.VariableIdx());
  }

  KOKKOS_INLINE_FUNCTION int GetUpperBound(const int b, PackIdx idx) const {
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

  // Host Bound overloads
  KOKKOS_INLINE_FUNCTION int GetLowerBoundHost(const int b) const {
    return (flat_ && (b > 0)) ? (bounds_h_(1, b - 1, nvar_) + 1) : 0;
  }

  KOKKOS_INLINE_FUNCTION int GetUpperBoundHost(const int b) const {
    return bounds_h_(1, b, nvar_);
  }

  KOKKOS_INLINE_FUNCTION int GetLowerBoundHost(const int b, PackIdx idx) const {
    static_assert(sizeof...(Ts) == 0);
    return bounds_h_(0, b, idx.VariableIdx());
  }

  KOKKOS_INLINE_FUNCTION int GetUpperBoundHost(const int b, PackIdx idx) const {
    static_assert(sizeof...(Ts) == 0);
    return bounds_h_(1, b, idx.VariableIdx());
  }

  template <class TIn, REQUIRES(IncludesType<TIn, Ts...>::value)>
  KOKKOS_INLINE_FUNCTION int GetLowerBoundHost(const int b, const TIn &) const {
    const int vidx = GetTypeIdx<TIn, Ts...>::value;
    return bounds_h_(0, b, vidx);
  }

  template <class TIn, REQUIRES(IncludesType<TIn, Ts...>::value)>
  KOKKOS_INLINE_FUNCTION int GetUpperBoundHost(const int b, const TIn &) const {
    const int vidx = GetTypeIdx<TIn, Ts...>::value;
    return bounds_h_(1, b, vidx);
  }

  /* Usage:
   * Contains(b, v1(), v2(), v3())
   *
   * returns true if all listed vars are present on block b, false
   * otherwise.
   */
  KOKKOS_INLINE_FUNCTION bool Contains(const int b) const {
    return GetUpperBound(b) >= 0;
  }
  template <typename T>
  KOKKOS_INLINE_FUNCTION bool Contains(const int b, const T t) const {
    return GetUpperBound(b, t) >= 0;
  }
  template <typename... Args>
  KOKKOS_INLINE_FUNCTION bool Contains(const int b, Args... args) const {
    return (... && Contains(b, args));
  }
  // Version that takes templates but no arguments passed
  template <typename... Args, REQUIRES(sizeof...(Args) > 0)>
  KOKKOS_INLINE_FUNCTION bool Contains(const int b) const {
    return (... && Contains(b, Args()));
  }
  // Host versions
  KOKKOS_INLINE_FUNCTION bool ContainsHost(const int b) const {
    return GetUpperBoundHost(b) >= 0;
  }
  template <typename T>
  KOKKOS_INLINE_FUNCTION bool ContainsHost(const int b, const T t) const {
    return GetUpperBoundHost(b, t) >= 0;
  }
  template <typename... Args>
  KOKKOS_INLINE_FUNCTION bool ContainsHost(const int b, Args... args) const {
    return (... && ContainsHost(b, args));
  }
  template <typename... Args, REQUIRES(sizeof...(Args) > 0)>
  KOKKOS_INLINE_FUNCTION bool ContainsHost(const int b) const {
    return (... && ContainsHost(b, Args()));
  }

  // operator() overloads
  using TE = TopologicalElement;
  KOKKOS_INLINE_FUNCTION auto &operator()(const int b, const TE el, const int idx) const {
    return pack_(static_cast<int>(el) % 3, b, idx);
  }
  KOKKOS_INLINE_FUNCTION auto &operator()(const int b, const int idx) const {
    PARTHENON_DEBUG_REQUIRE(pack_(0, b, idx).topological_type == TopologicalType::Cell,
                            "Suppressed topological element index assumes that this is a "
                            "cell variable, but it isn't");
    return pack_(0, b, idx);
  }

  KOKKOS_INLINE_FUNCTION auto &operator()(const int b, const TE el, PackIdx idx) const {
    static_assert(sizeof...(Ts) == 0);
    PARTHENON_DEBUG_REQUIRE(!flat_, "Accessor cannot be used for flat packs");
    const int n = bounds_(0, b, idx.VariableIdx()) + idx.Offset();
    return pack_(static_cast<int>(el) % 3, b, n);
  }
  KOKKOS_INLINE_FUNCTION auto &operator()(const int b, PackIdx idx) const {
    return (*this)(b, TE::CC, idx);
  }

  template <class TIn, REQUIRES(IncludesType<TIn, Ts...>::value)>
  KOKKOS_INLINE_FUNCTION auto &operator()(const int b, const TE el, const TIn &t) const {
    const int vidx = GetLowerBound(b, t) + t.idx;
    return pack_(static_cast<int>(el) % 3, b, vidx);
  }

  template <class TIn, REQUIRES(IncludesType<TIn, Ts...>::value)>
  KOKKOS_INLINE_FUNCTION auto &operator()(const int b, const TIn &t) const {
    PARTHENON_DEBUG_REQUIRE(!flat_, "Accessor cannot be used for flat packs");
    const int vidx = GetLowerBound(b, t) + t.idx;
    return pack_(0, b, vidx);
  }

  KOKKOS_INLINE_FUNCTION
  Real &operator()(const int b, const int idx, const int k, const int j,
                   const int i) const {
    PARTHENON_DEBUG_REQUIRE(!flat_, "Accessor cannot be used for flat packs");
    return pack_(0, b, idx)(k, j, i);
  }
  KOKKOS_INLINE_FUNCTION
  Real &operator()(int idx, const int k, const int j, const int i) const {
    PARTHENON_DEBUG_REQUIRE(flat_, "Accessor valid only for flat packs");
    return pack_(0, 0, idx)(k, j, i);
  }

  KOKKOS_INLINE_FUNCTION Real &operator()(const int b, const TE el, const int idx,
                                          const int k, const int j, const int i) const {
    return pack_(static_cast<int>(el) % 3, b, idx)(k, j, i);
  }

  KOKKOS_INLINE_FUNCTION Real &operator()(const int b, PackIdx idx, const int k,
                                          const int j, const int i) const {
    static_assert(sizeof...(Ts) == 0, "Cannot create a string/type hybrid pack");
    PARTHENON_DEBUG_REQUIRE(!flat_, "Accessor cannot be used for flat packs");
    const int n = bounds_(0, b, idx.VariableIdx()) + idx.Offset();
    return pack_(0, b, n)(k, j, i);
  }

  KOKKOS_INLINE_FUNCTION Real &operator()(const int b, const TE el, PackIdx idx,
                                          const int k, const int j, const int i) const {
    static_assert(sizeof...(Ts) == 0, "Cannot create a string/type hybrid pack");
    const int n = bounds_(0, b, idx.VariableIdx()) + idx.Offset();
    return pack_(static_cast<int>(el) % 3, b, n)(k, j, i);
  }

  template <class TIn, REQUIRES(IncludesType<TIn, Ts...>::value)>
  KOKKOS_INLINE_FUNCTION Real &operator()(const int b, const TIn &t, const int k,
                                          const int j, const int i) const {
    PARTHENON_DEBUG_REQUIRE(!flat_, "Accessor cannot be used for flat packs");
    const int vidx = GetLowerBound(b, t) + t.idx;
    return pack_(0, b, vidx)(k, j, i);
  }

  template <class TIn, REQUIRES(IncludesType<TIn, Ts...>::value)>
  KOKKOS_INLINE_FUNCTION Real &operator()(const int b, const TE el, const TIn &t,
                                          const int k, const int j, const int i) const {
    const int vidx = GetLowerBound(b, t) + t.idx;
    return pack_(static_cast<int>(el) % 3, b, vidx)(k, j, i);
  }

  // flux() overloads
  KOKKOS_INLINE_FUNCTION
  auto &flux(const int b, const int dir, const int idx) const {
    PARTHENON_DEBUG_REQUIRE(!flat_, "Accessor cannot be used for flat packs");
    PARTHENON_DEBUG_REQUIRE(dir > 0 && dir < 4 && with_fluxes_, "Bad input to flux call");
    return pack_(dir, b, idx);
  }

  KOKKOS_INLINE_FUNCTION
  Real &flux(const int b, const int dir, const int idx, const int k, const int j,
             const int i) const {
    PARTHENON_DEBUG_REQUIRE(!flat_, "Accessor cannot be used for flat packs");
    PARTHENON_DEBUG_REQUIRE(dir > 0 && dir < 4 && with_fluxes_, "Bad input to flux call");
    return pack_(dir, b, idx)(k, j, i);
  }

  KOKKOS_INLINE_FUNCTION
  Real &flux(const int dir, const int idx, const int k, const int j, const int i) const {
    PARTHENON_DEBUG_REQUIRE(flat_, "Accessor must only be used for flat packs");
    PARTHENON_DEBUG_REQUIRE(dir > 0 && dir < 4 && with_fluxes_, "Bad input to flux call");
    return pack_(dir, 0, idx)(k, j, i);
  }

  KOKKOS_INLINE_FUNCTION
  Real &flux(const int b, const int dir, PackIdx idx, const int k, const int j,
             const int i) const {
    static_assert(sizeof...(Ts) == 0, "Cannot create a string/type hybrid pack");
    PARTHENON_DEBUG_REQUIRE(!flat_, "Accessor cannot be used for flat packs");
    PARTHENON_DEBUG_REQUIRE(dir > 0 && dir < 4 && with_fluxes_, "Bad input to flux call");
    const int n = bounds_(0, b, idx.VariableIdx()) + idx.Offset();
    return pack_(dir, b, n)(k, j, i);
  }

  template <class TIn, REQUIRES(IncludesType<TIn, Ts...>::value)>
  KOKKOS_INLINE_FUNCTION Real &flux(const int b, const int dir, const TIn &t, const int k,
                                    const int j, const int i) const {
    PARTHENON_DEBUG_REQUIRE(!flat_, "Accessor cannot be used for flat packs");
    PARTHENON_DEBUG_REQUIRE(dir > 0 && dir < 4 && with_fluxes_, "Bad input to flux call");
    const int vidx = GetLowerBound(b, t) + t.idx;
    return pack_(dir, b, vidx)(k, j, i);
  }

  template <class... VTs>
  KOKKOS_INLINE_FUNCTION auto GetPtrs(const int b, const TE el, int k, int j, int i,
                                      VTs... vts) const {
    return std::make_tuple(&(*this)(b, el, vts, k, j, i)...);
  }
};

} // namespace parthenon

#endif // INTERFACE_SPARSE_PACK_HPP_
