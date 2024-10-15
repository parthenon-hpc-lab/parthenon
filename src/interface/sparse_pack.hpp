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
#include <iostream>
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
#include "interface/pack_utils.hpp"
#include "interface/sparse_pack_base.hpp"
#include "interface/variable.hpp"
#include "utils/concepts_lite.hpp"
#include "utils/type_list.hpp"
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
    SparsePack GetPack(T *pmd, std::vector<bool> &include_block,
                       bool only_fine_two_level_composite_blocks = true) const {
      // If this is a composite grid MeshData object and if
      // only_fine_two_level_composite_blocks is true, only
      // include blocks on the finer level
      if constexpr (std::is_same<T, MeshData<Real>>::value) {
        if (pmd->grid.type == GridType::two_level_composite &&
            only_fine_two_level_composite_blocks) {
          PARTHENON_REQUIRE(include_block.size() == pmd->NumBlocks(),
                            "Passed wrong size block include list.");
          int fine_level = pmd->grid.logical_level;
          for (int b = 0; b < pmd->NumBlocks(); ++b)
            include_block[b] =
                include_block[b] &&
                (fine_level == pmd->GetBlockData(b)->GetBlockPointer()->loc.level());
        }
      }
      return SparsePack(SparsePackBase::GetPack(pmd, *this, include_block));
    }

    template <class T>
    SparsePack GetPack(T *pmd, bool only_fine_two_level_composite_blocks = true) const {
      // If this is a composite grid MeshData object, only include blocks on
      // the finer level
      if constexpr (std::is_same<T, MeshData<Real>>::value) {
        if (pmd->grid.type == GridType::two_level_composite &&
            only_fine_two_level_composite_blocks) {
          auto include_block = std::vector<bool>(pmd->NumBlocks(), true);
          int fine_level = pmd->grid.logical_level;
          for (int b = 0; b < pmd->NumBlocks(); ++b)
            include_block[b] =
                include_block[b] &&
                (fine_level == pmd->GetBlockData(b)->GetBlockPointer()->loc.level());
          return SparsePack(SparsePackBase::GetPack(pmd, *this, include_block));
        } else {
          return SparsePack(SparsePackBase::GetPack(pmd, *this, std::vector<bool>{}));
        }
      }
      return SparsePack(SparsePackBase::GetPack(pmd, *this, std::vector<bool>{}));
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
  int GetLowerBoundHost(const int b) const {
    return (flat_ && (b > 0)) ? (bounds_h_(1, b - 1, nvar_) + 1) : 0;
  }

  int GetUpperBoundHost(const int b) const { return bounds_h_(1, b, nvar_); }

  int GetLowerBoundHost(const int b, PackIdx idx) const {
    static_assert(sizeof...(Ts) == 0);
    return bounds_h_(0, b, idx.VariableIdx());
  }

  int GetUpperBoundHost(const int b, PackIdx idx) const {
    static_assert(sizeof...(Ts) == 0);
    return bounds_h_(1, b, idx.VariableIdx());
  }

  template <class TIn, REQUIRES(IncludesType<TIn, Ts...>::value)>
  int GetLowerBoundHost(const int b, const TIn &) const {
    const int vidx = GetTypeIdx<TIn, Ts...>::value;
    return bounds_h_(0, b, vidx);
  }

  template <class TIn, REQUIRES(IncludesType<TIn, Ts...>::value)>
  int GetUpperBoundHost(const int b, const TIn &) const {
    const int vidx = GetTypeIdx<TIn, Ts...>::value;
    return bounds_h_(1, b, vidx);
  }

  KOKKOS_INLINE_FUNCTION int GetLevel(const int b, const int off3, const int off2,
                                      const int off1) const {
    return block_props_(b, (off3 + 1) + 3 * ((off2 + 1) + 3 * (off1 + 1)));
  }

  KOKKOS_INLINE_FUNCTION int GetGID(const int b) const { return block_props_(b, 27); }

  int GetLevelHost(const int b, const int off3, const int off2, const int off1) const {
    return block_props_h_(b, (off3 + 1) + 3 * ((off2 + 1) + 3 * (off1 + 1)));
  }

  int GetGIDHost(const int b) const { return block_props_h_(b, 27); }

  // Number of components of a variable on a block
  template <typename T>
  KOKKOS_INLINE_FUNCTION int GetSize(const int b, const T &t) const {
    return GetUpperBound(b, t) - GetLowerBound(b, t) + 1;
  }
  template <typename T>
  int GetSizeHost(const int b, const T &t) const {
    return GetUpperBoundHost(b, t) - GetLowerBoundHost(b, t) + 1;
  }

  // Index in pack
  template <typename TIn, REQUIRES(IncludesType<TIn, Ts...>::value)>
  KOKKOS_INLINE_FUNCTION int GetIndex(const int b, const TIn &var) const {
    return GetLowerBound(b, var) + var.idx;
  }
  template <typename TIn, REQUIRES(IncludesType<TIn, Ts...>::value)>
  int GetIndexHost(const int b, const TIn &var) const {
    return GetLowerBoundHost(b, var) + var.idx;
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
  bool ContainsHost(const int b) const { return GetUpperBoundHost(b) >= 0; }
  template <typename T>
  bool ContainsHost(const int b, const T t) const {
    return GetUpperBoundHost(b, t) >= 0;
  }
  template <typename... Args>
  bool ContainsHost(const int b, Args... args) const {
    return (... && ContainsHost(b, args));
  }
  template <typename... Args, REQUIRES(sizeof...(Args) > 0)>
  bool ContainsHost(const int b) const {
    return (... && ContainsHost(b, Args()));
  }

  // Informational
  auto LabelHost(int b, int idx) const { return pack_h_(0, b, idx).label(); }
  template <typename... Vars>
  friend std::ostream &operator<<(std::ostream &os, const SparsePack<Vars...> &sp);

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
  template <class... Args>
  KOKKOS_INLINE_FUNCTION auto &flux(const int b, const TopologicalElement te,
                                    Args &&...args) const {
    const int dir = (static_cast<int>(te) % 3) + 1;
    return flux(b, dir, std::forward<Args>(args)...);
  }

  template <class... Args>
  KOKKOS_INLINE_FUNCTION auto &flux(const TopologicalElement te, Args &&...args) const {
    const int dir = (static_cast<int>(te) % 3) + 1;
    return flux(dir, std::forward<Args>(args)...);
  }

  KOKKOS_INLINE_FUNCTION
  auto &flux(const int b, const int dir, const int idx) const {
    PARTHENON_DEBUG_REQUIRE(!flat_, "Accessor cannot be used for flat packs");
    PARTHENON_DEBUG_REQUIRE(dir > 0 && dir < 4 && with_fluxes_, "Bad input to flux call");
    return pack_(dir - 1 + flx_idx_, b, idx);
  }

  KOKKOS_INLINE_FUNCTION
  Real &flux(const int b, const int dir, const int idx, const int k, const int j,
             const int i) const {
    PARTHENON_DEBUG_REQUIRE(!flat_, "Accessor cannot be used for flat packs");
    PARTHENON_DEBUG_REQUIRE(dir > 0 && dir < 4 && with_fluxes_, "Bad input to flux call");
    return pack_(dir - 1 + flx_idx_, b, idx)(k, j, i);
  }

  KOKKOS_INLINE_FUNCTION
  Real &flux(const int dir, const int idx, const int k, const int j, const int i) const {
    PARTHENON_DEBUG_REQUIRE(flat_, "Accessor must only be used for flat packs");
    PARTHENON_DEBUG_REQUIRE(dir > 0 && dir < 4 && with_fluxes_, "Bad input to flux call");
    return pack_(dir - 1 + flx_idx_, 0, idx)(k, j, i);
  }

  KOKKOS_INLINE_FUNCTION
  Real &flux(const int b, const int dir, PackIdx idx, const int k, const int j,
             const int i) const {
    static_assert(sizeof...(Ts) == 0, "Cannot create a string/type hybrid pack");
    PARTHENON_DEBUG_REQUIRE(!flat_, "Accessor cannot be used for flat packs");
    PARTHENON_DEBUG_REQUIRE(dir > 0 && dir < 4 && with_fluxes_, "Bad input to flux call");
    const int n = bounds_(0, b, idx.VariableIdx()) + idx.Offset();
    return pack_(dir - 1 + flx_idx_, b, n)(k, j, i);
  }

  template <class TIn, REQUIRES(IncludesType<TIn, Ts...>::value)>
  KOKKOS_INLINE_FUNCTION Real &flux(const int b, const int dir, const TIn &t, const int k,
                                    const int j, const int i) const {
    PARTHENON_DEBUG_REQUIRE(!flat_, "Accessor cannot be used for flat packs");
    PARTHENON_DEBUG_REQUIRE(dir > 0 && dir < 4 && with_fluxes_, "Bad input to flux call");
    const int vidx = GetLowerBound(b, t) + t.idx;
    return pack_(dir - 1 + flx_idx_, b, vidx)(k, j, i);
  }

  template <class... VTs>
  KOKKOS_INLINE_FUNCTION auto GetPtrs(const int b, const TE el, int k, int j, int i,
                                      VTs... vts) const {
    return std::make_tuple(&(*this)(b, el, vts, k, j, i)...);
  }
};

template <typename... Vars>
inline std::ostream &operator<<(std::ostream &os, const SparsePack<Vars...> &sp) {
  os << "Sparse pack contains on each block:\n";
  for (int b = 0; b < sp.GetNBlocks(); b++) {
    os << "\tb = " << b << "\n";
    for (int n = sp.GetLowerBoundHost(b); n <= sp.GetUpperBoundHost(b); n++) {
      os << "\t\t" << sp.LabelHost(b, n) << "\n";
    }
  }
  os << "\n";
  return os;
}

} // namespace parthenon

#endif // INTERFACE_SPARSE_PACK_HPP_
