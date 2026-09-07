//========================================================================================
// (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
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
#ifndef LOOP_ABSTRACTION_SCRATCH_INDEXED_HPP_
#define LOOP_ABSTRACTION_SCRATCH_INDEXED_HPP_

// This file was made in part with generative AI.

// Type-indexed view over a flat per-point scratch buffer. IndexedVarTypeList defines
// the compile-time layout of a set of variable types within the buffer, and
// TypeIndexedPerPointScratch adapts a raw scratch buffer (from
// scratch.hpp) so it can be indexed by variable type, component, and
// an optional sparse/material index. Also provides the Add*Scratch entry points that
// register scratch sizing on an IndexSpace.

#include <utility>

#include "pack/pack_utils.hpp"

#include "loop_abstraction/scratch.hpp"

namespace parthenon::loop_abstraction {

// A compile-time list of variable types that indexes per-point scratch by variable
// type (and, optionally, material). Each variable occupies a contiguous block of
// components sized by that variable's size(); StartIdx gives the offset of a given
// variable's block within the flat scratch.
template <class... Var_Types>
struct IndexedVarTypeList {
  using var_types = parthenon::TypeList<Var_Types...>;

  template <class VarT>
  KOKKOS_INLINE_FUNCTION static constexpr auto StartIdx() {
    return SumSizesBefore<var_types, VarT>();
  }

  template <class VarT>
  KOKKOS_INLINE_FUNCTION static constexpr auto StartIdx(VarT) {
    return SumSizesBefore<var_types, VarT>();
  }

  KOKKOS_INLINE_FUNCTION static constexpr auto size() {
    return SumSizesBefore<var_types>();
  }
};

// Wraps a flat per-point scratch buffer so it can be indexed by variable type
// (field_tag), component (field_tag.idx), and (optionally) a sparse/material index,
// using the layout defined by VarTL.
template <class Scratch, class VarTL, int NSPARSE = 1>
class TypeIndexedPerPointScratch {
 public:
  KOKKOS_INLINE_FUNCTION
  explicit TypeIndexedPerPointScratch(Scratch scratch) : scratch_(std::move(scratch)) {}

  template <class Var, class Index>
    requires(NSPARSE == 1)
  KOKKOS_INLINE_FUNCTION decltype(auto) operator()(Var &&field_tag, Index &&index) const {
    return scratch_(VarTL::StartIdx(field_tag) + field_tag.idx,
                    std::forward<Index>(index));
  }

  template <class Var, class Index>
    requires(NSPARSE == 1)
  KOKKOS_INLINE_FUNCTION decltype(auto) operator()(TopologicalElement te, Var &&field_tag,
                                                   Index &&index) const {
    return scratch_(VarTL::StartIdx(field_tag) +
                        (static_cast<int>(te) % 3) * Var::size() + field_tag.idx,
                    std::forward<Index>(index));
  }

  template <class Var, class Index>
  KOKKOS_INLINE_FUNCTION decltype(auto) operator()(Var field_tag, int sparse_idx,
                                                   Index &&index) const {
    return scratch_(VarTL::StartIdx(field_tag) + field_tag.idx +
                        VarTL::size() * sparse_idx,
                    std::forward<Index>(index));
  }

  template <class Var, class Index>
  KOKKOS_INLINE_FUNCTION decltype(auto) operator()(TopologicalElement te, Var field_tag,
                                                   int sparse_idx, Index &&index) const {
    return scratch_(VarTL::StartIdx(field_tag) +
                        (static_cast<int>(te) % 3) * Var::size() + field_tag.idx +
                        VarTL::size() * sparse_idx,
                    std::forward<Index>(index));
  }

  Scratch &raw() { return scratch_; }
  const Scratch &raw() const { return scratch_; }

  KOKKOS_FORCEINLINE_FUNCTION void Zero() const { scratch_.Zero(); }

 private:
  Scratch scratch_;
};

// Hand out a type-indexed per-point scratch buffer sized for ReconTypes (times
// NSPARSE materials).
template <class Real, class ReconTypes, int NSPARSE = 1, class HaloRange>
KOKKOS_INLINE_FUNCTION auto GetTypeIndexedPerPointScratch(HaloRange &&halo_range) {
  auto scratch = GetPerPointScratch<Real, ReconTypes::size() * NSPARSE>(
      std::forward<HaloRange>(halo_range));
  using Scratch = decltype(scratch);
  return TypeIndexedPerPointScratch<Scratch, ReconTypes, NSPARSE>{std::move(scratch)};
}

template <class T, class Halo, class VarTL, int NSPARSE = 1, class IdxSpace>
void AddTypeIndexedPerPointScratch(IdxSpace &idx_space, int ncopies = 1) {
  idx_space.template AddPerPointScratch<T, Halo, VarTL::size() * NSPARSE>(ncopies);
}

template <class T, class Halo, int... Shape, class IdxSpace>
void AddPerPointScratch(IdxSpace &idx_space, int ncopies = 1) {
  idx_space.template AddPerPointScratch<T, Halo, Shape...>(ncopies);
}

template <class T, class VarTL, int NSPARSE = 1, class IdxSpace>
void AddTypeIndexedPerPointScratch(IdxSpace &idx_space, int ncopies = 1) {
  idx_space.template AddPerPointScratch<T, VarTL::size() * NSPARSE>(ncopies);
}

template <class T, int... Shape, class IdxSpace>
void AddPerPointScratch(IdxSpace &idx_space, int ncopies = 1) {
  idx_space.template AddPerPointScratch<T, Shape...>(ncopies);
}

} // namespace parthenon::loop_abstraction

#endif // LOOP_ABSTRACTION_SCRATCH_INDEXED_HPP_
