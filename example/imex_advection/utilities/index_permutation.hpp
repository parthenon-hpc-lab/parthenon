//========================================================================================
// (C) (or copyright) 2020-2024. Triad National Security, LLC. All rights
// reserved.
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
#ifndef SRC_UTILITIES_INDEX_PERMUTATION_HPP_
#define SRC_UTILITIES_INDEX_PERMUTATION_HPP_

#include <parthenon/parthenon.hpp>

namespace parthenon {
namespace utils {

struct IndexingData : public IndexShape {
  int ndim;
  IndexShape cellbounds;
  IndexingData(MeshData<Real> *md)
      : ndim{md->GetMeshPointer()->ndim}, IndexShape(md->GetBlockData(0)->GetBlockPointer()->cellbounds) {}

  KOKKOS_INLINE_FUNCTION
  auto Get3DIndexRangeWithHalo(IndexDomain id, std::array<int, 3> halo,
                               TopologicalElement te = TopologicalElement::CC) const {
    IndexRange ib = GetBoundsI(id, te);
    IndexRange jb = GetBoundsJ(id, te);
    IndexRange kb = GetBoundsK(id, te);
    kb.s -= halo[0] * (ndim > 2);
    kb.e += halo[0] * (ndim > 2);
    jb.s -= halo[1] * (ndim > 1);
    jb.e += halo[1] * (ndim > 1);
    ib.s -= halo[2] * (ndim > 0);
    ib.e += halo[2] * (ndim > 0);
    return std::make_tuple(kb, jb, ib);
  }

  KOKKOS_INLINE_FUNCTION
  auto Get3DIndexRange(IndexDomain id, TopologicalElement te = TopologicalElement::CC) const {
    return Get3DIndexRangeWithHalo(id, {0, 0, 0}, te);
  }

  KOKKOS_INLINE_FUNCTION
  auto GetReconstructionRange(TopologicalElement flux_te) const {
    return Get3DIndexRangeWithHalo(IndexDomain::interior, GetOffsetArray(flux_te));
  }

  KOKKOS_INLINE_FUNCTION
  std::array<int, 3> GetOffsetArray(TopologicalElement flux_te) const {
    return std::array<int, 3>{TopologicalOffsetK(flux_te) * (ndim > 2), TopologicalOffsetJ(flux_te) * (ndim > 1),
                              TopologicalOffsetI(flux_te) * (ndim > 0)};
  }
};

inline auto Get3DIndexRangeWithHalo(MeshData<Real> *md, IndexDomain id, std::array<int, 3> halo,
                                    TopologicalElement te = TopologicalElement::CC) {
  const int ndim = md->GetMeshPointer()->ndim;
  IndexRange ib = md->GetBoundsI(id, te);
  IndexRange jb = md->GetBoundsJ(id, te);
  IndexRange kb = md->GetBoundsK(id, te);
  kb.s -= halo[0] * (ndim > 2);
  kb.e += halo[0] * (ndim > 2);
  jb.s -= halo[1] * (ndim > 1);
  jb.e += halo[1] * (ndim > 1);
  ib.s -= halo[2] * (ndim > 0);
  ib.e += halo[2] * (ndim > 0);
  return std::make_tuple(kb, jb, ib);
}

inline auto Get3DIndexRange(MeshData<Real> *md, IndexDomain id, TopologicalElement te = TopologicalElement::CC) {
  return Get3DIndexRangeWithHalo(md, id, {0, 0, 0}, te);
}

inline CoordinateDirection PermuteDirection(CoordinateDirection base_dir, CoordinateDirection relative_dir) {
  return static_cast<CoordinateDirection>(((base_dir - 1 + relative_dir - 1) % 3) + 1);
}

inline auto GetOffsetsForDirection(MeshData<Real> *md, CoordinateDirection dir) {
  const int ndim = md->GetMeshPointer()->ndim;
  int di = (dir == CoordinateDirection::X1DIR) * (ndim > 0);
  int dj = (dir == CoordinateDirection::X2DIR) * (ndim > 1);
  int dk = (dir == CoordinateDirection::X3DIR) * (ndim > 2);
  return std::make_tuple(dk, dj, di);
}

inline auto GetPermutedOffsetsForRelativeDirection(MeshData<Real> *md, CoordinateDirection base_dir,
                                                   CoordinateDirection relative_dir) {
  auto absolute_dir = PermuteDirection(base_dir, relative_dir);
  return GetOffsetsForDirection(md, absolute_dir);
}

inline auto GetDirection(TopologicalElement te) {
  using TE = TopologicalElement;
  if (te == TE::F1 || te == TE::E1)
    return CoordinateDirection::X1DIR;
  else if (te == TE::F2 || te == TE::E2)
    return CoordinateDirection::X2DIR;
  else if (te == TE::F3 || te == TE::E3)
    return CoordinateDirection::X3DIR;
  PARTHENON_FAIL("It makes no sense to be asking for the direction of a node or cell "
                 "topological element.");
  return CoordinateDirection::NODIR;
}

inline auto GetPermutedTE(CoordinateDirection base_dir, TopologicalElement te) {
  const auto absolute_dir = PermuteDirection(base_dir, GetDirection(te));
  return GetTopologicalElements(GetTopologicalType(te))[absolute_dir - 1];
}

} // namespace utils
} // namespace parthenon

#endif // SRC_UTILITIES_INDEX_PERMUTATION_HPP_
