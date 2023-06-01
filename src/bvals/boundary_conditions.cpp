//========================================================================================
// (C) (or copyright) 2020-2023. Triad National Security, LLC. All rights reserved.
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

#include <memory>
#include <vector>

#include "bvals/boundary_conditions.hpp"
#include "bvals/bvals_interfaces.hpp"
#include "defs.hpp"
#include "interface/meshblock_data.hpp"
#include "mesh/domain.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "mesh/meshblock.hpp"

namespace parthenon {

namespace boundary_cond_impl {
bool DoPhysicalBoundary_(const BoundaryFlag flag, const BoundaryFace face,
                         const int ndim);
} // namespace boundary_cond_impl

TaskStatus ApplyBoundaryConditionsOnCoarseOrFine(std::shared_ptr<MeshBlockData<Real>> &rc,
                                                 bool coarse) {
  Kokkos::Profiling::pushRegion("Task_ApplyBoundaryConditionsOnCoarseOrFine");
  using namespace boundary_cond_impl;
  std::shared_ptr<MeshBlock> pmb = rc->GetBlockPointer();
  Mesh *pmesh = pmb->pmy_mesh;
  const int ndim = pmesh->ndim;

  for (int i = 0; i < BOUNDARY_NFACES; i++) {
    if (DoPhysicalBoundary_(pmb->boundary_flag[i], static_cast<BoundaryFace>(i), ndim)) {
      PARTHENON_DEBUG_REQUIRE(pmesh->MeshBndryFnctn[i] != nullptr,
                              "boundary function must not be null");
      pmesh->MeshBndryFnctn[i](rc, coarse);
    }
  }

  Kokkos::Profiling::popRegion(); // Task_ApplyBoundaryConditionsOnCoarseOrFine
  return TaskStatus::complete;
}

namespace BoundaryFunction {

enum class BCSide { Inner, Outer };
enum class BCType { Outflow, Reflect };

template <CoordinateDirection DIR, BCSide SIDE, BCType TYPE>
void GenericBC(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse, TopologicalElement el = TopologicalElement::CC) {
  // make sure DIR is X[123]DIR so we don't have to check again
  static_assert(DIR == X1DIR || DIR == X2DIR || DIR == X3DIR, "DIR must be X[123]DIR");

  // convenient shorthands
  constexpr bool X1 = (DIR == X1DIR);
  constexpr bool X2 = (DIR == X2DIR);
  constexpr bool X3 = (DIR == X3DIR);
  constexpr bool INNER = (SIDE == BCSide::Inner);

  std::shared_ptr<MeshBlock> pmb = rc->GetBlockPointer();
  const auto &bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;

  const auto &range = X1 ? bounds.GetBoundsI(IndexDomain::interior, el)
                         : (X2 ? bounds.GetBoundsJ(IndexDomain::interior, el)
                               : bounds.GetBoundsK(IndexDomain::interior, el));
  const int ref = INNER ? range.s : range.e;
  
  std::vector<MetadataFlag> flags{Metadata::FillGhost}; 
  if (GetTopologicalType(el) == TopologicalType::Cell) flags.push_back(Metadata::Cell);
  if (GetTopologicalType(el) == TopologicalType::Face) flags.push_back(Metadata::Face);
  if (GetTopologicalType(el) == TopologicalType::Edge) flags.push_back(Metadata::Edge);
  if (GetTopologicalType(el) == TopologicalType::Node) flags.push_back(Metadata::Node);

  auto q = rc->PackVariables(flags, coarse);
  auto nb = IndexRange{0, q.GetDim(4) - 1};

  std::string label = (TYPE == BCType::Reflect ? "Reflect" : "Outflow");
  label += (INNER ? "Inner" : "Outer");
  label += "X" + std::to_string(DIR);

  constexpr IndexDomain domain =
      INNER ? (X1 ? IndexDomain::inner_x1
                  : (X2 ? IndexDomain::inner_x2 : IndexDomain::inner_x3))
            : (X1 ? IndexDomain::outer_x1
                  : (X2 ? IndexDomain::outer_x2 : IndexDomain::outer_x3));

  // used for reflections
  const int offset = 2 * ref + (INNER ? -1 : 1);

  pmb->par_for_bndry(
      label, nb, domain, el, coarse,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        if (!q.IsAllocated(l)) return;
        if (TYPE == BCType::Reflect) {
          const bool reflect = (q.VectorComponent(l) == DIR);
          q(el, l, k, j, i) =
              (reflect ? -1.0 : 1.0) *
              q(el, l, X3 ? offset - k : k, X2 ? offset - j : j, X1 ? offset - i : i);
        } else {
          q(el, l, k, j, i) = q(el, l, X3 ? ref : k, X2 ? ref : j, X1 ? ref : i);
        }
      });
}

void OutflowInnerX1(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
  using TE = TopologicalElement;
  for (auto el : {TE::CC, TE::F1, TE::F2, TE::F3, TE::E1, TE::E2, TE::E3, TE::NN})
    GenericBC<X1DIR, BCSide::Inner, BCType::Outflow>(rc, coarse, el);
}

void OutflowOuterX1(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
  using TE = TopologicalElement;
  for (auto el : {TE::CC, TE::F1, TE::F2, TE::F3, TE::E1, TE::E2, TE::E3, TE::NN})
    GenericBC<X1DIR, BCSide::Outer, BCType::Outflow>(rc, coarse, el);
}

void OutflowInnerX2(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
  using TE = TopologicalElement;
  for (auto el : {TE::CC, TE::F1, TE::F2, TE::F3, TE::E1, TE::E2, TE::E3, TE::NN})
    GenericBC<X2DIR, BCSide::Inner, BCType::Outflow>(rc, coarse, el);
}

void OutflowOuterX2(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
  using TE = TopologicalElement;
  for (auto el : {TE::CC, TE::F1, TE::F2, TE::F3, TE::E1, TE::E2, TE::E3, TE::NN})
    GenericBC<X2DIR, BCSide::Outer, BCType::Outflow>(rc, coarse, el);
}

void OutflowInnerX3(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
  using TE = TopologicalElement;
  for (auto el : {TE::CC, TE::F1, TE::F2, TE::F3, TE::E1, TE::E2, TE::E3, TE::NN})
    GenericBC<X3DIR, BCSide::Inner, BCType::Outflow>(rc, coarse, el);
}

void OutflowOuterX3(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
  using TE = TopologicalElement;
  for (auto el : {TE::CC, TE::F1, TE::F2, TE::F3, TE::E1, TE::E2, TE::E3, TE::NN})
    GenericBC<X3DIR, BCSide::Outer, BCType::Outflow>(rc, coarse, el);
}

void ReflectInnerX1(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
  using TE = TopologicalElement;
  for (auto el : {TE::CC, TE::F1, TE::F2, TE::F3, TE::E1, TE::E2, TE::E3, TE::NN})
    GenericBC<X1DIR, BCSide::Inner, BCType::Reflect>(rc, coarse, el);
}

void ReflectOuterX1(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
  using TE = TopologicalElement;
  for (auto el : {TE::CC, TE::F1, TE::F2, TE::F3, TE::E1, TE::E2, TE::E3, TE::NN})
    GenericBC<X1DIR, BCSide::Outer, BCType::Reflect>(rc, coarse, el);
}

void ReflectInnerX2(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
  using TE = TopologicalElement;
  for (auto el : {TE::CC, TE::F1, TE::F2, TE::F3, TE::E1, TE::E2, TE::E3, TE::NN})
    GenericBC<X2DIR, BCSide::Inner, BCType::Reflect>(rc, coarse, el);
}

void ReflectOuterX2(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
  using TE = TopologicalElement;
  for (auto el : {TE::CC, TE::F1, TE::F2, TE::F3, TE::E1, TE::E2, TE::E3, TE::NN})
    GenericBC<X2DIR, BCSide::Outer, BCType::Reflect>(rc, coarse, el);
}

void ReflectInnerX3(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
  using TE = TopologicalElement;
  for (auto el : {TE::CC, TE::F1, TE::F2, TE::F3, TE::E1, TE::E2, TE::E3, TE::NN})
    GenericBC<X3DIR, BCSide::Inner, BCType::Reflect>(rc, coarse, el);
}

void ReflectOuterX3(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
  using TE = TopologicalElement;
  for (auto el : {TE::CC, TE::F1, TE::F2, TE::F3, TE::E1, TE::E2, TE::E3, TE::NN})
    GenericBC<X3DIR, BCSide::Outer, BCType::Reflect>(rc, coarse, el);
}

} // namespace BoundaryFunction

namespace boundary_cond_impl {
bool DoPhysicalBoundary_(const BoundaryFlag flag, const BoundaryFace face,
                         const int ndim) {
  if (flag == BoundaryFlag::block) return false;
  if (flag == BoundaryFlag::undef) return false;
  if (flag == BoundaryFlag::periodic) return false;

  if (ndim < 3 && (face == BoundaryFace::inner_x3 || face == BoundaryFace::outer_x3)) {
    return false;
  }
  if (ndim < 2 && (face == BoundaryFace::inner_x2 || face == BoundaryFace::outer_x2)) {
    return false;
  } // ndim always at least 1

  return true; // reflect, outflow, user, dims correct
}

void ProlongateGhostCells_(std::shared_ptr<MeshBlockData<Real>> &rc,
                           const NeighborBlock &nb, int si, int ei, int sj, int ej,
                           int sk, int ek) {
  std::shared_ptr<MeshBlock> pmb = rc->GetBlockPointer();
  auto &pmr = pmb->pmr;

  for (auto var : rc->GetVariableVector()) {
    if (!var->IsAllocated()) continue;
    if (!(var->IsSet(Metadata::Independent) || var->IsSet(Metadata::FillGhost))) continue;

    if (var->IsSet(Metadata::Cell)) {
      pmr->ProlongateCellCenteredValues(var.get(), si, ei, sj, ej, sk, ek);
    } else {
      PARTHENON_FAIL("Prolongation not implemented for non-cell centered variables.");
    }
  }

  // TODO(LFR): Deal with prolongation of non-cell centered values
}

void ComputeProlongationBounds_(const std::shared_ptr<MeshBlock> &pmb,
                                const NeighborBlock &nb, IndexRange &bi, IndexRange &bj,
                                IndexRange &bk) {
  const IndexDomain interior = IndexDomain::interior;
  int cn = pmb->cnghost - 1;

  auto getbounds = [=](const int nbx, const std::int64_t &lx, const IndexRange bblock,
                       IndexRange &bprol) {
    if (nbx == 0) {
      bprol.s = bblock.s;
      bprol.e = bblock.e;
      if ((lx & 1LL) == 0LL) {
        bprol.e += cn;
      } else {
        bprol.s -= cn;
      }
    } else if (nbx > 0) {
      bprol.s = bblock.e + 1;
      bprol.e = bblock.e + cn;
    } else {
      bprol.s = bblock.s - cn;
      bprol.e = bblock.s - 1;
    }
  };

  getbounds(nb.ni.ox1, pmb->loc.lx1, pmb->c_cellbounds.GetBoundsI(interior), bi);
  getbounds(nb.ni.ox2, pmb->loc.lx2, pmb->c_cellbounds.GetBoundsJ(interior), bj);
  getbounds(nb.ni.ox3, pmb->loc.lx3, pmb->c_cellbounds.GetBoundsK(interior), bk);
}

} // namespace boundary_cond_impl

} // namespace parthenon
