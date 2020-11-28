//========================================================================================
// (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
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
#include "interface/meshblock_data.hpp"
#include "interface/meshblock_data_iterator.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"

namespace parthenon {

namespace boundary_cond_impl {
bool DoPhysicalBoundary_(const BoundaryFlag flag, const BoundaryFace face,
                         const int ndim);
} // namespace boundary_cond_impl

TaskStatus ProlongateBoundaries(std::shared_ptr<MeshBlockData<Real>> &rc) {
  if (!(rc->GetBlockPointer()->pmy_mesh->multilevel)) return TaskStatus::complete;
  Kokkos::Profiling::pushRegion("Task_ProlongateBoundaries");

  // This hardcoded technique is also used to manually specify the coupling between
  // physical variables in:
  // - step 2, ApplyPhysicalBoundariesOnCoarseLevel(): calls to W(U) and user BoundaryFunc
  // - step 3, ProlongateGhostCells(): calls to calculate bcc and U(W)

  // downcast BoundaryVariable pointers to known derived class pointer types:
  // RTTI via dynamic_case

  // For each finer neighbor, to prolongate a boundary we need to fill one more cell
  // surrounding the boundary zone to calculate the slopes ("ghost-ghost zone"). 3x steps:

  // Step 1. Apply necessary variable restrictions when ghost-ghost zone is on same lvl
  rc->RestrictBoundaries(); // Step 1: restrict physical boundaries

  // Step 2. Re-apply physical boundaries on the coarse boundary,
  ApplyBoundaryConditionsOnCoarseOrFine(rc, true);

  // Step 3. Finally, the ghost-ghost zones are ready for prolongation:
  rc->ProlongateBoundaries();

  Kokkos::Profiling::popRegion(); // Task_ProlongateBoundaries
  return TaskStatus::complete;
}

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

// TODO(JMM): These are all awfully similar. There's gotta be a way to
// save some code here.
void OutflowInnerX1(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
  std::shared_ptr<MeshBlock> pmb = rc->GetBlockPointer();
  auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
  int ref = bounds.GetBoundsI(IndexDomain::interior).s;
  auto q = rc->PackVariables(std::vector<MetadataFlag>{Metadata::FillGhost}, coarse);
  auto nb = IndexRange{0, q.GetDim(4) - 1};
  pmb->par_for_bndry(
      "OutflowInnerX1", nb, IndexDomain::inner_x1, coarse,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        q(l, k, j, i) = q(l, k, j, ref);
      });
}

void OutflowOuterX1(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
  std::shared_ptr<MeshBlock> pmb = rc->GetBlockPointer();
  auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
  int ref = bounds.GetBoundsI(IndexDomain::interior).e;
  auto q = rc->PackVariables(std::vector<MetadataFlag>{Metadata::FillGhost}, coarse);
  auto nb = IndexRange{0, q.GetDim(4) - 1};
  pmb->par_for_bndry(
      "OutflowOuterX1", nb, IndexDomain::outer_x1, coarse,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        q(l, k, j, i) = q(l, k, j, ref);
      });
}

void OutflowInnerX2(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
  std::shared_ptr<MeshBlock> pmb = rc->GetBlockPointer();
  auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
  int ref = bounds.GetBoundsJ(IndexDomain::interior).s;
  auto q = rc->PackVariables(std::vector<MetadataFlag>{Metadata::FillGhost}, coarse);
  auto nb = IndexRange{0, q.GetDim(4) - 1};
  pmb->par_for_bndry(
      "OutflowInnerX2", nb, IndexDomain::inner_x2, coarse,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        q(l, k, j, i) = q(l, k, ref, i);
      });
}

void OutflowOuterX2(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
  std::shared_ptr<MeshBlock> pmb = rc->GetBlockPointer();
  auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
  int ref = bounds.GetBoundsJ(IndexDomain::interior).e;
  auto q = rc->PackVariables(std::vector<MetadataFlag>{Metadata::FillGhost}, coarse);
  auto nb = IndexRange{0, q.GetDim(4) - 1};
  pmb->par_for_bndry(
      "OutflowOuterX2", nb, IndexDomain::outer_x2, coarse,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        q(l, k, j, i) = q(l, k, ref, i);
      });
}

void OutflowInnerX3(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
  std::shared_ptr<MeshBlock> pmb = rc->GetBlockPointer();
  auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
  int ref = bounds.GetBoundsJ(IndexDomain::interior).s;
  auto q = rc->PackVariables(std::vector<MetadataFlag>{Metadata::FillGhost}, coarse);
  auto nb = IndexRange{0, q.GetDim(4) - 1};
  pmb->par_for_bndry(
      "OutflowInnerX3", nb, IndexDomain::inner_x3, coarse,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        q(l, k, j, i) = q(l, ref, j, i);
      });
}

void OutflowOuterX3(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
  std::shared_ptr<MeshBlock> pmb = rc->GetBlockPointer();
  auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
  int ref = bounds.GetBoundsJ(IndexDomain::interior).e;
  auto q = rc->PackVariables(std::vector<MetadataFlag>{Metadata::FillGhost}, coarse);
  auto nb = IndexRange{0, q.GetDim(4) - 1};
  pmb->par_for_bndry(
      "OutflowOuterX3", nb, IndexDomain::outer_x3, coarse,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        q(l, k, j, i) = q(l, ref, j, i);
      });
}

void ReflectInnerX1(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
  std::shared_ptr<MeshBlock> pmb = rc->GetBlockPointer();
  auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
  int ref = bounds.GetBoundsI(IndexDomain::interior).s;
  auto q = rc->PackVariables(std::vector<MetadataFlag>{Metadata::FillGhost}, coarse);
  auto nb = IndexRange{0, q.GetDim(4) - 1};
  pmb->par_for_bndry(
      "ReflectInnerX1", nb, IndexDomain::inner_x1, coarse,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        Real reflect = q.VectorComponent(l) == X1DIR ? -1.0 : 1.0;
        q(l, k, j, i) = reflect * q(l, k, j, 2 * ref - i - 1);
      });
}

void ReflectOuterX1(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
  std::shared_ptr<MeshBlock> pmb = rc->GetBlockPointer();
  auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
  int ref = bounds.GetBoundsI(IndexDomain::interior).e;
  auto q = rc->PackVariables(std::vector<MetadataFlag>{Metadata::FillGhost}, coarse);
  auto nb = IndexRange{0, q.GetDim(4) - 1};
  pmb->par_for_bndry(
      "ReflectOuterX1", nb, IndexDomain::outer_x1, coarse,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        Real reflect = q.VectorComponent(l) == X1DIR ? -1.0 : 1.0;
        q(l, k, j, i) = reflect * q(l, k, j, 2 * ref - i + 1);
      });
}

void ReflectInnerX2(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
  std::shared_ptr<MeshBlock> pmb = rc->GetBlockPointer();
  auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
  int ref = bounds.GetBoundsJ(IndexDomain::interior).s;
  auto q = rc->PackVariables(std::vector<MetadataFlag>{Metadata::FillGhost}, coarse);
  auto nb = IndexRange{0, q.GetDim(4) - 1};
  pmb->par_for_bndry(
      "ReflectInnerX2", nb, IndexDomain::inner_x2, coarse,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        Real reflect = q.VectorComponent(l) == X2DIR ? -1.0 : 1.0;
        q(l, k, j, i) = reflect * q(l, k, 2 * ref - j - 1, i);
      });
}

void ReflectOuterX2(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
  std::shared_ptr<MeshBlock> pmb = rc->GetBlockPointer();
  auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
  int ref = bounds.GetBoundsJ(IndexDomain::interior).e;
  auto q = rc->PackVariables(std::vector<MetadataFlag>{Metadata::FillGhost}, coarse);
  auto nb = IndexRange{0, q.GetDim(4) - 1};
  pmb->par_for_bndry(
      "ReflectOuterX2", nb, IndexDomain::outer_x2, coarse,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        Real reflect = q.VectorComponent(l) == X2DIR ? -1.0 : 1.0;
        q(l, k, j, i) = reflect * q(l, k, 2 * ref - j + 1, i);
      });
}

void ReflectInnerX3(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
  std::shared_ptr<MeshBlock> pmb = rc->GetBlockPointer();
  auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
  int ref = bounds.GetBoundsK(IndexDomain::interior).s;
  auto q = rc->PackVariables(std::vector<MetadataFlag>{Metadata::FillGhost}, coarse);
  auto nb = IndexRange{0, q.GetDim(4) - 1};
  pmb->par_for_bndry(
      "ReflectInnerX3", nb, IndexDomain::inner_x3, coarse,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        Real reflect = q.VectorComponent(l) == X3DIR ? -1.0 : 1.0;
        q(l, k, j, i) = reflect * q(l, 2 * ref - k - 1, j, i);
      });
}

void ReflectOuterX3(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) {
  std::shared_ptr<MeshBlock> pmb = rc->GetBlockPointer();
  auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
  int ref = bounds.GetBoundsK(IndexDomain::interior).e;
  auto q = rc->PackVariables(std::vector<MetadataFlag>{Metadata::FillGhost}, coarse);
  auto nb = IndexRange{0, q.GetDim(4) - 1};
  pmb->par_for_bndry(
      "ReflectOuterX3", nb, IndexDomain::outer_x3, coarse,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        Real reflect = q.VectorComponent(l) == X3DIR ? -1.0 : 1.0;
        q(l, k, j, i) = reflect * q(l, 2 * ref - k + 1, j, i);
      });
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
} // namespace boundary_cond_impl

} // namespace parthenon
