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

#include "interface/update.hpp"

#include <algorithm>
#include <limits>
#include <memory>

#include "config.hpp"
#include "coordinates/coordinates.hpp"
#include "interface/container.hpp"
#include "interface/metadata.hpp"
#include "mesh/mesh.hpp"

#include "kokkos_abstraction.hpp"
#include "mesh/meshblock_pack.hpp"

namespace parthenon {

namespace Update {
/*
template <typename T>
TaskStatus FluxDivergence(T *in_obj, T *dudt_obj) {

  const IndexDomain interior = IndexDomain::interior;
  const IndexRange ib = in_obj.cellbounds.GetBoundsI(interior);
  const IndexRange jb = in_obj.cellbounds.GetBoundsJ(interior);
  const IndexRange kb = in_obj.cellbounds.GetBoundsK(interior);

  auto vin = in_obj->PackVariablesAndFluxes({Metadata::Independent});
  auto dudt = dudt_obj->PackVariables({Metadata::Independent});

  auto &coords_array = in_obj->GetCoords();
  const int ndim = in_obj->GetNdim();

  parthenon::par_for(DEFAULT_LOOP_PATTERN, "flux divergence", in_obj->exec_space,
    0, vin.GetDim(5) - 1, 0, vin.GetDim(4) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
    KOKKOS_LAMBDA(const int m, const int l, const int k, const int j, const int i) {
      const auto coords = coords_array(m);
      const auto v = vin(m);
      dudt(m, l, k, j, i) =
          (coords.Area(X1DIR, k, j, i + 1) * v.flux(X1DIR, l, k, j, i + 1) -
           coords.Area(X1DIR, k, j, i) * v.flux(X1DIR, l, k, j, i));
      if (ndim >= 2) {
        dudt(m, l, k, j, i) +=
            (coords.Area(X2DIR, k, j + 1, i) * v.flux(X2DIR, l, k, j + 1, i) -
             coords.Area(X2DIR, k, j, i) * v.flux(X2DIR, l, k, j, i));
      }
      if (ndim == 3) {
        dudt(m, l, k, j, i) +=
            (coords.Area(X3DIR, k + 1, j, i) * v.flux(X3DIR, l, k + 1, j, i) -
             coords.Area(X3DIR, k, j, i) * v.flux(X3DIR, l, k, j, i));
      }
      dudt(m, l, k, j, i) /= -coords.Volume(k, j, i);
    });
  return TaskStatus::complete;
}
*/

TaskStatus FluxDivergenceBlock(std::shared_ptr<MeshBlockData<Real>> &in,
                               std::shared_ptr<MeshBlockData<Real>> &dudt_cont) {
  std::shared_ptr<MeshBlock> pmb = in->GetBlockPointer();

  const IndexDomain interior = IndexDomain::interior;
  IndexRange ib = pmb->cellbounds.GetBoundsI(interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(interior);

  auto vin = in->PackVariablesAndFluxes({Metadata::Independent});
  auto dudt = dudt_cont->PackVariables({Metadata::Independent});

  auto &coords = pmb->coords;
  int ndim = pmb->pmy_mesh->ndim;
  pmb->par_for(
      "flux divergence", 0, vin.GetDim(4) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int l, const int k, const int j, const int i) {
        dudt(l, k, j, i) = 0.0;
        dudt(l, k, j, i) +=
            (coords.Area(X1DIR, k, j, i + 1) * vin.flux(X1DIR, l, k, j, i + 1) -
             coords.Area(X1DIR, k, j, i) * vin.flux(X1DIR, l, k, j, i));
        if (ndim >= 2) {
          dudt(l, k, j, i) +=
              (coords.Area(X2DIR, k, j + 1, i) * vin.flux(X2DIR, l, k, j + 1, i) -
               coords.Area(X2DIR, k, j, i) * vin.flux(X2DIR, l, k, j, i));
        }
        if (ndim == 3) {
          dudt(l, k, j, i) +=
              (coords.Area(X3DIR, k + 1, j, i) * vin.flux(X3DIR, l, k + 1, j, i) -
               coords.Area(X3DIR, k, j, i) * vin.flux(X3DIR, l, k, j, i));
        }
        dudt(l, k, j, i) /= -coords.Volume(k, j, i);
      });

  return TaskStatus::complete;
}

auto FluxDivergenceMesh(const MeshBlockVarFluxPack<Real> &in_pack,
                        MeshBlockVarPack<Real> &dudt_pack) -> TaskStatus {
  const IndexDomain interior = IndexDomain::interior;
  const IndexRange ib = in_pack.cellbounds.GetBoundsI(interior);
  const IndexRange jb = in_pack.cellbounds.GetBoundsJ(interior);
  const IndexRange kb = in_pack.cellbounds.GetBoundsK(interior);

  auto ndim = in_pack.GetNdim();
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "flux divergence", DevExecSpace(), 0, in_pack.GetDim(5) - 1,
      0, in_pack.GetDim(4) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int l, const int k, const int j, const int i) {
        const auto coords = in_pack.coords(b);
        auto dudt = dudt_pack(b);
        const auto vin = in_pack(b);
        dudt(l, k, j, i) = 0.0;
        dudt(l, k, j, i) +=
            (coords.Area(X1DIR, k, j, i + 1) * vin.flux(X1DIR, l, k, j, i + 1) -
             coords.Area(X1DIR, k, j, i) * vin.flux(X1DIR, l, k, j, i));
        if (ndim >= 2) {
          dudt(l, k, j, i) +=
              (coords.Area(X2DIR, k, j + 1, i) * vin.flux(X2DIR, l, k, j + 1, i) -
               coords.Area(X2DIR, k, j, i) * vin.flux(X2DIR, l, k, j, i));
        }
        if (ndim == 3) {
          dudt(l, k, j, i) +=
              (coords.Area(X3DIR, k + 1, j, i) * vin.flux(X3DIR, l, k + 1, j, i) -
               coords.Area(X3DIR, k, j, i) * vin.flux(X3DIR, l, k, j, i));
        }
        dudt(l, k, j, i) /= -coords.Volume(k, j, i);
      });

  return TaskStatus::complete;
}

void UpdateContainer(std::shared_ptr<MeshBlockData<Real>> &in,
                     std::shared_ptr<MeshBlockData<Real>> &dudt_cont, const Real dt,
                     std::shared_ptr<MeshBlockData<Real>> &out) {
  std::shared_ptr<MeshBlock> pmb = in->GetBlockPointer();

  auto vin = in->PackVariables({Metadata::Independent});
  auto vout = out->PackVariables({Metadata::Independent});
  auto dudt = dudt_cont->PackVariables({Metadata::Independent});

  pmb->par_for(
      "UpdateContainer", 0, vin.GetDim(4) - 1, 0, vin.GetDim(3) - 1, 0, vin.GetDim(2) - 1,
      0, vin.GetDim(1) - 1,
      KOKKOS_LAMBDA(const int l, const int k, const int j, const int i) {
        vout(l, k, j, i) = vin(l, k, j, i) + dt * dudt(l, k, j, i);
      });
  return;
}

void UpdateContainer(const MeshBlockVarPack<Real> &in_pack,
                     const MeshBlockVarPack<Real> &dudt_pack, const Real dt,
                     MeshBlockVarPack<Real> &out_pack) {
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "UpdateContainer", DevExecSpace(), 0, in_pack.GetDim(5) - 1,
      0, in_pack.GetDim(4) - 1, 0, in_pack.GetDim(3) - 1, 0, in_pack.GetDim(2) - 1, 0,
      in_pack.GetDim(1) - 1,
      KOKKOS_LAMBDA(const int b, const int l, const int k, const int j, const int i) {
        out_pack(b, l, k, j, i) = in_pack(b, l, k, j, i) + dt * dudt_pack(b, l, k, j, i);
      });
}

void AverageContainers(MeshBlockVarPack<Real> &c1_pack,
                       const MeshBlockVarPack<Real> &c2_pack, const Real wgt1) {
  const IndexDomain interior = IndexDomain::interior;
  IndexRange ib = c1_pack.cellbounds.GetBoundsI(interior);
  IndexRange jb = c1_pack.cellbounds.GetBoundsJ(interior);
  IndexRange kb = c1_pack.cellbounds.GetBoundsK(interior);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "AverageContainer", DevExecSpace(), 0, c1_pack.GetDim(5) - 1,
      0, c1_pack.GetDim(4) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int l, const int k, const int j, const int i) {
        c1_pack(b, l, k, j, i) =
            wgt1 * c1_pack(b, l, k, j, i) + (1 - wgt1) * c2_pack(b, l, k, j, i);
      });
}

Real EstimateTimestep(std::shared_ptr<MeshBlockData<Real>> &rc) {
  std::shared_ptr<MeshBlock> pmb = rc->GetBlockPointer();
  Real dt_min = std::numeric_limits<Real>::max();
  for (auto &pkg : pmb->packages) {
    auto &desc = pkg.second;
    if (desc->EstimateTimestep != nullptr) {
      Real dt = desc->EstimateTimestep(rc);
      dt_min = std::min(dt_min, dt);
    }
  }
  return dt_min;
}

} // namespace Update

static FillDerivedVariables::FillDerivedFunc *pre_package_fill_ = nullptr;
static FillDerivedVariables::FillDerivedFunc *post_package_fill_ = nullptr;

void FillDerivedVariables::SetFillDerivedFunctions(FillDerivedFunc *pre,
                                                   FillDerivedFunc *post) {
  pre_package_fill_ = pre;
  post_package_fill_ = post;
}

TaskStatus FillDerivedVariables::FillDerived(std::shared_ptr<MeshBlockData<Real>> &rc) {
  if (pre_package_fill_ != nullptr) {
    pre_package_fill_(rc);
  }
  for (auto &pkg : rc->GetBlockPointer()->packages) {
    auto &desc = pkg.second;
    if (desc->FillDerived != nullptr) {
      desc->FillDerived(rc);
    }
  }
  if (post_package_fill_ != nullptr) {
    post_package_fill_(rc);
  }
  return TaskStatus::complete;
}

} // namespace parthenon
