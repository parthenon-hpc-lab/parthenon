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
#include "mesh/mesh_pack.hpp"

namespace parthenon {

namespace Update {

TaskStatus FluxDivergence(std::shared_ptr<Container<Real>> &in,
                          std::shared_ptr<Container<Real>> &dudt_cont) {
  MeshBlock *pmb = in->pmy_block;

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

auto FluxDivergenceMesh(std::vector<MeshBlock *> &blocks, const std::string &in_cont,
                        const std::string &dudt_cont) -> TaskStatus {
  auto pack_in = parthenon::PackVariablesAndFluxesOnMesh(
      blocks, in_cont, std::vector<MetadataFlag>{Metadata::Independent});
  auto pack_dudt = parthenon::PackVariablesOnMesh(
      blocks, dudt_cont, std::vector<MetadataFlag>{Metadata::Independent});

  const IndexDomain interior = IndexDomain::interior;
  const IndexRange ib = pack_in.cellbounds.GetBoundsI(interior);
  const IndexRange jb = pack_in.cellbounds.GetBoundsJ(interior);
  const IndexRange kb = pack_in.cellbounds.GetBoundsK(interior);

  int ndim = blocks[0]->pmy_mesh->ndim;
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "flux divergence", DevExecSpace(), 0, blocks.size() - 1, 0,
      pack_in.GetDim(4) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int l, const int k, const int j, const int i) {
        const auto coords = pack_in.coords(b);
        auto dudt = pack_dudt(b);
        const auto vin = pack_in(b);
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

void UpdateContainer(std::vector<MeshBlock *> &blocks, const std::string &in_cont_name,
                     const std::string &dudt_cont_name, const Real dt,
                     const std::string &out_cont_name) {
  auto in_pack = parthenon::PackVariablesOnMesh(
      blocks, in_cont_name, std::vector<MetadataFlag>{Metadata::Independent});
  auto out_pack = parthenon::PackVariablesOnMesh(
      blocks, out_cont_name, std::vector<MetadataFlag>{Metadata::Independent});
  auto dudt_pack = parthenon::PackVariablesOnMesh(
      blocks, dudt_cont_name, std::vector<MetadataFlag>{Metadata::Independent});

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "UpdateContainer", DevExecSpace(), 0, blocks.size() - 1, 0,
      in_pack.GetDim(4) - 1, 0, in_pack.GetDim(3) - 1, 0, in_pack.GetDim(2) - 1, 0,
      in_pack.GetDim(1) - 1,
      KOKKOS_LAMBDA(const int b, const int l, const int k, const int j, const int i) {
        out_pack(b, l, k, j, i) = in_pack(b, l, k, j, i) + dt * dudt_pack(b, l, k, j, i);
      });
}

void AverageContainers(std::vector<MeshBlock *> &blocks, const std::string &c1_cont_name,
                       const std::string &c2_cont_name, const Real wgt1) {
  auto c1_pack = parthenon::PackVariablesOnMesh(
      blocks, c1_cont_name, std::vector<MetadataFlag>{Metadata::Independent});
  auto c2_pack = parthenon::PackVariablesOnMesh(
      blocks, c2_cont_name, std::vector<MetadataFlag>{Metadata::Independent});

  const IndexDomain interior = IndexDomain::interior;
  IndexRange ib = c1_pack.cellbounds.GetBoundsI(interior);
  IndexRange jb = c1_pack.cellbounds.GetBoundsJ(interior);
  IndexRange kb = c1_pack.cellbounds.GetBoundsK(interior);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "AverageContainer", DevExecSpace(), 0, blocks.size() - 1, 0,
      c1_pack.GetDim(4) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int l, const int k, const int j, const int i) {
        c1_pack(b, l, k, j, i) =
            wgt1 * c1_pack(b, l, k, j, i) + (1 - wgt1) * c2_pack(b, l, k, j, i);
      });
}

Real EstimateTimestep(std::shared_ptr<Container<Real>> &rc) {
  MeshBlock *pmb = rc->pmy_block;
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

TaskStatus FillDerivedVariables::FillDerived(std::shared_ptr<Container<Real>> &rc) {
  if (pre_package_fill_ != nullptr) {
    pre_package_fill_(rc);
  }
  for (auto &pkg : rc->pmy_block->packages) {
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
