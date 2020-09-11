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

#include "coordinates/coordinates.hpp"
#include "interface/container.hpp"
#include "mesh/mesh.hpp"

#include "kokkos_abstraction.hpp"

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

TaskStatus TransportSwarm(SP_Swarm &in, SP_Swarm &out, const Real dt) {
  int nmax_active_index = in->get_max_active_index();
  int nmax_active = nmax_active_index + 1;

  MeshBlock *pmb = in->pmy_block;

  auto &x_in = in->GetReal("x").Get();
  auto &y_in = in->GetReal("y").Get();
  auto &z_in = in->GetReal("z").Get();
  auto &x_out = out->GetReal("x").Get();
  auto &y_out = out->GetReal("y").Get();
  auto &z_out = out->GetReal("z").Get();
  auto &vx = in->GetReal("vx").Get();
  auto &vy = in->GetReal("vy").Get();
  auto &vz = in->GetReal("vz").Get();
  auto &mask = in->GetMask().Get();

  Real dx_i = pmb->coords.dx1f(pmb->cellbounds.is(IndexDomain::interior));       
  Real dx_j = pmb->coords.dx2f(pmb->cellbounds.js(IndexDomain::interior));       
  Real dx_k = pmb->coords.dx3f(pmb->cellbounds.ks(IndexDomain::interior));       
  Real dx_push = std::min<Real>(dx_i, std::min<Real>(dx_j, dx_k));

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);             
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);             
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);             
                                                                                 
  Real x_min = pmb->coords.x1v(ib.s) - dx_i / 2.;                                
  Real y_min = pmb->coords.x2v(jb.s) - dx_j / 2.;                                
  Real z_min = pmb->coords.x3v(kb.s) - dx_k / 2.;                                
  Real x_max = pmb->coords.x1v(ib.e) + dx_i / 2.;                                
  Real y_max = pmb->coords.x2v(jb.e) + dx_j / 2.;                                
  Real z_max = pmb->coords.x3v(kb.e) + dx_k / 2.;

  ParArrayND<Real> t("time", nmax_active);

  pmb->par_for("TransportSwarm", 0, nmax_active_index,
    KOKKOS_LAMBDA(const int n) {
      if (mask(n)) {
        t(n) = 0.;
        Real v = sqrt(vx(n)*vx(n) + vy(n)*vy(n) + vz(n)*vz(n));
        x_out(n) = x_in(n);
        y_out(n) = y_in(n);
        z_out(n) = z_in(n);
        while (t(n) < dt) {
          Real dt_cell = dx_push / v;
          Real dt_end = dt - t(n);
          Real dt_push = dt_cell;
          if (dt_end < dt_cell) {
            dt_push = dt_end;
          }

          x_out(n) = x_out(n) + vx(n)*dt_push;
          y_out(n) = y_out(n) + vy(n)*dt_push;
          z_out(n) = z_out(n) + vz(n)*dt_push;
          t(n) += dt_push;

          // Reflecting boundaries
          if (x_out(n) < x_min) {
            x_out(n) = x_min + (x_min - x_out(n));
            vx(n) = -vx(n);
          }
          if (x_out(n) > x_max) {                                                    
            x_out(n) = x_max - (x_out(n) - x_max);                                       
            vx(n) = -vx(n);                                                      
          }                                                                      
          if (y_out(n) < y_min) {                                                    
            y_out(n) = y_max - (y_min - y_out(n));                                       
          }                                                                      
          if (y_out(n) > y_max) {                                                    
            y_out(n) = y_min + (y_out(n) - y_max);                                       
          }                                                                      
          if (z_out(n) < z_min) {                                                    
            z_out(n) = z_max - (z_min - z_out(n));                                       
          }                                                                      
          if (z_out(n) > z_max) {                                                    
            z_out(n) = z_min + (z_out(n) - z_max);                                       
          }
        }
      }
    });

  return TaskStatus::complete;
}

void UpdateContainer(std::shared_ptr<Container<Real>> &in,
                     std::shared_ptr<Container<Real>> &dudt_cont, const Real dt,
                     std::shared_ptr<Container<Real>> &out) {
  MeshBlock *pmb = in->pmy_block;

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

void AverageContainers(std::shared_ptr<Container<Real>> &c1,
                       std::shared_ptr<Container<Real>> &c2, const Real wgt1) {
  MeshBlock *pmb = c1->pmy_block;
  const IndexDomain interior = IndexDomain::interior;
  IndexRange ib = pmb->cellbounds.GetBoundsI(interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(interior);

  auto v1 = c1->PackVariables({Metadata::Independent});
  auto v2 = c2->PackVariables({Metadata::Independent});

  pmb->par_for(
      "AverageContainers", 0, v1.GetDim(4) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int l, const int k, const int j, const int i) {
        v1(l, k, j, i) = wgt1 * v1(l, k, j, i) + (1 - wgt1) * v2(l, k, j, i);
      });

  return;
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
