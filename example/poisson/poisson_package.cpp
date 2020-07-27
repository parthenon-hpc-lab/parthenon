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

#include <cmath>
#include <string>

#include <parthenon/package.hpp>

#include "poisson_package.hpp"
#include "reconstruct/dc_inline.hpp"

using namespace parthenon::package::prelude;

namespace poisson {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto pkg = std::make_shared<StateDescriptor>("poisson_package");

  Real K = pin->GetOrAddReal("poisson", "K", 1);
  pkg->AddParam<>("K", K);

  Real w = pin->GetOrAddReal("poisson", "weight", 1);
  pkg->AddParam<>("weight", w);

  int subcycles = pin->GetOrAddInteger("poisson", "subcycles", 1);
  pkg->AddParam<>("subcycles", subcycles);

  std::string field_name = "field";
  Metadata m({Metadata::Cell, Metadata::Independent, Metadata::FillGhost});
  pkg->AddField(field_name, m);

  field_name = "potential";
  m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy});
  pkg->AddField(field_name, m);

  field_name = "residual";
  m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy});
  pkg->AddField(field_name, m);

  // For linear systems, inv_diagonal doesn't need to be a grid
  // variable, it could be an inline function. However, more
  // generally, it's useful to store on the grid.
  field_name = "inv_diagonal";
  m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy});
  pkg->AddField(field_name, m);

  return pkg;
}

// TODO(JMM): Refactor to reduced repeated code
Real GetL1Residual(std::shared_ptr<Container<Real>> &rc) {
  MeshBlock *pmb = rc->pmy_block;
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  ParArrayND<Real> res = rc->Get("residual").data;

  Real max;
  Kokkos::parallel_reduce(
      "Poisson Get Residual",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>(pmb->exec_space, {kb.s, jb.s, ib.s},
                                             {kb.e + 1, jb.e + 1, ib.e + 1}),
      KOKKOS_LAMBDA(int k, int j, int i, Real &lmax) {
        lmax = fmax(fabs(res(k, j, i)), lmax);
      },
      Kokkos::Max<Real>(max));
  return max;
}

TaskStatus ComputeResidualAndDiagonal(std::shared_ptr<Container<Real>> &div,
                                      std::shared_ptr<Container<Real>> &update) {
  MeshBlock *pmb = div->pmy_block;
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  auto &coords = pmb->coords;
  auto pkg = pmb->packages["poisson_package"];
  int ndim = pmb->pmy_mesh->ndim;
  ParArrayND<Real> dphi = div->Get("field").data; // div(grad(phi))
  ParArrayND<Real> rho = update->Get("potential").data;
  ParArrayND<Real> res = update->Get("residual").data;
  ParArrayND<Real> Dinv = update->Get("inv_diagonal").data;
  Real K = pkg->Param<Real>("K");

  pmb->par_for(
      "ComputeResidualAndDiagonal", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        Real dx = coords.Dx(X1DIR, k, j, i);
        Real dy = coords.Dx(X2DIR, k, j, i);
        Real dz = coords.Dx(X3DIR, k, j, i);
        Real dx2 = dx * dx;
        Real dy2 = dy * dy;
        Real dz2 = dz * dz;
        Real ds2;
        if (ndim >= 3) {
          ds2 = (dx2 * dy2 * dz2) / (dx2 * dy2 + dx2 * dz2 + dy2 * dz2);
        } else if (ndim >= 2) {
          ds2 = (dx2 * dy2) / (dx2 + dy2);
        } else {
          ds2 = dx2;
        }
        Dinv(k, j, i) = -ds2 / 2;
        res(k, j, i) = K * rho(k, j, i);
        res(k, j, i) -= dphi(k, j, i);
      });
  return TaskStatus::complete;
}

TaskStatus Smooth(std::shared_ptr<Container<Real>> &rc_in,
                  std::shared_ptr<Container<Real>> &rc_out) {
  MeshBlock *pmb = rc_in->pmy_block;
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  auto pkg = pmb->packages["poisson_package"];
  ParArrayND<Real> in = rc_in->Get("field").data;
  ParArrayND<Real> out = rc_out->Get("field").data;
  ParArrayND<Real> res = rc_out->Get("residual").data;
  ParArrayND<Real> Dinv = rc_out->Get("inv_diagonal").data;
  Real w = pkg->Param<Real>("weight");
  int nsub = pkg->Param<int>("subcycles");

  pmb->par_for(
      "JacobiSmooth", 1, nsub, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int n, const int k, const int j, const int i) {
        out(k, j, i) = in(k, j, i) + w * Dinv(k, j, i) * res(k, j, i);
      });
  return TaskStatus::complete;
}

TaskStatus CalculateFluxes(std::shared_ptr<Container<Real>> &rc) {
  MeshBlock *pmb = rc->pmy_block;
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  ParArrayND<Real> field = rc->Get("field").data;

  auto &coords = pmb->coords;

  const int nx1 = pmb->cellbounds.ncellsi(IndexDomain::entire);
  const int nvar = field.GetDim(4);
  const int scratch_level = 1; // 0 is actual scratch (tiny); 1 is HBM
  size_t scratch_size_in_bytes = parthenon::ScratchPad2D<Real>::shmem_size(nvar, nx1);

  auto x1flux = rc->Get("field").flux[X1DIR].Get<4>();
  pmb->par_for_outer(
      "X1 flux", 2 * scratch_size_in_bytes, scratch_level, kb.s, kb.e, jb.s, jb.e,
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int k, const int j) {
        parthenon::ScratchPad2D<Real> ql(member.team_scratch(scratch_level), nvar, nx1);
        parthenon::ScratchPad2D<Real> qr(member.team_scratch(scratch_level), nvar, nx1);
        // We're re-using DonorCell but this is *not* a reconstruction.
        // we simply want to pre-cache phi[i] and phi[i+1] in ql and qr.
        parthenon::DonorCellX1(member, k, j, ib.s - 1, ib.e + 1, field, ql, qr);
        member.team_barrier();
        for (int n = 0; n < nvar; n++) {
          parthenon::par_for_inner(member, ib.s, ib.e + 1, [&](const int i) {
            // Flux is forward differences. Extension to higher-order
            // is a higher-order derivative operator here
            x1flux(n, k, j, i) = (qr(n, i) - ql(n, i)) / coords.Dx(X1DIR, k, j, i);
          });
        }
      });
  if (pmb->pmy_mesh->ndim >= 2) {
    auto x2flux = rc->Get("field").flux[X2DIR].Get<4>();
    pmb->par_for_outer(
        "X2 flux", 3 * scratch_size_in_bytes, scratch_level, kb.s, kb.e, jb.s, jb.e + 1,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int k, const int j) {
          // We can't re-use reconstructed arrays with different j
          parthenon::ScratchPad2D<Real> ql(member.team_scratch(scratch_level), nvar, nx1);
          parthenon::ScratchPad2D<Real> qr(member.team_scratch(scratch_level), nvar, nx1);
          parthenon::ScratchPad2D<Real> q_unused(member.team_scratch(scratch_level), nvar,
                                                 nx1);
          // We're re-using DonorCell but this is *not* a reconstruction.
          // we simply want to pre-cache phi[j] and phi[j+1] in ql and qr.
          parthenon::DonorCellX2(member, k, j - 1, ib.s, ib.e, field, ql, q_unused);
          parthenon::DonorCellX2(member, k, j, ib.s, ib.e, field, q_unused, qr);
          member.team_barrier();
          for (int n = 0; n < nvar; n++) {
            parthenon::par_for_inner(member, ib.s, ib.e, [&](const int i) {
              // Flux is forward differences. Extension to higher-order
              // is a higher-order derivative operator here
              x2flux(n, k, j, i) = (qr(n, i) - ql(n, i)) / coords.Dx(X2DIR, k, j, i);
            });
          }
        });
  }
  if (pmb->pmy_mesh->ndim >= 3) {
    auto x3flux = rc->Get("field").flux[X3DIR].Get<4>();
    pmb->par_for_outer(
        "X3 flux", 3 * scratch_size_in_bytes, scratch_level, kb.s, kb.e + 1, jb.s, jb.e,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int k, const int j) {
          parthenon::ScratchPad2D<Real> ql(member.team_scratch(scratch_level), nvar, nx1);
          parthenon::ScratchPad2D<Real> qr(member.team_scratch(scratch_level), nvar, nx1);
          parthenon::ScratchPad2D<Real> q_unused(member.team_scratch(scratch_level), nvar,
                                                 nx1);
          // We're re-using DonorCell but this is *not* a reconstruction.
          // we simply want to pre-cache phi[k] and phi[k+1] in ql and qr.
          parthenon::DonorCellX3(member, k - 1, j, ib.s, ib.e, field, ql, q_unused);
          parthenon::DonorCellX3(member, k, j, ib.s, ib.e, field, q_unused, qr);
          member.team_barrier();
          for (int n = 0; n < nvar; n++) {
            parthenon::par_for_inner(member, ib.s, ib.e, [&](const int i) {
              // Flux is forward differences. Extension to higher-order
              // is a higher-order derivative operator here
              x3flux(n, k, j, i) = (qr(n, i) - ql(n, i)) / coords.Dx(X3DIR, k, j, i);
            });
          }
        });
  }
  return TaskStatus::complete;
}

} // namespace poisson
