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

#include "../coordinates/coordinates.hpp"
#include "../interface/Container.hpp"
#include "../interface/ContainerIterator.hpp"
#include "../mesh/mesh.hpp"

namespace parthenon {
namespace Update {

void FluxDivergence(Container<Real> &in, Container<Real> &dudt_cont) {
  MeshBlock *pmb = in.pmy_block;
  int is = pmb->active_cells.x.at(0).s;
  int js = pmb->active_cells.x.at(1).s;
  int ks = pmb->active_cells.x.at(2).s;
  int ie = pmb->active_cells.x.at(0).e;
  int je = pmb->active_cells.x.at(1).e;
  int ke = pmb->active_cells.x.at(2).e;

  Metadata m;
  ContainerIterator<Real> cin_iter(in, {m.independent});
  ContainerIterator<Real> cout_iter(dudt_cont, {m.independent});
  int nvars = cout_iter.vars.size();

  AthenaArray<Real> x1area(pmb->all_cells.x.at(0).n());
  AthenaArray<Real> x2area0(pmb->all_cells.x.at(0).n());
  AthenaArray<Real> x2area1(pmb->all_cells.x.at(0).n());
  AthenaArray<Real> x3area0(pmb->all_cells.x.at(0).n());
  AthenaArray<Real> x3area1(pmb->all_cells.x.at(0).n());
  AthenaArray<Real> vol(pmb->all_cells.x.at(0).n());

  /*for (int n = 0; n < nvars; n++) {
    Variable<Real>& dudt = *cout_iter.vars[n];
    dudt.ZeroClear();
  }*/
  int ndim = pmb->pmy_mesh->ndim;
  AthenaArray<Real> du(pmb->all_cells.x.at(0).n());
  for (int k = ks; k <= ke; k++) {
    for (int j = js; j <= je; j++) {
      pmb->pcoord->Face1Area(k, j, is, ie + 1, x1area);
      pmb->pcoord->CellVolume(k, j, is, ie, vol);
      if (pmb->pmy_mesh->f2) {
        pmb->pcoord->Face2Area(k, j, is, ie, x2area0);
        pmb->pcoord->Face2Area(k, j + 1, is, ie, x2area1);
      }
      if (pmb->pmy_mesh->f3) {
        pmb->pcoord->Face3Area(k, j, is, ie, x3area0);
        pmb->pcoord->Face3Area(k + 1, j, is, ie, x3area1);
      }
      for (int n = 0; n < nvars; n++) {
        Variable<Real> &q = *cin_iter.vars[n];
        AthenaArray<Real> &x1flux = q.flux[0];
        AthenaArray<Real> &x2flux = q.flux[1];
        AthenaArray<Real> &x3flux = q.flux[2];
        Variable<Real> &dudt = *cout_iter.vars[n];
        for (int l = 0; l < q.GetDim4(); l++) {
          du.ZeroClear();
          for (int i = is; i <= ie; i++) {
            du(i) = (x1area(i + 1) * x1flux(l, k, j, i + 1) -
                     x1area(i) * x1flux(l, k, j, i));
          }

          if (pmb->pmy_mesh->f2) {
            for (int i = is; i <= ie; i++) {
              du(i) += (x2area1(i) * x2flux(l, k, j + 1, i) -
                        x2area0(i) * x2flux(l, k, j, i));
            }
          }
          // TODO(jcd): should the next block be in the preceding if??
          if (pmb->pmy_mesh->f3) {
            for (int i = is; i <= ie; i++) {
              du(i) += (x3area1(i) * x3flux(l, k + 1, j, i) -
                        x3area0(i) * x3flux(l, k, j, i));
            }
          }

          for (int i = is; i <= ie; i++) {
            dudt(l, k, j, i) = -du(i) / vol(i);
          }
        }
      }
    }
  }

  return;
}

void UpdateContainer(Container<Real> &in, Container<Real> &dudt_cont,
                     const Real dt, Container<Real> &out) {
  MeshBlock *pmb = in.pmy_block;
  int is = pmb->active_cells.x.at(0).s;
  int js = pmb->active_cells.x.at(1).s;
  int ks = pmb->active_cells.x.at(2).s;
  int ie = pmb->active_cells.x.at(0).e;
  int je = pmb->active_cells.x.at(1).e;
  int ke = pmb->active_cells.x.at(2).e;

  Metadata m;
  ContainerIterator<Real> cin_iter(in, {m.independent});
  ContainerIterator<Real> cout_iter(out, {m.independent});
  ContainerIterator<Real> du_iter(dudt_cont, {m.independent});
  int nvars = cout_iter.vars.size();

  for (int n = 0; n < nvars; n++) {
    Variable<Real> &qin = *cin_iter.vars[n];
    Variable<Real> &dudt = *du_iter.vars[n];
    Variable<Real> &qout = *cout_iter.vars[n];
    for (int l = 0; l < qout.GetDim4(); l++) {
      for (int k = ks; k <= ke; k++) {
        for (int j = js; j <= je; j++) {
          for (int i = is; i <= ie; i++) {
            qout(l, k, j, i) = qin(l, k, j, i) + dt * dudt(l, k, j, i);
          }
        }
      }
    }
  }
  return;
}

void AverageContainers(Container<Real> &c1, Container<Real> &c2,
                       const Real wgt1) {
  MeshBlock *pmb = c1.pmy_block;
  int is = pmb->active_cells.x.at(0).s;
  int js = pmb->active_cells.x.at(1).s;
  int ks = pmb->active_cells.x.at(2).s;
  int ie = pmb->active_cells.x.at(0).e;
  int je = pmb->active_cells.x.at(1).e;
  int ke = pmb->active_cells.x.at(2).e;

  Metadata m;
  ContainerIterator<Real> c1_iter(c1, {m.independent});
  ContainerIterator<Real> c2_iter(c2, {m.independent});
  int nvars = c2_iter.vars.size();

  for (int n = 0; n < nvars; n++) {
    Variable<Real> &q1 = *c1_iter.vars[n];
    Variable<Real> &q2 = *c2_iter.vars[n];
    for (int l = 0; l < q1.GetDim4(); l++) {
      for (int k = ks; k <= ke; k++) {
        for (int j = js; j <= je; j++) {
          for (int i = is; i <= ie; i++) {
            q1(l, k, j, i) =
                wgt1 * q1(l, k, j, i) + (1 - wgt1) * q2(l, k, j, i);
          }
        }
      }
    }
  }
  return;
}

void FillDerived(PreFillDerivedFunc pre_fill_derived, Container<Real> &rc) {
  pre_fill_derived(rc);

  for (auto &phys : rc.pmy_block->physics) {
    auto &desc = phys.second;
    if (desc->FillDerived != nullptr) {
      desc->FillDerived(rc);
    }
  }
}

Real EstimateTimestep(Container<Real> &rc) {
  MeshBlock *pmb = rc.pmy_block;
  Real dt_min = std::numeric_limits<Real>::max();
  for (auto &phys : pmb->physics) {
    auto &desc = phys.second;
    if (desc->EstimateTimestep != nullptr) {
      Real dt_phys = desc->EstimateTimestep(rc);
      dt_min = std::min(dt_min, dt_phys);
    }
  }
  return dt_min;
}

} // namespace Update
}
