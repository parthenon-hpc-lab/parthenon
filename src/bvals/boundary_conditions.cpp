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

#include "bvals/boundary_conditions.hpp"

#include "bvals/bvals_interfaces.hpp"
#include "interface/meshblock_data.hpp"
#include "interface/meshblock_data_iterator.hpp"
#include "mesh/mesh.hpp"

namespace parthenon {

TaskStatus ApplyBoundaryConditions(std::shared_ptr<MeshBlockData<Real>> &rc) {
  std::shared_ptr<MeshBlock> pmb = rc->GetBlockPointer();
  const IndexDomain interior = IndexDomain::interior;
  const IndexDomain entire = IndexDomain::entire;
  IndexRange ib = pmb->cellbounds.GetBoundsI(interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(interior);
  const int imax = pmb->cellbounds.ncellsi(entire);
  const int jmax = pmb->cellbounds.ncellsj(entire);
  const int kmax = pmb->cellbounds.ncellsk(entire);

  Metadata m;
  MeshBlockDataIterator<Real> citer(rc, {Metadata::Independent});
  const int nvars = citer.vars.size();

  switch (pmb->boundary_flag[BoundaryFace::inner_x1]) {
  case BoundaryFlag::outflow:
    for (int n = 0; n < nvars; n++) {
      ParArrayND<Real> q = citer.vars[n]->data;
      for (int l = 0; l < q.GetDim(4); l++) {
        for (int k = kb.s; k <= kb.e; k++) {
          for (int j = 0; j < jmax; j++) {
            for (int i = 0; i < ib.s; i++) {
              q(l, k, j, i) = q(l, k, j, ib.s);
            }
          }
        }
      }
    }
    break;

  case BoundaryFlag::reflect:
    for (int n = 0; n < nvars; n++) {
      ParArrayND<Real> q = citer.vars[n]->data;
      bool vec = citer.vars[n]->IsSet(Metadata::Vector);
      for (int l = 0; l < q.GetDim(4); l++) {
        Real reflect = (l == 0 && vec ? -1.0 : 1.0);
        for (int k = kb.s; k <= kb.e; k++) {
          for (int j = 0; j < jmax; j++) {
            for (int i = 0; i < ib.s; i++) {
              q(l, k, j, i) = reflect * q(l, k, j, 2 * ib.s - i - 1);
            }
          }
        }
      }
    }
    break;

  default:
    break;
  }

  switch (pmb->boundary_flag[BoundaryFace::outer_x1]) {
  case BoundaryFlag::outflow:
    for (int n = 0; n < nvars; n++) {
      ParArrayND<Real> q = citer.vars[n]->data;
      for (int l = 0; l < q.GetDim(4); l++) {
        for (int k = kb.s; k <= kb.e; k++) {
          for (int j = 0; j < jmax; j++) {
            for (int i = ib.e + 1; i < imax; i++) {
              q(l, k, j, i) = q(l, k, j, ib.e);
            }
          }
        }
      }
    }
    break;

  case BoundaryFlag::reflect:
    for (int n = 0; n < nvars; n++) {
      ParArrayND<Real> q = citer.vars[n]->data;
      bool vec = citer.vars[n]->IsSet(Metadata::Vector);
      for (int l = 0; l < q.GetDim(4); l++) {
        Real reflect = (l == 0 && vec ? -1.0 : 1.0);
        for (int k = kb.s; k <= kb.e; k++) {
          for (int j = 0; j < jmax; j++) {
            for (int i = ib.e + 1; i < imax; i++) {
              q(l, k, j, i) = reflect * q(l, k, j, 2 * ib.e - i + 1);
            }
          }
        }
      }
    }
    break;

  default:
    break;
  }

  if (pmb->pmy_mesh->ndim >= 2) {
    switch (pmb->boundary_flag[BoundaryFace::inner_x2]) {
    case BoundaryFlag::outflow:
      for (int n = 0; n < nvars; n++) {
        ParArrayND<Real> q = citer.vars[n]->data;
        for (int l = 0; l < q.GetDim(4); l++) {
          for (int k = kb.s; k <= kb.e; k++) {
            for (int j = 0; j < jb.s; j++) {
              for (int i = 0; i < imax; i++) {
                q(l, k, j, i) = q(l, k, jb.s, i);
              }
            }
          }
        }
      }
      break;

    case BoundaryFlag::reflect:
      for (int n = 0; n < nvars; n++) {
        ParArrayND<Real> q = citer.vars[n]->data;
        bool vec = citer.vars[n]->IsSet(Metadata::Vector);
        for (int l = 0; l < q.GetDim(4); l++) {
          Real reflect = (l == 1 && vec ? -1.0 : 1.0);
          for (int k = kb.s; k <= kb.e; k++) {
            for (int j = 0; j < jb.s; j++) {
              for (int i = 0; i < imax; i++) {
                q(l, k, j, i) = reflect * q(l, k, 2 * jb.s - j - 1, i);
              }
            }
          }
        }
      }
      break;

    default:
      break;
    }

    switch (pmb->boundary_flag[BoundaryFace::outer_x2]) {
    case BoundaryFlag::outflow:
      for (int n = 0; n < nvars; n++) {
        ParArrayND<Real> q = citer.vars[n]->data;
        for (int l = 0; l < q.GetDim(4); l++) {
          for (int k = kb.s; k <= kb.e; k++) {
            for (int j = jb.e + 1; j < jmax; j++) {
              for (int i = 0; i < imax; i++) {
                q(l, k, j, i) = q(l, k, jb.e, i);
              }
            }
          }
        }
      }
      break;

    case BoundaryFlag::reflect:
      for (int n = 0; n < nvars; n++) {
        ParArrayND<Real> q = citer.vars[n]->data;
        bool vec = citer.vars[n]->IsSet(Metadata::Vector);
        for (int l = 0; l < q.GetDim(4); l++) {
          Real reflect = (l == 1 && vec ? -1.0 : 1.0);
          for (int k = kb.s; k <= kb.e; k++) {
            for (int j = jb.e + 1; j < jmax; j++) {
              for (int i = 0; i < imax; i++) {
                q(l, k, j, i) = reflect * q(l, k, 2 * jb.e - j + 1, i);
              }
            }
          }
        }
      }
      break;

    default:
      break;
    }
  } // if ndim>=2

  if (pmb->pmy_mesh->ndim >= 3) {
    switch (pmb->boundary_flag[BoundaryFace::inner_x3]) {
    case BoundaryFlag::outflow:
      for (int n = 0; n < nvars; n++) {
        ParArrayND<Real> q = citer.vars[n]->data;
        for (int l = 0; l < q.GetDim(4); l++) {
          for (int k = 0; k < kb.s; k++) {
            for (int j = 0; j < jmax; j++) {
              for (int i = 0; i < imax; i++) {
                q(l, k, j, i) = q(l, kb.s, j, i);
              }
            }
          }
        }
      }
      break;

    case BoundaryFlag::reflect:
      for (int n = 0; n < nvars; n++) {
        ParArrayND<Real> q = citer.vars[n]->data;
        bool vec = citer.vars[n]->IsSet(Metadata::Vector);
        for (int l = 0; l < q.GetDim(4); l++) {
          Real reflect = (l == 2 && vec ? -1.0 : 1.0);
          for (int k = 0; k < kb.s; k++) {
            for (int j = 0; j < jmax; j++) {
              for (int i = 0; i < imax; i++) {
                q(l, k, j, i) = reflect * q(l, 2 * kb.s - k - 1, j, i);
              }
            }
          }
        }
      }
      break;

    default:
      break;
    }

    switch (pmb->boundary_flag[BoundaryFace::outer_x3]) {
    case BoundaryFlag::outflow:
      for (int n = 0; n < nvars; n++) {
        ParArrayND<Real> q = citer.vars[n]->data;
        for (int l = 0; l < q.GetDim(4); l++) {
          for (int k = kb.e + 1; k < kmax; k++) {
            for (int j = 0; j < jmax; j++) {
              for (int i = 0; i < imax; i++) {
                q(l, k, j, i) = q(l, kb.e, j, i);
              }
            }
          }
        }
      }
      break;

    case BoundaryFlag::reflect:
      for (int n = 0; n < nvars; n++) {
        ParArrayND<Real> q = citer.vars[n]->data;
        bool vec = citer.vars[n]->IsSet(Metadata::Vector);
        for (int l = 0; l < q.GetDim(4); l++) {
          Real reflect = (l == 2 && vec ? -1.0 : 1.0);
          for (int k = kb.e + 1; k < kmax; k++) {
            for (int j = 0; j < jmax; j++) {
              for (int i = 0; i < imax; i++) {
                q(l, k, j, i) = reflect * q(l, 2 * kb.e - k + 1, j, i);
              }
            }
          }
        }
      }
      break;

    default:
      break;
    }
  } // if ndim >= 3

  return TaskStatus::complete;
}

} // namespace parthenon
