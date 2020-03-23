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

#include "boundary_conditions.hpp"

#include "bvals/bvals_interfaces.hpp"
#include "interface/Container.hpp"
#include "interface/ContainerIterator.hpp"
#include "mesh/mesh.hpp"

namespace parthenon {

void ApplyBoundaryConditions(Container<Real>& rc) {
    MeshBlock *pmb = rc.pmy_block;
    const int is = pmb->is; const int js = pmb->js; const int ks = pmb->ks;
    const int ie = pmb->ie; const int je = pmb->je; const int ke = pmb->ke;
    const int imax = pmb->ncells1; const int jmax = pmb->ncells2; const int kmax = pmb->ncells3;

    Metadata m;
    ContainerIterator<Real> citer(rc, std::vector<parthenon::Metadata::flags> {m.independent});
    const int nvars = citer.vars.size();

    switch (pmb->boundary_flag[BoundaryFace::inner_x1]) {
        case BoundaryFlag::outflow: {
            for (int n=0; n<nvars; n++) {
                Variable<Real>& q = *citer.vars[n];
                for (int l=0; l<q.GetDim4(); l++) {
                    for (int k=ks; k<=ke; k++) {
                        for (int j=0; j<jmax; j++) {
                            for (int i=0; i<is; i++) {
                                q(l,k,j,i) = q(l,k,j,is);
                            }
                        }
                    }
                }
            }
            break;
        }
        case BoundaryFlag::reflect: {
            for (int n=0; n<nvars; n++) {
                Variable<Real>& q = *citer.vars[n];
                bool vec = q.metadata().isVector();
                for (int l=0; l<q.GetDim4(); l++) {
                    Real reflect = (l==0 && vec ? -1.0 : 1.0);
                    for (int k=ks; k<=ke; k++) {
                        for (int j=0; j<jmax; j++) {
                            for (int i=0; i<is; i++) {
                                q(l,k,j,i) = reflect*q(l,k,j,2*is-i-1);
                            }
                        }
                    }
                }
            }
            break;
        }
    default: {
      break;
    }
    };

    switch (pmb->boundary_flag[BoundaryFace::outer_x1]) {
        case BoundaryFlag::outflow: {
            for (int n=0; n<nvars; n++) {
                Variable<Real>& q = *citer.vars[n];
                for (int l=0; l<q.GetDim4(); l++) {
                    for (int k=ks; k<=ke; k++) {
                        for (int j=0; j<jmax; j++) {
                            for (int i=ie+1; i<imax; i++) {
                                q(l,k,j,i) = q(l,k,j,ie);
                            }
                        }
                    }
                }
            }
            break;
        }
        case BoundaryFlag::reflect: {
            for (int n=0; n<nvars; n++) {
                Variable<Real>& q = *citer.vars[n];
                bool vec = q.metadata().isVector();
                for (int l=0; l<q.GetDim4(); l++) {
                    Real reflect = (l==0 && vec ? -1.0 : 1.0);
                    for (int k=ks; k<=ke; k++) {
                        for (int j=0; j<jmax; j++) {
                            for (int i=ie+1; i<imax; i++) {
                                q(l,k,j,i) = reflect*q(l,k,j,2*ie-i+1);
                            }
                        }
                    }
                }
            }
            break;
        }
    default: {
      break;
    }
    };



    if (pmb->pmy_mesh->ndim >= 2) {

    switch (pmb->boundary_flag[BoundaryFace::inner_x2]) {
        case BoundaryFlag::outflow: {
            for (int n=0; n<nvars; n++) {
                Variable<Real>& q = *citer.vars[n];
                for (int l=0; l<q.GetDim4(); l++) {
                    for (int k=ks; k<=ke; k++) {
                        for (int j=0; j<js; j++) {
                            for (int i=0; i<imax; i++) {
                                q(l,k,j,i) = q(l,k,js,i);
                            }
                        }
                    }
                }
            }
            break;
        }
        case BoundaryFlag::reflect: {
            for (int n=0; n<nvars; n++) {
                Variable<Real>& q = *citer.vars[n];
                bool vec = q.metadata().isVector();
                for (int l=0; l<q.GetDim4(); l++) {
                    Real reflect = (l==1 && vec ? -1.0 : 1.0);
                    for (int k=ks; k<=ke; k++) {
                        for (int j=0; j<js; j++) {
                            for (int i=0; i<imax; i++) {
                                q(l,k,j,i) = reflect*q(l,k,2*js-j-1,i);
                            }
                        }
                    }
                }
            }
            break;
        }
    default: {
      break;
    }
    };

    switch (pmb->boundary_flag[BoundaryFace::outer_x2]) {
        case BoundaryFlag::outflow: {
            for (int n=0; n<nvars; n++) {
                Variable<Real>& q = *citer.vars[n];
                for (int l=0; l<q.GetDim4(); l++) {
                    for (int k=ks; k<=ke; k++) {
                        for (int j=je+1; j<jmax; j++) {
                            for (int i=0; i<imax; i++) {
                                q(l,k,j,i) = q(l,k,je,i);
                            }
                        }
                    }
                }
            }
            break;
        }
        case BoundaryFlag::reflect: {
            for (int n=0; n<nvars; n++) {
                Variable<Real>& q = *citer.vars[n];
                bool vec = q.metadata().isVector();
                for (int l=0; l<q.GetDim4(); l++) {
                    Real reflect = (l==1 && vec ? -1.0 : 1.0);
                    for (int k=ks; k<=ke; k++) {
                        for (int j=je+1; j<jmax; j++) {
                            for (int i=0; i<imax; i++) {
                                q(l,k,j,i) = reflect*q(l,k,2*je-j+1,i);
                            }
                        }
                    }
                }
            }
            break;
        }
    default: {
      break;
    }
    };

    } // if ndim>=2


    if (pmb->pmy_mesh->ndim >= 3) {

   switch (pmb->boundary_flag[BoundaryFace::inner_x3]) {
        case BoundaryFlag::outflow: {
            for (int n=0; n<nvars; n++) {
                Variable<Real>& q = *citer.vars[n];
                for (int l=0; l<q.GetDim4(); l++) {
                    for (int k=0; k<ks; k++) {
                        for (int j=0; j<jmax; j++) {
                            for (int i=0; i<imax; i++) {
                                q(l,k,j,i) = q(l,ks,j,i);
                            }
                        }
                    }
                }
            }
            break;
        }
        case BoundaryFlag::reflect: {
            for (int n=0; n<nvars; n++) {
                Variable<Real>& q = *citer.vars[n];
                bool vec = q.metadata().isVector();
                for (int l=0; l<q.GetDim4(); l++) {
                    Real reflect = (l==2 && vec ? -1.0 : 1.0);
                    for (int k=0; k<ks; k++) {
                        for (int j=0; j<jmax; j++) {
                            for (int i=0; i<imax; i++) {
                                q(l,k,j,i) = reflect*q(l,2*ks-k-1,j,i);
                            }
                        }
                    }
                }
            }
            break;
        }
    default: {
      break;
    }
    };

    switch (pmb->boundary_flag[BoundaryFace::outer_x3]) {
        case BoundaryFlag::outflow: {
            for (int n=0; n<nvars; n++) {
                Variable<Real>& q = *citer.vars[n];
                for (int l=0; l<q.GetDim4(); l++) {
                    for (int k=ke+1; k<kmax; k++) {
                        for (int j=0; j<jmax; j++) {
                            for (int i=0; i<imax; i++) {
                                q(l,k,j,i) = q(l,ke,j,i);
                            }
                        }
                    }
                }
            }
            break;
        }
        case BoundaryFlag::reflect: {
            for (int n=0; n<nvars; n++) {
                Variable<Real>& q = *citer.vars[n];
                bool vec = q.metadata().isVector();
                for (int l=0; l<q.GetDim4(); l++) {
                    Real reflect = (l==2 && vec ? -1.0 : 1.0);
                    for (int k=ke+1; k<kmax; k++) {
                        for (int j=0; j<jmax; j++) {
                            for (int i=0; i<imax; i++) {
                                q(l,k,j,i) = reflect*q(l,2*ke-k+1,j,i);
                            }
                        }
                    }
                }
            }
            break;
        }
    default: {
      break;
    }
    };

    } // if ndim >= 3

}

}
