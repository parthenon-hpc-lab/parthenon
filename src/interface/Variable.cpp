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

#include <array>
#include <iostream>
#include <string>


#include "bvals/cc/bvals_cc.hpp"
#ifndef TEST_PK
#include "mesh/mesh.hpp"
#endif // TEST_PK
#include "athena_arrays.hpp"
#include "Variable.hpp"

namespace parthenon {
/*template <typename T>
Variable<T>::~Variable() {
  //  std::cout << "_________DELETING VAR: " << _label << ":" << this << std::endl;
  _label = "Deleted";
  if (_m.isSet(_m.fillGhost)) {

    // not sure if destructor is called for flux, need to check --Sriram

    // flux is a different variable even if shared
    // so always delete
    //    for (int i=0; i<3; i++) flux[i].DeleteAthenaArray();

    // Delete vbvar, coarse_r, and coarse_s only if not shared
    if ( _m.isSet(_m.sharedComms) ) {
      // do not delete unallocated variables
      vbvar = nullptr;
      coarse_r = coarse_s = nullptr;
    } else {
      // delete allocated variables
      //      coarse_r->DeleteAthenaArray();
      //      coarse_s->DeleteAthenaArray();
      delete coarse_r;
      delete coarse_s;
      delete vbvar;
    }
  }
}*/

template <typename T>
void Variable<T>::resetBoundary() {
  this->vbvar->var_cc = this;
}

template <typename T>
std::string Variable<T>::info() {
    char tmp[100] = "";
    char *stmp = tmp;

    // first add label
    std::string s = this->label();
    s.resize(20,'.');
    s += " : ";

    // now append size
    snprintf(tmp, sizeof(tmp),
             "%dx%dx%dx%dx%dx%d",
             this->GetDim6(),
             this->GetDim5(),
             this->GetDim4(),
             this->GetDim3(),
             this->GetDim2(),
             this->GetDim1()
             );
    while (! strncmp(stmp,"1x",2)) {
      stmp += 2;
    }
    s += stmp;
    // now append flag
    s += " : " + std::to_string(this->metadata().mask());
    s += " : " + this->metadata().maskAsString();

    return s;
  }

// copy constructor
template <typename T>
Variable<T>::Variable(const Variable<T> &src,
                      const bool allocComms,
                      MeshBlock *pmb) :
  AthenaArray<T>(src), _label(src.label()), _m(src.metadata()), mpiStatus(false) {
  //std::cout << "_____CREATED VAR COPY: " << _label << ":" << this << std::endl;
  if (_m.isSet(Metadata::fillGhost)) {
    // Ghost cells are communicated, so make shallow copies
    // of communication arrays and swap out the boundary array

    if ( allocComms ) {
      this->allocateComms(pmb);
    } else {
      _m.set(_m.sharedComms); // note that comms are shared

      // set data pointer for the boundary communication
      // Note that vbvar->var_cc will be set when stage is selected
       vbvar = src.vbvar;

      // fluxes, etc are always a copy
      for (int i = 0; i<3; i++) {
        if (src.flux[i].data()) {
          //int n6 = src.flux[i].GetDim6();
          //flux[i].InitWithShallowSlice(src.flux[i],6,0,n6);
          flux[i].ShallowCopy(src.flux[i]);
        } else {
          flux[i] = src.flux[i];
        }
      }

      // These members are pointers,
      // point at same memory as src
      coarse_r = src.coarse_r;
      coarse_s = src.coarse_s;
    }
  }
}


/// allocate communication space based on info in MeshBlock
/// Initialize a 6D variable
template <typename T>
void Variable<T>::allocateComms(MeshBlock *pmb) {
  if ( ! pmb ) return;

  // set up communication variables
  //const int _dim1 = this->GetDim1();
  //const int _dim2 = this->GetDim2();
  //const int _dim3 = this->GetDim3();
  const int _dim4 = this->GetDim4();
  //const int _dim5 = this->GetDim5();
  //const int _dim6 = this->GetDim6();
  flux[0].NewAthenaArray(_dim4, pmb->ncells3, pmb->ncells2, pmb->ncells1+1);
  if (pmb->pmy_mesh->ndim >= 2) {
    flux[1].NewAthenaArray(_dim4, pmb->ncells3, pmb->ncells2+1, pmb->ncells1);
  }
  if (pmb->pmy_mesh->ndim >= 3) {
    flux[2].NewAthenaArray(_dim4, pmb->ncells3+1, pmb->ncells2, pmb->ncells1);
  }
  coarse_s = new AthenaArray<Real>(_dim4, pmb->ncc3, pmb->ncc2, pmb->ncc1,
                                (pmb->pmy_mesh->multilevel ?
                                 AthenaArray<Real>::DataStatus::allocated :
                                 AthenaArray<Real>::DataStatus::empty));

  coarse_r = new AthenaArray<Real>(_dim4, pmb->ncc3, pmb->ncc2, pmb->ncc1,
                                (pmb->pmy_mesh->multilevel ?
                                 AthenaArray<Real>::DataStatus::allocated :
                                 AthenaArray<Real>::DataStatus::empty));

  // Create the boundary object
  vbvar = new CellCenteredBoundaryVariable(pmb, this, coarse_s, flux);

  // enroll CellCenteredBoundaryVariable object
  vbvar->bvar_index = pmb->pbval->bvars.size();
  pmb->pbval->bvars.push_back(vbvar);
  pmb->pbval->bvars_main_int.push_back(vbvar);

  // register the variable
  //pmb->RegisterMeshBlockData(*this);

  mpiStatus = false;
}

std::string FaceVariable::info() {
  char tmp[100] = "";

  // first add label
  std::string s = this->label();
  s.resize(20,'.');
  s +=  " : ";

  // now append size
  snprintf(tmp, sizeof(tmp),
          "%dx%dx%d",
          this->x1f.GetDim3(),
          this->x1f.GetDim2(),
          this->x1f.GetDim1()
          );
  s += std::string(tmp);

  // now append flag
  s += " : " + std::to_string(this->metadata().mask());
  s += " : " + this->metadata().maskAsString();

  return s;
}

std::string EdgeVariable::info() {
    char tmp[100] = "";

    // first add label
    //    snprintf(tmp, sizeof(tmp), "%40s : ",this->label().cstr());
    std::string s = this->label();
    s.resize(20,'.');

    // now append size
    snprintf(tmp, sizeof(tmp),
             "%dx%dx%d",
             this->x1e.GetDim3(),
             this->x1e.GetDim2(),
             this->x1e.GetDim1()
             );
    s += std::string(tmp);

    // now append flag
    s += " : " + std::to_string(this->metadata().mask());
    s += " : " + this->metadata().maskAsString();

    return s;
}

template class Variable<Real>;
} // namespace parthenon
