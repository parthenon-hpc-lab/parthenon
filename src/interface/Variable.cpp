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
#include "mesh/mesh.hpp"
#include "parthenon_arrays.hpp"
#include "Variable.hpp"

namespace parthenon {

template <typename T>
std::string CellVariable<T>::info() {
    char tmp[100] = "";
    char *stmp = tmp;

    // first add label
    std::string s = label();
    s.resize(20,'.');
    s += " : ";

    // now append size
    snprintf(tmp, sizeof(tmp),
             "%dx%dx%dx%dx%dx%d",
             GetDim(6),
             GetDim(5),
             GetDim(4),
             GetDim(3),
             GetDim(2),
             GetDim(1)
             );
    while (! strncmp(stmp,"1x",2)) {
      stmp += 2;
    }
    s += stmp;
    // now append flag
    s += " : " + _m.MaskAsString();

    return s;
  }

// copy constructor
template <typename T>
CellVariable<T>::CellVariable(const CellVariable<T> &src,
                      const bool allocComms,
                      MeshBlock *pmb) :
  mpiStatus(false), _m(src._m)  {
  data = ParArrayND<T>(src.label(), src.GetDim(6), src.GetDim(5), src.GetDim(4),
                       src.GetDim(3), src.GetDim(2), src.GetDim(1));
  if (_m.IsSet(Metadata::FillGhost)) {
    // Ghost cells are communicated, so make shallow copies
    // of communication arrays and swap out the boundary array

    if ( allocComms ) {
      allocateComms(pmb);
    } else {
      _m.Set(Metadata::SharedComms); // note that comms are shared

      // set data pointer for the boundary communication
      // Note that vbvar->var_cc will be set when stage is selected
       vbvar = src.vbvar;

      // fluxes, etc are always a copy
      for (int i = 0; i<3; i++) {
        flux[i] = src.flux[i];
      }

      // These members are pointers,
      // point at same memory as src
      //coarse_r = src.coarse_r;
      coarse_s = src.coarse_s;
    }
  }
}


/// allocate communication space based on info in MeshBlock
/// Initialize a 6D variable
template <typename T>
void CellVariable<T>::allocateComms(MeshBlock *pmb) {
  if ( ! pmb ) return;

  // set up fluxes
  std::string base_name = label();
  if (isSet(Metadata::Independent)) {
    flux[0] = ParArrayND<T>(base_name +   ".flux0", GetDim(6), GetDim(5), GetDim(4), GetDim(3), GetDim(2), GetDim(1));
    if (pmb->pmy_mesh->ndim >= 2)
      flux[1] = ParArrayND<T>(base_name + ".flux1", GetDim(6), GetDim(5), GetDim(4), GetDim(3), GetDim(2), GetDim(1));
    if (pmb->pmy_mesh->ndim >= 3)
      flux[2] = ParArrayND<T>(base_name + ".flux2", GetDim(6), GetDim(5), GetDim(4), GetDim(3), GetDim(2), GetDim(1));
  }
  // set up communication variables
  //if (pmb->pmy_mesh->multilevel)
  //  coarse_s = std::make_shared<ParArrayND<T>>(_label+".coarse", _dims[5], _dims[4], _dims[3], 
  //                                             pmb->ncc3, pmb->ncc2, pmb->ncc1);
  if (pmb->pmy_mesh->multilevel)
    coarse_s = ParArrayND<T>(base_name+".coarse", GetDim(6), GetDim(5), GetDim(4), 
                                               pmb->ncc3, pmb->ncc2, pmb->ncc1);

  // Create the boundary object
  //vbvar = std::make_shared<CellCenteredBoundaryVariable>(pmb, &data, coarse_s.get(), flux);
  vbvar = std::make_shared<CellCenteredBoundaryVariable>(pmb, data, coarse_s, flux);

  // enroll CellCenteredBoundaryVariable object
  vbvar->bvar_index = pmb->pbval->bvars.size();
  pmb->pbval->bvars.push_back(vbvar);
  pmb->pbval->bvars_main_int.push_back(vbvar);

  // register the variable
  //pmb->RegisterMeshBlockData(*this);

  mpiStatus = false;
}

// TODO(jcd): clean these next two info routines up
template <typename T>
std::string FaceVariable<T>::info() {
  char tmp[100] = "";

  // first add label
  std::string s = this->label();
  s.resize(20,'.');
  s +=  " : ";

  // now append size
  snprintf(tmp, sizeof(tmp),
          "%dx%dx%d",
          data.x1f.GetDim(3),
          data.x1f.GetDim(2),
          data.x1f.GetDim(1)
          );
  s += std::string(tmp);

  // now append flag
  s += " : " + this->metadata().MaskAsString();

  return s;
}

template <typename T>
std::string EdgeVariable<T>::info() {
    char tmp[100] = "";

    // first add label
    //    snprintf(tmp, sizeof(tmp), "%40s : ",this->label().cstr());
    std::string s = this->label();
    s.resize(20,'.');

    // now append size
    snprintf(tmp, sizeof(tmp),
             "%dx%dx%d",
             data.x1e.GetDim(3),
             data.x1e.GetDim(2),
             data.x1e.GetDim(1)
             );
    s += std::string(tmp);

    // now append flag
    s += " : " + this->metadata().MaskAsString();

    return s;
}

template class CellVariable<Real>;
template class FaceVariable<Real>;
template class EdgeVariable<Real>;
} // namespace parthenon
