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

#include "interface/variable.hpp"

#include <iostream>

#include "bvals/cc/bvals_cc.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "parthenon_arrays.hpp"

namespace parthenon {

template <typename T>
std::string CellVariable<T>::info() {
  char tmp[100] = "";
  char *stmp = tmp;

  // first add label
  std::string s = label();
  s.resize(20, '.');
  s += " : ";

  // now append size
  snprintf(tmp, sizeof(tmp), "%dx%dx%dx%dx%dx%d", GetDim(6), GetDim(5), GetDim(4),
           GetDim(3), GetDim(2), GetDim(1));
  while (!strncmp(stmp, "1x", 2)) {
    stmp += 2;
  }
  s += stmp;
  // now append flag
  s += " : " + m_.MaskAsString();

  return s;
}

// copy constructor
template <typename T>
std::shared_ptr<CellVariable<T>>
CellVariable<T>::AllocateCopy(const bool allocComms, std::weak_ptr<MeshBlock> wpmb) {
  std::array<int, 6> dims = {GetDim(1), GetDim(2), GetDim(3),
                             GetDim(4), GetDim(5), GetDim(6)};

  // copy the Metadata and set the SharedComms flag if appropriate
  Metadata m = m_;
  if (IsSet(Metadata::FillGhost) && !allocComms) {
    m.Set(Metadata::SharedComms);
  }

  // make the new CellVariable
  auto cv = std::make_shared<CellVariable<T>>(label(), dims, m);

  if (IsSet(Metadata::FillGhost)) {
    if (allocComms) {
      cv->allocateComms(wpmb);
    } else {
      // set data pointer for the boundary communication
      // Note that vbvar->var_cc will be set when stage is selected
      cv->vbvar = vbvar;

      // fluxes, etc are always a copy
      for (int i = 1; i <= 3; i++) {
        cv->flux[i] = flux[i];
      }

      // These members are pointers,      // point at same memory as src
      cv->coarse_s = coarse_s;
    }
  }
  return cv;
}

/// allocate communication space based on info in MeshBlock
/// Initialize a 6D variable
template <typename T>
void CellVariable<T>::allocateComms(std::weak_ptr<MeshBlock> wpmb) {
  // set up fluxes
  std::string base_name = label();
  if (IsSet(Metadata::Independent)) {
    flux[X1DIR] = ParArrayND<T>(base_name + ".fluxX1", GetDim(6), GetDim(5), GetDim(4),
                                GetDim(3), GetDim(2), GetDim(1));
    if (GetDim(2) > 1)
      flux[X2DIR] = ParArrayND<T>(base_name + ".fluxX2", GetDim(6), GetDim(5), GetDim(4),
                                  GetDim(3), GetDim(2), GetDim(1));
    if (GetDim(3) > 1)
      flux[X3DIR] = ParArrayND<T>(base_name + ".fluxX3", GetDim(6), GetDim(5), GetDim(4),
                                  GetDim(3), GetDim(2), GetDim(1));
  }

  if (wpmb.expired()) return;

  std::shared_ptr<MeshBlock> pmb = wpmb.lock();

  if (pmb->pmy_mesh->multilevel)
    coarse_s = ParArrayND<T>(base_name + ".coarse", GetDim(6), GetDim(5), GetDim(4),
                             pmb->c_cellbounds.ncellsk(IndexDomain::entire),
                             pmb->c_cellbounds.ncellsj(IndexDomain::entire),
                             pmb->c_cellbounds.ncellsi(IndexDomain::entire));

  // Create the boundary object
  vbvar = std::make_shared<CellCenteredBoundaryVariable>(pmb, data, coarse_s, flux);

  // enroll CellCenteredBoundaryVariable object
  vbvar->bvar_index = pmb->pbval->bvars.size();
  // TODO(JMM): This means RestrictBoundaries()
  // is called on EVERY stage, regardless of what
  // stage needs it.
  // The fix is to refactor BoundaryValues
  // to expose calls at either the `Variable`
  // or `MeshBlockData` and `MeshData` level.
  pmb->pbval->bvars.push_back(vbvar);

  mpiStatus = false;
}

// TODO(jcd): clean these next two info routines up
template <typename T>
std::string FaceVariable<T>::info() {
  char tmp[100] = "";

  // first add label
  std::string s = this->label();
  s.resize(20, '.');
  s += " : ";

  // now append size
  snprintf(tmp, sizeof(tmp), "%dx%dx%d", data.x1f.GetDim(3), data.x1f.GetDim(2),
           data.x1f.GetDim(1));
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
  s.resize(20, '.');

  // now append size
  snprintf(tmp, sizeof(tmp), "%dx%dx%d", data.x1e.GetDim(3), data.x1e.GetDim(2),
           data.x1e.GetDim(1));
  s += std::string(tmp);

  // now append flag
  s += " : " + this->metadata().MaskAsString();

  return s;
}

template <typename T>
std::string ParticleVariable<T>::info() const {
  std::stringstream ss;

  // first add label
  std::string s = this->label();
  s.resize(20, '.');

  // combine
  ss << s << data.GetDim(1) << ":" << this->metadata().MaskAsString();

  return ss.str();
}

template class CellVariable<Real>;
template class FaceVariable<Real>;
template class EdgeVariable<Real>;

} // namespace parthenon
