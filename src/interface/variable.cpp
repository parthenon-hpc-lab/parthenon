//========================================================================================
// (C) (or copyright) 2020-2021. Triad National Security, LLC. All rights reserved.
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
#include <utility>

#include "bvals/cc/bvals_cc.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "parthenon_arrays.hpp"
#include "utils/error_checking.hpp"

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
CellVariable<T>::AllocateCopy(const bool alloc_separate_fluxes_and_bvar,
                              std::weak_ptr<MeshBlock> wpmb) {
  // copy the Metadata
  Metadata m = m_;

  // make the new CellVariable
  auto cv = std::make_shared<CellVariable<T>>(label(), m, sparse_id_);

  if (is_allocated_) {
    cv->AllocateData(wpmb);
  }

  if (IsSet(Metadata::FillGhost)) {
    if (alloc_separate_fluxes_and_bvar) {
      cv->AllocateFluxesAndBdryVar(wpmb);
    } else {
      // set data pointer for the boundary communication
      // Note that vbvar->var_cc will be set when stage is selected
      cv->vbvar = vbvar;

      // fluxes, coarse buffers, etc., are always a copy
      // Rely on reference counting and shallow copy of kokkos views
      cv->flux_data_ = flux_data_; // reference counted
      for (int i = 1; i <= 3; i++) {
        cv->flux[i] = flux[i]; // these are subviews
      }
    }

    // These members are pointers,      // point at same memory as src
    cv->coarse_s = coarse_s;
  }

  return cv;
}

template <typename T>
void CellVariable<T>::Allocate(std::weak_ptr<MeshBlock> wpmb) {
  AllocateData(wpmb);
  AllocateFluxesAndBdryVar(wpmb);
}

template <typename T>
void CellVariable<T>::AllocateData(std::weak_ptr<MeshBlock> wpmb) {
  if (is_allocated_) {
    return;
  }

  if (m_.IsMeshTied() && wpmb.expired()) {
    // we can't allocate data because we need cellbounds from the meshblock but our
    // meshblock pointer is null
    return;
  }

  const auto dims = m_.GetArrayDims(wpmb);
  data = ParArrayND<T>(label(), dims[5], dims[4], dims[3], dims[2], dims[1], dims[0]);
  is_allocated_ = true;
}

/// allocate communication space based on info in MeshBlock
/// Initialize a 6D variable
template <typename T>
void CellVariable<T>::AllocateFluxesAndBdryVar(std::weak_ptr<MeshBlock> wpmb) {
  PARTHENON_REQUIRE_THROWS(
      is_allocated_, "Tried to allocate comms for un-allocated variable " + label());
  std::string base_name = label();

  // TODO(JMM): Note that this approach assumes LayoutRight. Otherwise
  // the stride will mess up the types.

  if (IsSet(Metadata::WithFluxes)) {
    // Compute size of unified flux_data object and create it. A unified
    // flux_data_ object reduces the number of memory allocations per
    // variable per meshblock from 5 to 3.
    int n_outer = 1 + (GetDim(2) > 1) * (1 + (GetDim(3) > 1));
    // allocate fluxes
    flux_data_ = ParArray7D<T>(base_name + ".flux_data", n_outer, GetDim(6), GetDim(5),
                               GetDim(4), GetDim(3), GetDim(2), GetDim(1));
    // set up fluxes
    for (int d = X1DIR; d <= n_outer; ++d) {
      flux[d] = ParArrayND<T>(Kokkos::subview(flux_data_, d - 1, Kokkos::ALL(),
                                              Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(),
                                              Kokkos::ALL(), Kokkos::ALL()));
    }
  }

  // Create the boundary object
  if (IsSet(Metadata::FillGhost) || IsSet(Metadata::Independent)) {
    if (wpmb.expired()) return;
    std::shared_ptr<MeshBlock> pmb = wpmb.lock();

    if (pmb->pmy_mesh != nullptr && pmb->pmy_mesh->multilevel) {
      coarse_s = ParArrayND<T>(base_name + ".coarse", GetDim(6), GetDim(5), GetDim(4),
                               pmb->c_cellbounds.ncellsk(IndexDomain::entire),
                               pmb->c_cellbounds.ncellsj(IndexDomain::entire),
                               pmb->c_cellbounds.ncellsi(IndexDomain::entire));
    }

    if (IsSet(Metadata::FillGhost)) {
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
    }
  }

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
