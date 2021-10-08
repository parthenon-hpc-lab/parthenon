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
#include "interface/metadata.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "parthenon_arrays.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

template <typename T>
CellVariable<T>::CellVariable(const std::string &base_name, const Metadata &metadata,
                                 int sparse_id, std::weak_ptr<MeshBlock> wpmb)
    : m_(metadata), base_name_(base_name), sparse_id_(sparse_id),
      dims_(m_.GetArrayDims(wpmb, false)), coarse_dims_(m_.GetArrayDims(wpmb, true)) {
  PARTHENON_REQUIRE_THROWS(m_.IsSet(Metadata::Real),
                           "Only Real data type is currently supported for CellVariable");

  PARTHENON_REQUIRE_THROWS(IsSparse() == (sparse_id_ != InvalidSparseID),
                           "Mismatch between sparse flag and sparse ID");

  if (m_.getAssociated() == "") {
    m_.Associate(label());
  }

  if (IsSet(Metadata::FillGhost)) {
    auto pmb = wpmb.lock();
    PARTHENON_REQUIRE_THROWS(
        GetDim(4) == NumComponents(),
        "CellCenteredBoundaryVariable currently only supports rank-1 variables");
    vbvar = std::make_shared<CellCenteredBoundaryVariable>(pmb, IsSparse(), label(),
                                                           GetDim(4));
    auto res = pmb->pbval->bvars.insert({label(), vbvar});
    PARTHENON_REQUIRE_THROWS(
        res.second || (pmb->pbval->bvars.at(label()).get(), vbvar.get()),
        "A boundary variable already existed and it's different from the new one.")
  }
}

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
void CellVariable<T>::CopyFluxesAndBdryVar(const CellVariable<T> *src) {
  if (IsSet(Metadata::WithFluxes)) {
    // fluxes, coarse buffers, etc., are always a copy
    // Rely on reference counting and shallow copy of kokkos views
    flux_data_ = src->flux_data_; // reference counted
    int n_outer = 1 + (GetDim(2) > 1) * (1 + (GetDim(3) > 1));
    for (int i = X1DIR; i <= n_outer; i++) {
      flux[i] = src->flux[i]; // these are subviews
    }
  }

  if (IsSet(Metadata::FillGhost) || IsSet(Metadata::Independent)) {
    // no need to check mesh->multilevel, if false, we're just making a shallow copy of
    // an empty ParArrayND
    coarse_s = src->coarse_s;

    if (IsSet(Metadata::FillGhost)) {
      // set data pointer for the boundary communication
      // Note that vbvar->var_cc will be set when stage is selected
      vbvar = src->vbvar;
    }
  }
}

template <typename T>
std::shared_ptr<CellVariable<T>>
CellVariable<T>::AllocateCopy(std::weak_ptr<MeshBlock> wpmb) {
  // copy the Metadata
  Metadata m = m_;

  // make the new CellVariable
  auto cv = std::make_shared<CellVariable<T>>(base_name_, m, sparse_id_, wpmb);

  if (IsAllocated()) {
    cv->AllocateData();
  }
  cv->CopyFluxesAndBdryVar(this);

  return cv;
}

template <typename T>
void CellVariable<T>::Allocate(std::weak_ptr<MeshBlock> wpmb) {
  if (IsAllocated()) {
    return;
  }

  AllocateData();
  AllocateFluxesAndBdryVar(wpmb);
}

template <typename T>
void CellVariable<T>::AllocateData() {
  PARTHENON_REQUIRE_THROWS(
      !IsAllocated(),
      "Tried to allocate data for variable that's already allocated: " + label());

  data =
      ParArrayND<T>(label(), dims_[5], dims_[4], dims_[3], dims_[2], dims_[1], dims_[0]);
  is_allocated_ = true;
}

/// allocate communication space based on info in MeshBlock
/// Initialize a 6D variable
template <typename T>
void CellVariable<T>::AllocateFluxesAndBdryVar(std::weak_ptr<MeshBlock> wpmb) {
  PARTHENON_REQUIRE_THROWS(
      IsAllocated(), "Tried to allocate comms for un-allocated variable " + label());
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
      coarse_s = ParArrayND<T>(base_name + ".coarse", coarse_dims_[5], coarse_dims_[4],
                               coarse_dims_[3], coarse_dims_[2], coarse_dims_[1],
                               coarse_dims_[0]);
    }

    if (IsSet(Metadata::FillGhost)) {
      vbvar->Reset(data, coarse_s, flux);

      // TODO(JMM): This means RestrictBoundaries()
      // is called on EVERY stage, regardless of what
      // stage needs it.
      // The fix is to refactor BoundaryValues
      // to expose calls at either the `Variable`
      // or `MeshBlockData` and `MeshData` level.
      auto res = pmb->pbval->bvars.insert({label(), vbvar});
      PARTHENON_REQUIRE_THROWS(
          res.second || (pmb->pbval->bvars.at(label()).get(), vbvar.get()),
          "A boundary variable already existed and it's different from the new one.")
    }
  }

  mpiStatus = false;
}

template <typename T>
void CellVariable<T>::Deallocate() {
  if (!IsAllocated()) {
    return;
  }

  data.Reset();

  if (IsSet(Metadata::WithFluxes)) {
    Kokkos::resize(flux_data_, 0, 0, 0, 0, 0, 0, 0);
    int n_outer = 1 + (GetDim(2) > 1) * (1 + (GetDim(3) > 1));
    for (int d = X1DIR; d <= n_outer; ++d) {
      flux[d].Reset();
    }
  }

  if (IsSet(Metadata::FillGhost) || IsSet(Metadata::Independent)) {
    coarse_s.Reset();
  }

  is_allocated_ = false;
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
