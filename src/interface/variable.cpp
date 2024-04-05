//========================================================================================
// (C) (or copyright) 2020-2023. Triad National Security, LLC. All rights reserved.
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
#include <memory>
#include <tuple>
#include <utility>

#include "interface/metadata.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "parthenon_arrays.hpp"
#include "utils/array_to_tuple.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

template <typename T>
Variable<T>::Variable(const std::string &base_name, const Metadata &metadata,
                      int sparse_id, std::weak_ptr<MeshBlock> wpmb)
    : m_(metadata), base_name_(base_name), sparse_id_(sparse_id),
      dims_(m_.GetArrayDims(wpmb, false)), coarse_dims_(m_.GetArrayDims(wpmb, true)) {
  PARTHENON_REQUIRE_THROWS(m_.IsSet(Metadata::Real),
                           "Only Real data type is currently supported for Variable");

  PARTHENON_REQUIRE_THROWS(IsSparse() == (sparse_id_ != InvalidSparseID),
                           "Mismatch between sparse flag and sparse ID");
  uid_ = get_uid_(label());

  if (m_.getAssociated() == "") {
    m_.Associate(label());
  }
}

template <typename T>
std::string Variable<T>::info() {
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

// Makes a shallow copy of the boundary buffer and fluxes of the source variable and
// assign them to this variable
template <typename T>
void Variable<T>::CopyFluxesAndBdryVar(const Variable<T> *src) {
  if (IsSet(Metadata::FillGhost) || IsSet(Metadata::Independent) ||
      IsSet(Metadata::ForceRemeshComm) || IsSet(Metadata::Flux)) {
    // no need to check mesh->multilevel, if false, we're just making a shallow copy of
    // an empty ParArrayND
    coarse_s = src->coarse_s;
  }
}

template <typename T>
std::shared_ptr<Variable<T>> Variable<T>::AllocateCopy(std::weak_ptr<MeshBlock> wpmb) {
  // copy the Metadata
  Metadata m = m_;

  // make the new Variable
  auto cv = std::make_shared<Variable<T>>(base_name_, m, sparse_id_, wpmb);

  if (is_allocated_) {
    cv->AllocateData(wpmb);
  }

  cv->CopyFluxesAndBdryVar(this);

  return cv;
}

template <typename T>
void Variable<T>::Allocate(std::weak_ptr<MeshBlock> wpmb, bool flag_uninitialized) {
  if (is_allocated_) {
    return;
  }

  AllocateData(wpmb, flag_uninitialized);
  AllocateCoarse(wpmb);
}

template <typename T>
void Variable<T>::AllocateData(MeshBlock *pmb, bool flag_uninitialized) {
  PARTHENON_REQUIRE_THROWS(
      !is_allocated_,
      "Tried to allocate data for variable that's already allocated: " + label());
  data = std::make_from_tuple<ParArrayND<T, VariableState>>(std::tuple_cat(
      std::make_tuple(label(), MakeVariableState()), ArrayToReverseTuple(dims_)));

  ++num_alloc_;

  data.initialized = !flag_uninitialized;
  is_allocated_ = true;

  if (pmb != nullptr) {
    pmb->LogMemUsage(data.size() * sizeof(T));
  }
}
template <typename T>
void Variable<T>::AllocateData(std::weak_ptr<MeshBlock> wpmb, bool flag_uninitialized) {
  // if wpmb is expired, then wpmb.lock().get() == nullptr
  AllocateData(wpmb.lock().get(), flag_uninitialized);
}

/// allocate communication space based on info in MeshBlock
/// Initialize a 6D variable
template <typename T>
void Variable<T>::AllocateCoarse(std::weak_ptr<MeshBlock> wpmb) {
  PARTHENON_REQUIRE_THROWS(
      IsAllocated(), "Tried to allocate coarse for un-allocated variable " + label());
  std::string base_name = label();

  // Create the boundary object
  if (IsSet(Metadata::FillGhost) || IsSet(Metadata::Independent) ||
      IsSet(Metadata::ForceRemeshComm) || IsSet(Metadata::Flux)) {
    if (wpmb.expired()) return;
    std::shared_ptr<MeshBlock> pmb = wpmb.lock();

    if (pmb->pmy_mesh != nullptr && pmb->pmy_mesh->multilevel) {
      coarse_s = std::make_from_tuple<ParArrayND<T, VariableState>>(
          std::tuple_cat(std::make_tuple(label() + ".coarse", MakeVariableState()),
                         ArrayToReverseTuple(coarse_dims_)));
      pmb->LogMemUsage(coarse_s.size() * sizeof(T));
    }
  }
}

template <typename T>
std::int64_t Variable<T>::Deallocate() {
  std::int64_t mem_size = 0;
#ifdef ENABLE_SPARSE
  if (!IsAllocated()) {
    return 0;
  }

  mem_size += data.size() * sizeof(T);
  data.Reset();

  if (IsSet(Metadata::FillGhost) || IsSet(Metadata::Independent) ||
      IsSet(Metadata::ForceRemeshComm) || IsSet(Metadata::Flux)) {
    mem_size += coarse_s.size() * sizeof(T);
    coarse_s.Reset();
  }

  is_allocated_ = false;
  return mem_size;
#else
  PARTHENON_THROW("Variable<T>::Deallocate(): Sparse is compile-time disabled");
  return 0;
#endif
}

template <typename T>
ParticleVariable<T>::ParticleVariable(const std::string &label, const int npool,
                                      const Metadata &metadata)
    : m_(metadata), label_(label),
      dims_(m_.GetArrayDims(std::weak_ptr<MeshBlock>(), false)),
      data(label_, dims_[5], dims_[4], dims_[3], dims_[2], dims_[1], npool) {
  dims_[0] = npool;
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

template class Variable<Real>;
template class ParticleVariable<Real>;
template class ParticleVariable<int>;
template class ParticleVariable<bool>;

} // namespace parthenon
