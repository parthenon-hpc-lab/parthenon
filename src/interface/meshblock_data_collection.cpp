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

#include "interface/meshblock_data_collection.hpp"

#include <string>

namespace parthenon {

template <typename T>
void MeshBlockDataCollection<T>::Add(const std::string &name,
                                     const std::shared_ptr<MeshBlockData<T>> &src) {
  // error check for duplicate names
  auto it = containers_.find(name);
  if (it != containers_.end()) {
    // check to make sure they are the same
    if (!(*src == *(it->second))) {
      throw std::runtime_error("Error attempting to add a Container to a Collection");
    }
    return;
  }

  auto c = std::make_shared<MeshBlockData<T>>();
  c->SetBlockPointer(src);
  for (auto v : src->GetCellVariableVector()) {
    if (v->IsSet(Metadata::OneCopy)) {
      // just copy the (shared) pointer
      c->Add(v);
    } else {
      // allocate new storage
      c->Add(v->AllocateCopy());
    }
  }

  for (auto v : src->GetFaceVector()) {
    if (v->IsSet(Metadata::OneCopy)) {
      c->Add(v);
    } else {
      throw std::runtime_error("Non-oneCopy face variables are not yet supported");
    }
  }

  for (auto v : src->GetSparseVector()) {
    if (v->IsSet(Metadata::OneCopy)) {
      // copy the shared pointer
      c->Add(v);
    } else {
      c->Add(v->AllocateCopy());
    }
  }

  containers_[name] = c;
}

template class MeshBlockDataCollection<Real>;

} // namespace parthenon
