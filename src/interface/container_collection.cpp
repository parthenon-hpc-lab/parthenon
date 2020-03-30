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

#include <string>

#include "container_collection.hpp"
#include "Container.hpp"

namespace parthenon {

template <typename T>
void ContainerCollection<T>::Add(const std::string& name, Container<T>& src) {
  // error check for duplicate names

  auto c = Container<T>();
  c.pmy_block = src.pmy_block;
  for (auto v : src.GetVariableVector()) {
    if (v->isSet(Metadata::OneCopy)) {
      c.Add(v);
    } else {
      c.Add( std::make_shared<Variable<T>>(*v) );
    }
  }

  for (auto v : src.GetFaceVector()) {
    if (v->isSet(Metadata::OneCopy)) {
      c.Add(v);
    } else {
      throw std::runtime_error("Non-oneCopy face variables are not yet supported");
    }
  }

  for (auto v : src.GetSparseVector()) {
    if (v->isSet(Metadata::OneCopy)) {
      c.Add(v);
    } else {
      c.Add( std::make_shared<SparseVariable<T>>(*v) );
    }
  }

  containers_[name] = std::move(c);

}

template class ContainerCollection<Real>;

} // namespace parthenon
