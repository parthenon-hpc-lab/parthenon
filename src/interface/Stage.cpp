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
#include "mesh/mesh.hpp"
#include "Stage.hpp"

/// The implementation file for class Stage
template <typename T>
Stage<T>::Stage(std::string name, Stage<T>& src) :
    _name(name), locked(true) {
  if ( ! name.compare(src.name())) {
    throw std::invalid_argument ("Duplicate stage name in copy constructor");
  }

  for ( auto v : src._varArray) {
    const Metadata &m = v->metadata();
    if (m.isSet(m.oneCopy)) {
      // push back an alias of the variable
      _varArray.push_back(v);//std::make_shared<Variable<T>>(v->label(),*v));
    } else {
      // create a deepcopy with shared fluxes
      _varArray.push_back(std::make_shared<Variable<T>>(*v));
    }
  }

  // // for now faces and edges are oneCopy
  // for (auto v : src._edgeArray) {
  //   EdgeVariable *vNew = new EdgeVariable(v->label(), *v);
  //   _edgeArray.push_back(vNew);
  // }
  // for (auto v : src._faceArray) {
  //   FaceVariable *vNew = new FaceVariable(v->label(), *v);
  //   _faceArray.push_back(vNew);
  // }

  // Now copy in the material arrays
  for (auto vars : src._matVars.getAllCellVars()) {
    auto& theLabel=vars.first;
    _matVars.AddCopy(theLabel, src._matVars);
  }
}

template class Stage<Real>;
