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
#include "athena.hpp"
#include "PropertiesInterface.hpp"
#include "SparseVariable.hpp"

namespace parthenon {

template <typename T>
void SparseVariable<T>::Add(int varIndex) {
  // Now allocate depending on topology
  if ( ( _metadata.Where() == Metadata::Cell) ||
       ( _metadata.Where() == Metadata::Node)) {
      // check if variable index already exists
    if (_varMap.find(varIndex) != _varMap.end()) {
      throw std::invalid_argument ("Duplicate index in create SparseVariable");
    }
    // create the variable and add to map
    std::string my_name = _label + "_" + PropertiesInterface::GetLabelFromID(varIndex);
    auto v = std::make_shared<CellVariable<T>>(my_name, _dims, _metadata);
    _varArray.push_back(v);
    _indexMap.push_back(varIndex);
    _varMap[varIndex] = v;
  } else {
    throw std::invalid_argument ("unsupported type in SparseVariable");
  }
}

template class SparseVariable<Real>;
} // namespace parthenon
