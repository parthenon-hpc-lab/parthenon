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
  if ( ( metadata_.Where() == Metadata::Cell) ||
       ( metadata_.Where() == Metadata::Node)) {
      // check if variable index already exists
    if (varMap_.find(varIndex) != varMap_.end()) {
      throw std::invalid_argument ("Duplicate index in create SparseVariable");
    }
    // create the variable and add to map
    std::string my_name = label_ + "_" + std::to_string(varIndex);
    auto v = std::make_shared<CellVariable<T>>(my_name, dims_, metadata_);
    varArray_.push_back(v);
    indexMap_.push_back(varIndex);
    varMap_[varIndex] = v;
  } else {
    throw std::invalid_argument ("unsupported type in SparseVariable");
  }
}

template class SparseVariable<Real>;
} // namespace parthenon
