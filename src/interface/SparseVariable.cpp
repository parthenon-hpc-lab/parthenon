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
#include <functional>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "mesh/mesh.hpp"
#include "SparseVariable.hpp"
#include "Metadata.hpp"

namespace parthenon {

/// create a new variable alias from variable 'theLabel' in input variable mv
template <typename T>
void SparseVariable<T>::AddAlias(const std::string& theLabel, SparseVariable<T>& mv) {
  // Create a new variable map
  auto& myMap = _cellVars[theLabel];

  // get a reference to the source map
  auto& theSrcMap = mv._cellVars[theLabel];

  // for every variable in the source map add an alias in current map
  for (auto& pairs : theSrcMap) {
    auto id = pairs.first;
    auto& var = *(pairs.second);
    myMap[id] = std::make_shared<Variable<T>>(theLabel, var);
  }
  _pcellVars[theLabel] = mv._pcellVars[theLabel];
  _indexMap[theLabel]  = mv._indexMap[theLabel];
}

template <typename T>
void SparseVariable<T>::AddCopy(const std::string& theLabel, SparseVariable<T>& mv) {
  // Create a new variable map
  auto& myMap = _cellVars[theLabel];
  auto& myPcell = _pcellVars[theLabel];
  auto& myIndex = _indexMap[theLabel];

  // get a reference to the source map
  auto& theSrcMap = mv._cellVars[theLabel];
  auto& theSrcVec = mv._pcellVars[theLabel];
  auto& theSrcInd = mv._indexMap[theLabel];

  // for every variable in the source map add an alias in current map
  for (auto& pairs : theSrcMap) {
    auto id = pairs.first;
    auto& var = pairs.second;
    if (var->metadata().isSet(var->metadata().oneCopy)) {
      // push an alias
      myMap[id] = var;//std::make_shared<Variable<T>>(theLabel, var);
      myPcell.push_back(var);
    } else {
      // push a copy
      myMap[id] = std::make_shared<Variable<T>>(*var);
      myPcell.push_back(myMap[id]);
    }
  }
  myIndex = theSrcInd;
}

template <typename T>
void SparseVariable<T>::Add(MeshBlock &pmb,
                              const std::string &label,
                              const Metadata &metadata,
                              const std::vector<int> &inDims) {
  // Now allocate depending on topology
  if ( ( metadata.where() == metadata.cell) ||
       ( metadata.where() == metadata.node)) {
    // check if dimensions are in range: at most 3 dimensions
    const int N = inDims.size();
    if (N > 3) {
      throw std::invalid_argument("_addArray() dims{} must have size in range [0,3]");
    }

    // Get map for label (will create one if it doesn't exist)
    auto& myMap = _cellVars[label];
    /*if (_pcellVars.find(label) == _pcellVars.end()) {
        std::cout << "inserting " << label << " into _pcellVars" << std::endl;
        _pcellVars.insert( std::pair<std::string, VariableVector<T>> (label, VariableVector<T>()) );
    }*/
    //auto& myVec = _pcellVars[label];

    // get field id from metadata
    int varIndex = metadata.getSparseID(); // FIXME

    // check if variable index already exists
    if (myMap.find(varIndex) != myMap.end()) {
      throw std::invalid_argument ("Duplicate index in create SparseVariable");
    }

    // determine size of variable needed
    int nc1 = pmb.all_cells.x.at(0).n();
    int nc2 = pmb.all_cells.x.at(1).n();
    int nc3 = pmb.all_cells.x.at(2).n();

    if ( metadata.where() == (Metadata::node) ) {
      nc1++; nc2++; nc3++;
    }

    // create array for variable dimensions
    std::array<int, 6> arrDims {nc1,nc2,nc3,1,1,1};
    for (int i=0; i<N; i++) {arrDims[i+3] = inDims[i];}


    // create the variable and add to map
    auto v = std::make_shared<Variable<T>>(label, arrDims, metadata);
    if ( metadata.fillsGhost()) {
      v->allocateComms(&pmb);
    }
    _pcellVars[label].push_back(v);
    _indexMap[label].push_back(varIndex);
    myMap.insert( std::pair<int,std::shared_ptr<Variable<T>>> (varIndex, v) );
  } else {
    throw std::invalid_argument ("unsupported type in SparseVariable");
  }
}

template <typename T>
void SparseVariable<T>::DeleteVariable(const int var_id) {
  // deletes given variable index from ALL variables
  for (auto varMap : _cellVars) {
    auto theLabel = varMap.first;
    DeleteVariable(var_id, theLabel);
  }
}

template <typename T>
void SparseVariable<T>::DeleteVariable(const int var_id, const std::string label) {
  // deletes the variable ID for only the specific variable
  // no failure if variable ID doesn't exist
  try {
    auto& myMap = this->Get(label);
    Variable<T>& vNew = Get(label, var_id);
    std::cout << "_______________________________________DELETING sparse id: "
              << label
              << std::endl;
    vNew.~Variable();
    myMap.erase(var_id);
  }
  catch (const std::invalid_argument& x) {
    // do nothing because this means that this sparse variable did not
    // have that variable id.
  }
}

template class SparseVariable<Real>;

} // namespace parthenon
