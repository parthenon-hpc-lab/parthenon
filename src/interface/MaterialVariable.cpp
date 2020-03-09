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
#include "MaterialVariable.hpp"
#include "Metadata.hpp"

namespace parthenon {
/// create a new material variable alias from material variable
/// 'theLabel' in input material variable mv
template <typename T>
void MaterialVariable<T>::AddAlias(const std::string& theLabel, MaterialVariable<T>& mv) {
  // Create a new variable map
  auto& myMap = _cellVars[theLabel];

  // get a reference to the source map
  auto& theSrcMap = mv._cellVars[theLabel];

  // for every material in the source map add an alias in current map
  for (auto& matPairs : theSrcMap) {
    auto matID = matPairs.first;
    auto& matVar = *(matPairs.second);
    myMap[matID] = std::make_shared<Variable<T>>(theLabel, matVar);
  }
  _pcellVars[theLabel] = mv._pcellVars[theLabel];
  _indexMap[theLabel]  = mv._indexMap[theLabel];
}

template <typename T>
void MaterialVariable<T>::AddCopy(const std::string& theLabel, MaterialVariable<T>& mv) {
  // Create a new variable map
  auto& myMap = _cellVars[theLabel];
  auto& myPcell = _pcellVars[theLabel];
  auto& myIndex = _indexMap[theLabel];

  // get a reference to the source map
  auto& theSrcMap = mv._cellVars[theLabel];
  auto& theSrcVec = mv._pcellVars[theLabel];
  auto& theSrcInd = mv._indexMap[theLabel];

  // for every material in the source map add an alias in current map
  for (auto& matPairs : theSrcMap) {
    auto matID = matPairs.first;
    auto& matVar = matPairs.second;
    if (matVar->metadata().isSet(matVar->metadata().oneCopy)) {
      // push an alias
      myMap[matID] = matVar;//std::make_shared<Variable<T>>(theLabel, matVar);
      myPcell.push_back(matVar);
    } else {
      // push a copy
      myMap[matID] = std::make_shared<Variable<T>>(*matVar);
      myPcell.push_back(myMap[matID]);
    }
  }
  myIndex = theSrcInd;
}

template <typename T>
void MaterialVariable<T>::Add(MeshBlock &pmb,
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

    // get material id from metadata
    int matIndex = metadata.getMaterialID();

    // check if material already exists in variable
    if (myMap.find(matIndex) != myMap.end()) {
      throw std::invalid_argument ("Duplicate material index in create MaterialVariable");
    }

    // determine size of variable needed
    int nc1 = pmb.ncells1;
    int nc2 = pmb.ncells2;
    int nc3 = pmb.ncells3;

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
    _indexMap[label].push_back(matIndex);
    myMap.insert( std::pair<int,std::shared_ptr<Variable<T>>> (matIndex, v) );
  } else {
    throw std::invalid_argument ("unsupported type in MaterialVariable");
  }
}

template <typename T>
void MaterialVariable<T>::DeleteMaterial(const int mat_id) {
  // deletes given material from ALL variables
  for (auto varMap : _cellVars) {
    auto theLabel = varMap.first;
    DeleteMaterial(mat_id, theLabel);
  }
}

template <typename T>
void MaterialVariable<T>::DeleteMaterial(const int mat_id, const std::string label) {
  // deletes the material ID for only the specific variable
  // no failure if material doesn't exist
  try {
    auto& myMap = this->Get(label);
    Variable<T>& vNew = Get(label, mat_id);
    std::cout << "_______________________________________DELETING mat: "
              << label
              << std::endl;
    vNew.~Variable();
    myMap.erase(mat_id);
  }
  catch (const std::invalid_argument& x) {
    // do nothing because this means that this variable did not
    // have that material id.
  }
}


template class MaterialVariable<Real>;
}
