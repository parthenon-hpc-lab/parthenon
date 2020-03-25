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
#ifndef INTERFACE_CONTAINER_HPP_
#define INTERFACE_CONTAINER_HPP_

#include <map>
#include <memory>
#include <string>
#include <utility> // <pair>
#include <vector>
#include "globals.hpp"
//#include "mesh/mesh.hpp"
#include "SparseVariable.hpp"
#include "Variable.hpp"

namespace parthenon {
///
/// Interface to underlying infrastructure for data declaration and access.
/// Date: August 22, 2019
///
///
/// The container class is a container for the variables that make up
/// the simulation.  At this point it is expected that this includes
/// both simulation parameters and state variables, but that could
/// change in the future.
///
/// The container class will provide the following methods:
///

using FaceArray = std::vector<std::shared_ptr<FaceVariable>>;
template <typename T>
using SparseArray = std::vector<std::shared_ptr<SparseVariable<T>>>;

template <typename T>
using MapToVars = std::map<std::string, std::shared_ptr<Variable<T>>>;
using MapToFace = std::map<std::string, std::shared_ptr<FaceVariable>>;
template <typename T>
using MapToSparse = std::map<std::string, std::shared_ptr<SparseVariable<T>>>;

class MeshBlock;

template <typename T>
class Container {
 public:
  //-----------------
  // Public Variables
  //-----------------
  MeshBlock* pmy_block = nullptr; // ptr to MeshBlock containing this Hydro

  //-----------------
  //Public Methods
  //-----------------
  /// Constructor
  Container<T>() = default;

  /// We can initialize a container with slices from a different
  /// container.  For variables that have the sparse tag, this will
  /// return the sparse slice.  All other variables are added as
  /// is. This call returns a new container.
  ///
  /// @param sparse_id The sparse id
  /// @return New container with slices from all variables
  Container<T> sparseSlice(int sparse_id);

  ///
  /// Set the pointer to the mesh block for this container
  void setBlock(MeshBlock *pmb) { pmy_block = pmb; }

  ///
  /// Allocate and add a variable<T> to the container
  ///
  /// This function will eventually look at the metadata flags to
  /// identify the size of the first dimension based on the
  /// topological location.
  ///
  /// @param label the name of the variable
  /// @param metadata the metadata associated with the variable
  /// @param dims the size of each element
  ///
  void Add(const std::string label,
           const Metadata &metadata,
           const std::vector<int> dims);

  ///
  /// Allocate and add a variable<T> to the container
  ///
  /// This function will eventually look at the metadata flags to
  /// identify the size of the first dimension based on the
  /// topological location.
  ///
  /// @param labelArray the array of names of variables
  /// @param metadata the metadata associated with the variable
  /// @param dims the size of each element
  ///
  void Add(const std::vector<std::string> labelArray,
           const Metadata &metadata,
           const std::vector<int> dims);

  ///
  /// Allocate and add a variable<T> to the container
  ///
  /// This function will eventually look at the metadata flags to
  /// identify the size of the first dimension based on the
  /// topological location.  Dimensions will be taken from the metadata.
  ///
  /// @param label the name of the variable
  /// @param metadata the metadata associated with the variable
  ///
  void Add(const std::string label, const Metadata &metadata);

  ///
  /// Allocate and add a variable<T> to the container
  ///
  /// This function will eventually look at the metadata flags to
  /// identify the size of the first dimension based on the
  /// topological location.  Dimensions will be taken from the metadata.
  ///
  /// @param labelArray the array of names of variables
  /// @param metadata the metadata associated with the variable
  ///
  void Add(const std::vector<std::string> labelArray, const Metadata &metadata);

  void Add(std::shared_ptr<Variable<T>> var) { _varArray.push_back(var); }
  void Add(std::shared_ptr<FaceVariable> var) { _faceArray.push_back(var);  }
  void Add(std::shared_ptr<SparseVariable<T>> var) {
    // TODO(jcd): deal with adding var to map
    _sparseArray.push_back(var); 
  }

  ///
  /// Get a raw / cell / node variable from the container
  /// @param label the name of the variable
  /// @return the Variable<T> if found or throw exception
  Variable<T>& Get(std::string label) {
    for (auto v : _varArray) {
      if (! v->label().compare(label)) return *v;
    }
    throw std::invalid_argument (std::string("\n") +
                                 std::string(label) +
                                 std::string(" array not found in Get()\n") );
  }

  Variable<T>& Get(const int index) {
    return *(_varArray[index]);
  }

  int Index(const std::string& label) {
    for (int i = 0; i < _varArray.size(); i++) {
      if (! _varArray[i]->label().compare(label)) return i;
    }
    return -1;
  }
//  int Index(std::string label) {return Index(label);}

  // returns the sparse map from the Sparse Variables
  SparseVariable<T>& GetSparseVariable(const std::string& label) {
    if (_sparseMap.find(label) == _sparseMap.end()) {
      throw std::invalid_argument("_sparseMap does not have " + label);
    }
    return *(_sparseMap[label]);
  }

  SparseMap<T>& GetSparseMap(const std::string& label) {
    return GetSparseVariable(label).GetMap();
  }

  VariableVector<T>& GetSparseVector(const std::string& label) {
    return GetSparseVariable(label).GetVector();
  }

  Variable<T>& Get(const std::string& label, const int sparse_id) {
    return GetSparseVariable(label).Get(sparse_id);
  }

  std::vector<int>& GetSparseIndexMap(const std::string& label) {
    return GetSparseVariable(label).GetIndexMap();
  }

  ///
  /// Get a face variable from the container
  /// @param label the name of the variable
  /// @return the FaceVariable if found or throw exception
  ///
  FaceVariable& GetFace(std::string label) {
    for (auto v : _faceArray) {
      if (! v->label().compare(label)) return *v;
    }
    throw std::invalid_argument (std::string("\n") +
                                 std::string(label) +
                                 std::string(" array not found in Get() Face\n") );
  }

    ///
  /// Get a face variable from the container
  /// @param label the name of the variable
  /// @param dir, which direction the face is normal to
  /// @return the AthenaArray in the face variable if found or throw exception
  ///
  AthenaArray<Real>& GetFace(std::string label, int dir) {
    for (auto v : _faceArray) {
      if (! v->label().compare(label)) return v->Get(dir);
    }
    throw std::invalid_argument (std::string("\n") +
                                 std::string(label) +
                                 std::string(" array not found in Get() Face\n") );
  }

  ///
  /// Get an edge variable from the container
  /// @param label the name of the variable
  /// @return the Variable<T> if found or throw exception
  ///
  EdgeVariable *GetEdge(std::string label) {
    // for (auto v : _edgeArray) {
    //   if (! v->label().compare(label)) return v;
    // }
    throw std::invalid_argument (std::string("\n") +
                                 std::string(label) +
                                 std::string(" array not found in Get() Edge\n") );
  }

  /// Gets an array of real variables from container.
  /// @param names is the variables we want
  /// @param indexCount a map of names to std::pair<index,count> for each name
  /// @param sparse_ids if specified is list of sparse ids we are interested in.  Note
  ///        that non-sparse variables specified are aliased in as is.
  int GetVariables(const std::vector<std::string>& names,
                   std::vector<Variable<T>>& vRet,
                   std::map<std::string,std::pair<int,int>>& indexCount,
                   const std::vector<int>& sparse_ids = {});

  ///
  /// get raw data for a variable from the container
  /// @param label the name of the variable
  /// @return a pointer of type T if found or NULL
  T *Raw(std::string label) {
    Variable<T>& v = Get(label);
    //if(v)
    return v.data();
    //return NULL;
  }

  ///
  /// Remove a variable from the container or throw exception if not
  /// found.
  /// @param label the name of the variable to be deleted
  void Remove(const std::string label);


  // Temporary functions till we implement a *real* iterator

  /// Print list of labels in container
  void print();

  // return number of stored arrays
  int size() {return _varArray.size();}

  // // returne variable at index
  // std::weak_ptr<Variable<T>>& at(const int index) {
  //   return _varArray.at(index);
  // }

  // Element accessor functions
  VariableVector<T>& allVars() {
    return _varArray;
  }

  SparseArray<T>& sparseVars() {
    return _sparseArray;
  }

  FaceArray& faceVars() {
    return _faceArray;
  }

  // std::vector<EdgeVariable*> edgeVars() {
  //   return _edgeArray;
  // }

  // Communication routines
  void ResetBoundaryVariables();
  void SetupPersistentMPI();
  void SetBoundaries();
  void SendBoundaryBuffers();
  void ReceiveAndSetBoundariesWithWait();
  bool ReceiveBoundaryBuffers();
  void StartReceiving(BoundaryCommSubset phase);
  void ClearBoundary(BoundaryCommSubset phase);
  void SendFluxCorrection();
  bool ReceiveFluxCorrection();

 private:
  int debug=0;
  
  VariableVector<T> _varArray = {}; ///< the saved variable array
  FaceArray _faceArray = {};  ///< the saved face arrays
  SparseArray<T> _sparseArray = {};

  MapToVars<T> _varMap = {};
  MapToFace _faceMap = {};
  MapToSparse<T> _sparseMap = {};

  void calcArrDims_(std::array<int, 6>& arrDims,
                    const std::vector<int>& dims);
};

} // namespace parthenon
#endif // INTERFACE_CONTAINER_HPP_
