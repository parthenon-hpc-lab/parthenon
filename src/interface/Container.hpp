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
#include "MaterialVariable.hpp"
#include "Stage.hpp"
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

class MeshBlock;

template <typename T>
class Container {
 public:
  //-----------------
  // Public Variables
  //-----------------
  MeshBlock* pmy_block;    // ptr to MeshBlock containing this Hydro

  //-----------------
  //Public Methods
  //-----------------
  /// Constructor initializes the "base" stage
  //Container<T>() : stages({{std::string("base"),(new Stage<T>("base"))}}),
  //                 s(stages["base"])
  //{}
  Container<T>() {
    StageAdd("base");
    StageSet("base");
  }

  // Construct a container from a specific stage of a different container
  Container<T>(std::string stageName, Container<T>&cSrc) {
    Stage<T>&stageSrc = *(cSrc.stages[stageName]);

    StageAdd("base");
    StageSet("base");

    // copy in private data
    this->pmy_block = cSrc.pmy_block;

    // add aliases of all arrays in state to the new container
    for (auto v : stageSrc._varArray) {
      this->s->_varArray.push_back(v);
    }
    // for (auto v : stageSrc._edgeArray) {
    //   EdgeVariable *vNew = new EdgeVariable(v->label(), *v);
    //   this->s->_edgeArray.push_back(vNew);
    // }
    for (auto v : stageSrc._faceArray) {
      this->s->_faceArray.push_back(v);
    }

    // Now copy in the material arrays
    for (auto vars : stageSrc._matVars.getAllCellVars()) {
      auto& theLabel=vars.first;
      auto& theMap = vars.second;
      this->s->_matVars.AddAlias(theLabel, stageSrc._matVars);
    }
  }

  /// We can initialize a container with slices from a different
  /// container.  For variables that have the material tag, this will
  /// return the material slice.  All other variables are added as
  /// is. This call returns a new container.
  ///
  /// @param mat_id The material id
  /// @return New container with slices from all variables
  Container<T> materialSlice(int mat_id);

  /// We can create a shallow copy of a container with a specific stage set as
  /// the base stage.
  /// @param src The name of the stage we want to create a new container from
  /// @return New container with slices from all variables
  Container<T> StageContainer(std::string src);

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

  ///
  /// Get a raw / cell / node variable from the container
  /// @param label the name of the variable
  /// @return the Variable<T> if found or throw exception
  Variable<T>& Get(std::string label) {
    for (auto v : s->_varArray) {
      if (! v->label().compare(label)) return *v;
    }
    throw std::invalid_argument (std::string("\n") +
                                 std::string(label) +
                                 std::string(" array not found in Get()\n") );
  }

  Variable<T>& Get(const int index) {
    return *(s->_varArray[index]);
  }

  int Index(const std::string& label) {
    for (int i = 0; i < s->_varArray.size(); i++) {
      if (! s->_varArray[i]->label().compare(label)) return i;
    }
    return -1;
  }
//  int Index(std::string label) {return Index(label);}

  // returns the material map from the Material Variables
  MaterialMap<T>& GetMaterial(const std::string& label) {
    return s->_matVars.Get(label);
  }

  ///
  /// returns the variable array for a single material for a single
  /// material variable from the Material Variables array
  Variable<T>& GetMaterial(const std::string& label, const int matID) {
    // returns the variable for the specific material from the Material Variables
    return s->_matVars.Get(label,matID);
  }

  Variable<T>& Get(const std::string& label, const int matID) { return GetMaterial(label, matID); }

  // returns a flattened array of the material variables
  VariableVector<T>& GetMaterialVector(const std::string& label) {
    return s->_matVars.GetVector(label);
  }

  std::vector<int>& GetMaterialIndexMap(const std::string& label) {
    return s->_matVars.GetIndexMap(label);
  }
  ///
  /// Get a face variable from the container
  /// @param label the name of the variable
  /// @return the FaceVariable if found or throw exception
  ///
  FaceVariable& GetFace(std::string label) {
    for (auto v : s->_faceArray) {
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
    for (auto v : s->_faceArray) {
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
    // for (auto v : s->_edgeArray) {
    //   if (! v->label().compare(label)) return v;
    // }
    throw std::invalid_argument (std::string("\n") +
                                 std::string(label) +
                                 std::string(" array not found in Get() Edge\n") );
  }

  /// Gets an array of real variables from container.
  /// @param names is the variables we want
  /// @param indexCount a map of names to std::pair<index,count> for each name
  /// @param matID if specified is list of materials we are interested in.  Note
  ///        that non-material variables specified are aliased in as is.
  int GetVariables(const std::vector<std::string>& names,
                   std::vector<Variable<T>>& vRet,
                   std::map<std::string,std::pair<int,int>>& indexCount,
                   const std::vector<int>& matID={});

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
  int size() {return s->_varArray.size();}

  // // returne variable at index
  // std::weak_ptr<Variable<T>>& at(const int index) {
  //   return s->_varArray.at(index);
  // }

  // Element accessor functions
  std::vector<std::shared_ptr<Variable<T>>>& allVars() {
    return s->_varArray;
  }

  MaterialVariable<T>& matVars() {
    return s->_matVars;
  }

  std::vector<std::shared_ptr<FaceVariable>>& faceVars() {
    return s->_faceArray;
  }

  // std::vector<EdgeVariable*> edgeVars() {
  //   return s->_edgeArray;
  // }

  // Communication routines
  void SetupPersistentMPI();
  void SetBoundaries();
  void SendBoundaryBuffers();
  void ReceiveAndSetBoundariesWithWait();
  bool ReceiveBoundaryBuffers();
  void StartReceiving(BoundaryCommSubset phase);
  void ClearBoundary(BoundaryCommSubset phase);
  void SendFluxCorrection();
  bool ReceiveFluxCorrection();

  // Stage Management
  ///Adds a stage to the stage directory
  void StageAdd(std::string name, std::string src=std::string("")) {
    if ( ! name.compare("base") ) {
      stages[name] = std::make_shared<Stage<T>>(name);
      return;
    }
    if ( ! src.compare("")) {
      src = std::string("base");
    }
    stages[name] = std::make_shared<Stage<T>>(name, *(stages[src]));
    return;
  }

  /// Sets current stage to named stage
  void StageSet(std::string name);/* {
    //    std::cout <<"_______***__________STAGE="<<name<<"_____***______________\n";
    s = stages[name];
  }*/

  /// Deletes named stage
  void StageDelete(std::string name) {
    if (s->name().compare(name) == 0) {
      StageSet(std::string("base"));
    }
    stages.erase(name);
  }

  /// Purges all stages other than 'base'
  void StagePurge() {
    auto baseStage = stages["base"];
    stages.clear();
    stages["base"] = baseStage;
  }

  void StagePrint() {
    for (auto & st : stages) {
      std::cout << "Stage " << st.first << std::endl;
    }
  }

 private:
  int debug=0;
  std::map< std::string,std::shared_ptr<Stage<T> >> stages;
  std::shared_ptr<Stage<T>> s;
};

}
#endif // INTERFACE_CONTAINER_HPP_
