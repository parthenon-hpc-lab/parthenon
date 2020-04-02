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
#ifndef INTERFACE_CONTAINERITERATOR_HPP_
#define INTERFACE_CONTAINERITERATOR_HPP_
///
/// Provides an iterator that iterates over the variables in a
/// container.  Eventually this will get transitioned to an iterator
/// type in the Container itself, but for now we have to do it this
/// way because Sriram doesn't know enough C++ to do this correctly.
///
//#include <iterator>
#include <array>
#include <memory>
#include <vector>
#include "interface/PropertiesInterface.hpp"
#include "Container.hpp"
#include "Variable.hpp"

namespace parthenon {
template <typename T>
class ContainerIterator {
 public:
  /// the subset of variables that match this iterator
  VariableVector<T> vars;
  //std::vector<FaceVariable> varsFace; // face vars that match
  //std::vector<EdgeVariable> varsEdge; // edge vars that match

  /// initializes the iterator with a container and a flag to match
  /// @param c the container on which you want the iterator
  /// @param flagVector: a vector of Metadata::flags that you want to match
  ContainerIterator<T>(Container<T>& c, const std::vector<MetadataFlag> &flagVector) {
    //c.print();
    auto allVars = c.GetVariableVector();
    for (auto & svar : c.GetSparseVector()) {
      VariableVector<T>& svec = svar->GetVector();
      allVars.insert(allVars.end(), svec.begin(), svec.end());
    }
    // faces not active yet    _allFaceVars = c.faceVars();
    // edges not active yet    _allEdgeVars = c.edgeVars();
    setMask(allVars, flagVector); // fill subset based on mask vector
  }

  //~ContainerIterator<T>() {
  //  _emptyVars();
  //}
  /// Changes the mask for the iterator and resets the iterator
  /// @param flagArray: a vector of MetadataFlag that you want to match
  void setMask(VariableVector<T>& allVars, const std::vector<MetadataFlag> &flagVector) {
    // 1: clear out variables stored so far
    _emptyVars();

    // 2: fill in the subset of variables that match mask
    for (auto pv : allVars) {
      if (pv->metadata().AnyFlagsSet(flagVector)) {
        vars.push_back(pv);
      }
    }
  }


 private:
  uint64_t _mask;
  FaceVector<T> _allFaceVars = {};
  //EdgeVector<T> _allEdgeVars = {};
  //VariableVector<T> _allVars = {};
  void _emptyVars() {
    vars.clear();
  //  varsFace.clear();
  //  varsEdge.clear();
  }
  static bool couldBeEdge(const std::vector<MetadataFlag> &flagVector) {
    // returns true if face is set or if no topology set
    for (auto &f : flagVector) {
      if ( f == Metadata::Edge) return true;
      else if ( f == Metadata::Cell) return false;
      else if ( f == Metadata::Face) return false;
      else if ( f == Metadata::Node) return false;
    }
    return true;
  }
  static bool couldBeFace(const std::vector<MetadataFlag> &flagVector) {
    // returns true if face is set or if no topology set
    for (auto &f : flagVector) {
      if ( f == Metadata::Face) return true;
      else if ( f == Metadata::Cell) return false;
      else if ( f == Metadata::Edge) return false;
      else if ( f == Metadata::Node) return false;
    }
    return true;
  }
};
}
#endif // INTERFACE_CONTAINERITERATOR_HPP_
