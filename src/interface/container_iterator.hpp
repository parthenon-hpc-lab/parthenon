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
#ifndef INTERFACE_CONTAINER_ITERATOR_HPP_
#define INTERFACE_CONTAINER_ITERATOR_HPP_

/// Provides an iterator that iterates over the variables in a
/// container.  Eventually this will get transitioned to an iterator
/// type in the Container itself, but for now we have to do it this
/// way because Sriram doesn't know enough C++ to do this correctly.

#include <array>
#include <forward_list>
#include <memory>
#include <string>
#include <vector>

#include "interface/container.hpp"
#include "interface/properties_interface.hpp"
#include "interface/variable.hpp"

namespace parthenon {

template <typename T>
class VariablePack {
 public:
  VariablePack(T view, std::array<int, 4> dims) : v_(view), dims_(dims) {}
  KOKKOS_FORCEINLINE_FUNCTION
  auto &operator()(const int n) { return v_(n); }
  KOKKOS_FORCEINLINE_FUNCTION
  auto &operator()(const int n, const int k, const int j, const int i) const {
    return v_(n)(k, j, i);
  }
  KOKKOS_FORCEINLINE_FUNCTION
  int GetDim(const int i) {
    assert(i > 0 && i < 5);
    return dims_[i - 1];
  }

 private:
  const T v_;
  const std::array<int, 4> dims_;
};

template <typename T>
using VarList = std::forward_list<std::shared_ptr<CellVariable<T>>>;

template <typename T>
auto MakePack(VarList<T> &vars) {
  // count up the size
  int vsize = 0;
  for (const auto &v : vars) {
    vsize += v->GetDim(6) * v->GetDim(5) * v->GetDim(4);
  }
  auto fvar = vars.front()->data;
  auto slice = fvar.Get(0, 0, 0);
  auto cv = Kokkos::View<decltype(slice) *>("MakePack::cv", vsize);
  auto host_view = Kokkos::create_mirror_view(cv);
  int vindex = 0;
  for (const auto &v : vars) {
    for (int k = 0; k < v->GetDim(6); k++) {
      for (int j = 0; j < v->GetDim(5); j++) {
        for (int i = 0; i < v->GetDim(4); i++) {
          host_view(vindex++) = v->data.Get(k, j, i);
        }
      }
    }
  }
  Kokkos::deep_copy(cv, host_view);
  std::array<int, 4> cv_size = {fvar.GetDim(1), fvar.GetDim(2), fvar.GetDim(3), vsize};
  return VariablePack<decltype(cv)>(cv, cv_size);
}

template <typename T>
auto PackVariables(const Container<T> &c, const std::vector<MetadataFlag> &flags) {
  VarList<T> vars;
  for (const auto &v : c.GetCellVariableVector()) {
    if (v->metadata().AnyFlagsSet(flags)) {
      vars.push_front(v);
    }
  }
  for (const auto &sv : c.GetSparseVector()) {
    if (sv->metadata().AnyFlagsSet(flags)) {
      for (const auto &v : sv->GetVector()) {
        vars.push_front(v);
      }
    }
  }

  return MakePack<T>(vars);
}

template <typename T>
auto PackVariables(const Container<T> &c, const std::vector<std::string> &names) {
  VarList<T> vars;
  for (const auto &v : c.GetCellVariableVector()) {
    if (std::find(names.begin(), names.end(), v->label()) != names.end()) {
      vars.push_front(v);
    }
  }

  for (const auto &sv : c.GetSparseVector()) {
    if (std::find(names.begin(), names.end(), sv->label()) != names.end())
      for (const auto &v : sv->GetVector()) {
        vars.push_front(v);
      }
  }

  return MakePack<T>(vars);
}

template <typename T>
class ContainerIterator {
 public:
  /// the subset of variables that match this iterator
  CellVariableVector<T> vars;
  // std::vector<FaceVariable> varsFace; // face vars that match
  // std::vector<EdgeVariable> varsEdge; // edge vars that match

  /// initializes the iterator with a container and a flag to match
  /// @param c the container on which you want the iterator
  /// @param flags: a vector of Metadata::flags that you want to match
  ContainerIterator<T>(const Container<T> &c, const std::vector<MetadataFlag> &flags) {
    // c.print();
    auto allVars = c.GetCellVariableVector();
    for (auto &svar : c.GetSparseVector()) {
      CellVariableVector<T> &svec = svar->GetVector();
      allVars.insert(allVars.end(), svec.begin(), svec.end());
    }
    // faces not active yet    _allFaceVars = c.faceVars();
    // edges not active yet    _allEdgeVars = c.edgeVars();
    setMask(allVars, flags); // fill subset based on mask vector
  }

  //~ContainerIterator<T>() {
  //  _emptyVars();
  //}
  /// Changes the mask for the iterator and resets the iterator
  /// @param flagArray: a vector of MetadataFlag that you want to match
  void setMask(const CellVariableVector<T> &allVars,
               const std::vector<MetadataFlag> &flags) {
    // 1: clear out variables stored so far
    _emptyVars();

    // 2: fill in the subset of variables that match mask
    for (auto pv : allVars) {
      if (pv->metadata().AnyFlagsSet(flags)) {
        vars.push_back(pv);
      }
    }
  }

 private:
  uint64_t _mask;
  FaceVector<T> _allFaceVars = {};
  // EdgeVector<T> _allEdgeVars = {};
  // CellVariableVector<T> _allVars = {};
  void _emptyVars() {
    vars.clear();
    //  varsFace.clear();
    //  varsEdge.clear();
  }
  static bool couldBeEdge(const std::vector<MetadataFlag> &flags) {
    // returns true if face is set or if no topology set
    for (auto &f : flags) {
      if (f == Metadata::Edge)
        return true;
      else if (f == Metadata::Cell)
        return false;
      else if (f == Metadata::Face)
        return false;
      else if (f == Metadata::Node)
        return false;
    }
    return true;
  }
  static bool couldBeFace(const std::vector<MetadataFlag> &flags) {
    // returns true if face is set or if no topology set
    for (auto &f : flags) {
      if (f == Metadata::Face)
        return true;
      else if (f == Metadata::Cell)
        return false;
      else if (f == Metadata::Edge)
        return false;
      else if (f == Metadata::Node)
        return false;
    }
    return true;
  }
};

} // namespace parthenon

#endif // INTERFACE_CONTAINER_ITERATOR_HPP_
