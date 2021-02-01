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
#ifndef INTERFACE_MESHBLOCK_DATA_ITERATOR_HPP_
#define INTERFACE_MESHBLOCK_DATA_ITERATOR_HPP_

/// Provides an iterator that iterates over the variables in a
/// container.  Eventually this will get transitioned to an iterator
/// type in the Container itself, but for now we have to do it this
/// way because Sriram doesn't know enough C++ to do this correctly.

#include <memory>
#include <string>
#include <vector>

#include "interface/meshblock_data.hpp"
#include "interface/properties_interface.hpp"
#include "interface/variable.hpp"

namespace parthenon {

template <typename T>
class MeshBlockDataIterator {
 public:
  /// the subset of variables that match this iterator
  CellVariableVector<T> vars;
  // std::vector<FaceVariable> varsFace; // face vars that match
  // std::vector<EdgeVariable> varsEdge; // edge vars that match

  void MakeList(const std::shared_ptr<MeshBlockData<T>> &c,
                const std::vector<std::string> &names) {
    auto var_map = c->GetCellVariableMap();
    auto sparse_map = c->GetSparseMap();
    // reverse iterator to end up with a list in the same order as requested
    for (const auto &name : names) {
      bool found = false;
      auto v = var_map.find(name);
      if (v != var_map.end()) {
        vars.push_back(v->second);
        found = true;
      }
      auto sv = sparse_map.find(name);
      if (sv != sparse_map.end()) {
        if (found) {
          // that's weird, found the name in both???
          std::cerr << name << " found in both var_map and sparse_map in PackVariables"
                    << std::endl;
          std::exit(1);
        }
        found = true;
        for (const auto &svar : sv->second->GetVector()) {
          vars.push_back(svar);
        }
      }
      /*if (!found) {
        std::cerr << name << " not found in var_map or sparse_map in PackVariables"
                  << std::endl;
        std::exit(1);
      }*/
    }
    return;
  }

  /// initializes the iterator with a container and a flag to match
  /// @param c the container on which you want the iterator
  /// @param flags: a vector of Metadata::flags that you want to match
  MeshBlockDataIterator<T>(const std::shared_ptr<MeshBlockData<T>> &c,
                           const std::vector<MetadataFlag> &flags, bool matchAny = false)
      : allVars_(c->GetCellVariableVector()) {
    for (auto &svar : c->GetSparseVector()) {
      CellVariableVector<T> &svec = svar->GetVector();
      allVars_.insert(allVars_.end(), svec.begin(), svec.end());
    }
    // faces not active yet    allFaceVars_ = c.faceVars();
    // edges not active yet    allEdgeVars_ = c.edgeVars();
    resetVars(flags, matchAny); // fill subset based on mask vector
  }

  /// initializes the iterator with a container and a flag to match
  /// @param c the container on which you want the iterator
  /// @param names: a vector of std::string with names you want to match
  MeshBlockDataIterator<T>(const std::shared_ptr<MeshBlockData<T>> &c,
                           const std::vector<std::string> &names) {
    MakeList(c, names);
    /*allVars_ = c.GetCellVariableVector();
    for (auto &svar : c.GetSparseVector()) {
      CellVariableVector<T> &svec = svar->GetVector();
      allVars_.insert(allVars_.end(), svec.begin(), svec.end());
    }
    // faces not active yet    allFaceVars_ = c.faceVars();
    // edges not active yet    allEdgeVars_ = c.edgeVars();
    resetVars(names); // fill subset based on mask vector*/
  }

  /// Changes the mask for the iterator and resets the iterator
  /// @param names: a vector of MetadataFlag that you want to match
  void resetVars(const std::vector<std::string> &names) {
    // 1: clear out variables stored so far
    emptyVars_();

    // 2: fill in the subset of variables that match at least one entry in names
    for (auto pv : allVars_) {
      if (std::find(names.begin(), names.end(), pv->label()) != names.end())
        vars.push_back(pv);
    }
  }

  /// Changes the mask for the iterator and resets the iterator
  /// @param flags: a vector of MetadataFlag that you want to match
  void resetVars(const std::vector<MetadataFlag> &flags, bool matchAny = false) {
    // 1: clear out variables stored so far
    emptyVars_();

    // 2: fill in the subset of variables that match mask
    for (auto pv : allVars_) {
      if ((matchAny && pv->metadata().AnyFlagsSet(flags)) ||
          ((!matchAny) && pv->metadata().AllFlagsSet(flags))) {
        vars.push_back(pv);
      }
    }
  }

 private:
  uint64_t mask_;
  CellVariableVector<T> allVars_;
  // FaceVector<T> allFaceVars_ = {};
  // EdgeVector<T> allEdgeVars_ = {};
  void emptyVars_() {
    vars.clear();
    //  varsFace.clear();
    //  varsEdge.clear();
  }
};

} // namespace parthenon

#endif // INTERFACE_MESHBLOCK_DATA_ITERATOR_HPP_
