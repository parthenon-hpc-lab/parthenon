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
///
/// A Sparse Variable type for Placebo-K.
/// Builds on AthenaArrays
/// Date: Sep 12, 2019
///
#ifndef INTERFACE_SPARSEVARIABLE_HPP_
#define INTERFACE_SPARSEVARIABLE_HPP_

#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "globals.hpp"
#include "Variable.hpp"

namespace parthenon {

template <typename T>
class SparseMap : public std::map<int, std::shared_ptr<Variable<T>>> {
 public:
  Variable<T>& operator()(int m) {
    return *(*this)[m];
  }
  T& operator()(int m, int i) {
    return (*(*this)[m])(i);
  }
  T& operator()(int m, int j, int i) {
    return (*(*this)[m])(j,i);
  }
  T& operator()(int m, int k, int j, int i) {
    return (*(*this)[m])(k,j,i);
  }
  T& operator()(int m, int l, int k, int j, int i) {
    return (*(*this)[m])(l,k,j,i);
  }
  T& operator()(int m, int n, int l, int k, int j, int i) {
    return (*(*this)[m])(n,l,k,j,i);
  }
  T& operator()(int m, int g, int n, int l, int k, int j, int i) {
    return (*(*this)[m])(g,n,l,k,j,i);
  }
  Metadata& metadata() const { return this->begin().second->metadata();}
};

///
/// SparseVariable builds on top of  the Variable class to include a map
template <typename T>
class SparseVariable {
 public:
  // Note only default constructor

  /// create a new variable alias from variable 'theLabel' in input variable mv
  void AddAlias(const std::string& theLabel, SparseVariable<T>& mv);

  /// create a new variable deep copy from variable 'theLabel' in input variable mv
  void AddCopy(const std::string& theLabel, SparseVariable<T>& mv);

  ///create a new variable
  void Add(MeshBlock &pmb,
           const std::string &label,
           const Metadata &metadata,
           const std::vector<int>& inDims={});


  /// return information string
  std::string info(const std::string &label) {
    char tmp[100] = "";

    if (_cellVars.find(label) == _cellVars.end()) {
      return (label + std::string("not found"));
    }

    auto myMap = _cellVars[label];

    std::string s = label;
    s.resize(20,'.');

    s += std::string(" variables:");
    for (auto const& items : myMap) s += std::to_string(items.first) + ":";

    // now append flag
    auto pVar = myMap.begin();
    s += " : " + std::to_string(pVar->second->metadata().mask());
    s += " : " + pVar->second->metadata().maskAsString();

    return s;
  }

  SparseMap<T>& Get(const std::string& label) {
    if (_cellVars.find(label) == _cellVars.end()) {
      throw std::invalid_argument ("Unable to find variable " +
                                   label +
                                   " in SparseMap<T> container::Get() ");
    }
    return _cellVars[label];
  }

  VariableVector<T>& GetVector(const std::string& label) {
    if (_pcellVars.find(label) == _pcellVars.end()) return _empty;
    /*{
      std::cerr << "Looking for " << label << " in GetVector" << std::endl;
      for (auto & v : _pcellVars) {
        std::cerr << v.first << std::endl;
      }
      for (auto & v : _cellVars) {
        std::cerr << v.first << std::endl;
      }
      throw std::invalid_argument ("Unable to find variable in container");
    }*/
    return _pcellVars[label];
  }

  std::vector<int>& GetIndexMap(const std::string& label) {
    if (_indexMap.find(label) == _indexMap.end()) {
      throw std::invalid_argument ("Unable to find variable in container");
    }
    return _indexMap[label];
  }

  std::map<std::string,SparseMap<T>> CellVars() { return _cellVars;}

  void DeleteVariable(const int var_id);
  void DeleteVariable(const int var_id, const std::string label);

  Variable<T>& Get(const std::string& label, int sparse_id) {
    auto myMap = this->Get(label);
    if (myMap.find(sparse_id) == myMap.end()) {
      throw std::invalid_argument ("Unable to find specific variable in container");
    }
    return *myMap[sparse_id];
  }

  std::map<std::string,SparseMap<T>>& getAllCellVars() {
    return _cellVars;
  }

  std::map<std::string,std::vector<int>> getIndexMap() { return _indexMap; }

  std::map<std::string,VariableVector<T>> getCellVarVectors() { return _pcellVars; }

  void print() {
    for ( auto &m : _cellVars) {
      std::cout << "    sparsevar:cell:" << m.second.begin()->second->info() << ":";
      for (auto &v : m.second) {
        std::cout << v.first << ":";
      }
      std::cout << std::endl;
    }
    for ( auto &m : _pcellVars) {
      std::cout << "    varvec: " << m.first
                << " has " << m.second.size() << " elements"
                << std::endl;
    }
  }

 private:
  std::map<std::string,SparseMap<T>> _cellVars;
  std::map<std::string,VariableVector<T>> _pcellVars;
  std::map<std::string,std::vector<int>> _indexMap;
  VariableVector<T> _empty;
};

} // namespace parthenon

#endif //INTERFACE_SPARSEVARIABLE_HPP_
