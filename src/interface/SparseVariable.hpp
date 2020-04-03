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
/// A Sparse Variable type.
/// Builds on ParArrayNDs
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
struct SparseMap : public std::map<int, std::shared_ptr<CellVariable<T>>> {
  CellVariable<T>& operator()(int m) {
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
};

///
/// SparseVariable builds on top of  the CellVariable class to include a map
template <typename T>
class SparseVariable {
 public:
  SparseVariable() = default;
  SparseVariable(const std::string& label, const Metadata& metadata, std::array<int,6>& dims) 
    : _dims(dims), _label(label), _metadata(metadata) {}

  SparseVariable(SparseVariable& src)
    : _dims(src._dims), _label(src._label), _metadata(src._metadata) {
    for (auto & v : src._varMap) {
      auto var = std::make_shared<CellVariable<T>>(*v.second);
      _varMap[v.first] = var;
      _varArray.push_back(var);
      _indexMap.push_back(v.first);
    }
  }

  /// create a new variable alias from variable 'theLabel' in input variable mv
  //void AddAlias(const std::string& theLabel, SparseVariable<T>& mv);

  /// create a new variable deep copy from variable 'theLabel' in input variable mv
  //void AddCopy(const std::string& theLabel, SparseVariable<T>& mv);

  ///create a new variable
  void Add(int sparse_index);

  bool isSet(const MetadataFlag flag) { return _metadata.IsSet(flag); }

  /// return information string
  std::string info() {
    std::string s = "info not yet implemented for sparse variables";
    return s;
  }

  CellVariable<T>& Get(const int index) {
    if (_varMap.find(index) == _varMap.end()) {
      throw std::invalid_argument("index " + std::to_string(index) + 
                                  "does not exist in SparseVariable");
    }
    return *(_varMap[index]);
  }

  int GetIndex(int id) {
    auto it = std::find(_indexMap.begin(), _indexMap.end(), id);
    if (it == _indexMap.end()) return -1; // indicate the id doesn't exist
    return std::distance(_indexMap.begin(), it);
  }

  std::vector<int>& GetIndexMap() { return _indexMap; }

  CellVariableVector<T>& GetVector() { return _varArray; }

  SparseMap<T>& GetMap() { return _varMap; }

  // might want to implement this at some point
  //void DeleteVariable(const int var_id);

  std::string& label() { return _label; }

  void print() {
    std::cout << "hello from sparse variables print" << std::endl;
  }

 private:
  std::array<int,6> _dims;
  std::string _label;
  Metadata _metadata;
  SparseMap<T> _varMap;
  CellVariableVector<T> _varArray;
  std::vector<int> _indexMap;
  CellVariableVector<T> _empty;
};

template <typename T>
using SparseVector = std::vector<std::shared_ptr<SparseVariable<T>>>;
template <typename T>
using MapToSparse = std::map<std::string, std::shared_ptr<SparseVariable<T>>>;


} // namespace parthenon

#endif //INTERFACE_SPARSEVARIABLE_HPP_
