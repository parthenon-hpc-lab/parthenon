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
#ifndef INTERFACE_SPARSE_VARIABLE_HPP_
#define INTERFACE_SPARSE_VARIABLE_HPP_

#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "globals.hpp"
#include "interface/variable.hpp"

namespace parthenon {

template <typename T>
using SparseMap = std::map<int, std::shared_ptr<CellVariable<T>>>;

///
/// SparseVariable builds on top of  the CellVariable class to include a map
template <typename T>
class SparseVariable {
 public:
  SparseVariable() = default;
  // Copies src variable but only including chosen sparse ids.
  SparseVariable(SparseVariable<T> &src, const std::vector<int> &sparse_ids)
      : dims_(src.dims_), label_(src.label_), metadata_(src.metadata_) {
    for (int id : sparse_ids) {
      auto var = src.varMap_[id];
      Add(id, var);
    }
  }
  SparseVariable(std::shared_ptr<SparseVariable<T>> src,
                 const std::vector<int> &sparse_ids)
      : dims_(src->dims_), label_(src->label_), metadata_(src->metadata_) {
    for (int id : sparse_ids) {
      auto var = src->varMap_[id];
      Add(id, var);
    }
  }
  SparseVariable(const std::string &label, const Metadata &m, std::array<int, 6> &dims)
      : dims_(dims), label_(label), metadata_(m) {}

  std::shared_ptr<SparseVariable<T>> AllocateCopy() {
    auto sv = std::make_shared<SparseVariable<T>>(label_, metadata_, dims_);
    for (auto &v : varMap_) {
      sv->Add(v.first, v.second->AllocateCopy());
    }
    return sv;
  }

  /// create a new variable alias from variable 'theLabel' in input variable mv
  // void AddAlias(const std::string& theLabel, SparseVariable<T>& mv);

  /// create a new variable deep copy from variable 'theLabel' in input variable mv
  // void AddCopy(const std::string& theLabel, SparseVariable<T>& mv);

  /// create a new variable
  void Add(int sparse_index, std::array<int, 6> &dims);

  // accessors
  inline CellVariable<T> &operator()(const int m) { return *(varMap_[m]); }
  inline T &operator()(const int m, const int i) { return (*(varMap_[m]))(i); }
  inline T &operator()(const int m, const int j, const int i) {
    return (*(varMap_[m]))(j, i);
  }
  inline T &operator()(const int m, const int k, const int j, const int i) {
    return (*(varMap_[m]))(k, j, i);
  }
  inline T &operator()(const int m, const int n, const int k, const int j, const int i) {
    return (*(varMap_[m]))(n, k, j, i);
  }
  inline T &operator()(const int m, const int l, const int n, const int k, const int j,
                       const int i) {
    return (*(varMap_[m]))(l, n, k, j, i);
  }
  inline T &operator()(const int m, const int p, const int l, const int n, const int k,
                       const int j, const int i) {
    return (*(varMap_[m]))(p, l, n, k, j, i);
  }

  bool IsSet(const MetadataFlag flag) { return metadata_.IsSet(flag); }

  /// return information string
  std::string info() {
    std::string s = "info not yet implemented for sparse variables";
    return s;
  }

  std::shared_ptr<CellVariable<T>> &Get(const int index) {
    auto it = varMap_.find(index);
    if (it == varMap_.end()) {
      throw std::invalid_argument("index " + std::to_string(index) +
                                  "does not exist in SparseVariable");
    }
    return it->second;
  }

  int GetIndex(int id) {
    auto it = std::find(indexMap_.begin(), indexMap_.end(), id);
    if (it == indexMap_.end()) return -1; // indicate the id doesn't exist
    return std::distance(indexMap_.begin(), it);
  }

  std::vector<int> &GetIndexMap() { return indexMap_; }

  CellVariableVector<T> &GetVector() { return varArray_; }

  SparseMap<T> &GetMap() { return varMap_; }

  // might want to implement this at some point
  // void DeleteVariable(const int var_id);

  std::string &label() { return label_; }
  int size() { return indexMap_.size(); }

  void print() { std::cout << "hello from sparse variables print" << std::endl; }

  const Metadata &metadata() { return metadata_; }
  const std::string &getAssociated() { return metadata_.getAssociated(); }

 private:
  std::array<int, 6> dims_;
  std::string label_;
  Metadata metadata_;
  SparseMap<T> varMap_;
  CellVariableVector<T> varArray_;
  std::vector<int> indexMap_;

  void Add(int varIndex, std::shared_ptr<CellVariable<T>> cv) {
    varArray_.push_back(cv);
    indexMap_.push_back(varIndex);
    varMap_[varIndex] = cv;
  }
};

template <typename T>
using SparseVector = std::vector<std::shared_ptr<SparseVariable<T>>>;
template <typename T>
using MapToSparse = std::map<std::string, std::shared_ptr<SparseVariable<T>>>;

} // namespace parthenon

#endif // INTERFACE_SPARSE_VARIABLE_HPP_
