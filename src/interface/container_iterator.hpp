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
#include <map>
#include <memory>
#include <string>
#include <utility>
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

 protected:
  const T v_;
  const std::array<int, 4> dims_;
};

template <typename T>
class VariableFluxPack : public VariablePack<T> {
 public:
  VariableFluxPack(T view, T f0, T f1, T f2, std::array<int, 4> dims)
      : VariablePack<T>(view, dims), f_({f0, f1, f2}) {}

  KOKKOS_FORCEINLINE_FUNCTION
  auto &flux(const int dir) { return f_[dir]; }

  KOKKOS_FORCEINLINE_FUNCTION
  auto &flux(const int dir, const int n, const int k, const int j, const int i) {
    return f_[dir](n)(k, j, i);
  }

 private:
  const std::array<T, 3> f_;
};

template <typename T>
using VarList = std::forward_list<std::shared_ptr<CellVariable<T>>>;
using IndexPair = std::pair<int, int>;
using PackIndexMap = std::map<std::string, IndexPair>;

template <typename T>
auto MakeFluxPack(VarList<T> &vars, VarList<T> &flux_vars, PackIndexMap *vmap = nullptr) {
  // count up the size
  int vsize = 0;
  int nvars = 0;
  for (const auto &v : vars) {
    nvars++;
    vsize += v->GetDim(6) * v->GetDim(5) * v->GetDim(4);
  }

  auto fvar = vars.front()->data;
  auto slice = fvar.Get(0, 0, 0);
  // make the outer view
  auto cv = Kokkos::View<decltype(slice) *>("MakeFluxPack::cv", vsize);
  auto f0 = Kokkos::View<decltype(slice) *>("MakeFluxPack::f0", vsize);
  auto f1 = Kokkos::View<decltype(slice) *>("MakeFluxPack::f1", vsize);
  auto f2 = Kokkos::View<decltype(slice) *>("MakeFluxPack::f2", vsize);
  auto host_view = Kokkos::create_mirror_view(cv);
  auto host_f0 = Kokkos::create_mirror_view(f0);
  auto host_f1 = Kokkos::create_mirror_view(f1);
  auto host_f2 = Kokkos::create_mirror_view(f2);
  int vindex = 0;
  for (int n = 0; n < nvars; n++) {
    int vstart = vindex;
    auto v = vars[n];
    auto fv = flux_vars[n];
    for (int k = 0; k < v->GetDim(6); k++) {
      for (int j = 0; j < v->GetDim(5); j++) {
        for (int i = 0; i < v->GetDim(4); i++) {
          host_view(vindex) = v->data.Get(k, j, i);
          host_f0(vindex) = fv->flux[0].Get(k, j, i);
          host_f1(vindex) = fv->flux[1].Get(k, j, i);
          host_f2(vindex++) = fv->flux[2].Get(k, j, i);
        }
      }
    }
    if (vmap != nullptr) {
      vmap->insert(
          std::pair<std::string, IndexPair>(v->label(), IndexPair(vstart, vindex - 1)));
      vmap->insert(
          std::pair<std::string, IndexPair>(fv->label(), IndexPair(vstart, vindex - 1)));
    }
  }
  Kokkos::deep_copy(cv, host_view);
  Kokkos::deep_copy(f0, host_f0);
  Kokkos::deep_copy(f1, host_f1);
  Kokkos::deep_copy(f2, host_f2);
  std::array<int, 4> cv_size = {fvar.GetDim(1), fvar.GetDim(2), fvar.GetDim(3), vsize};
  return VariableFluxPack<decltype(cv)>(cv, f0, f1, f2, cv_size);
}

template <typename T>
auto MakePack(VarList<T> &vars, PackIndexMap *vmap = nullptr) {
  // count up the size
  int vsize = 0;
  for (const auto &v : vars) {
    vsize += v->GetDim(6) * v->GetDim(5) * v->GetDim(4);
  }

  auto fvar = vars.front()->data;
  auto slice = fvar.Get(0, 0, 0);
  // make the outer view
  auto cv = Kokkos::View<decltype(slice) *>("MakePack::cv", vsize);
  auto host_view = Kokkos::create_mirror_view(cv);
  int vindex = 0;
  for (const auto &v : vars) {
    int vstart = vindex;
    for (int k = 0; k < v->GetDim(6); k++) {
      for (int j = 0; j < v->GetDim(5); j++) {
        for (int i = 0; i < v->GetDim(4); i++) {
          host_view(vindex++) = v->data.Get(k, j, i);
        }
      }
    }
    if (vmap != nullptr)
      vmap->insert(
          std::pair<std::string, IndexPair>(v->label(), IndexPair(vstart, vindex - 1)));
  }
  Kokkos::deep_copy(cv, host_view);
  std::array<int, 4> cv_size = {fvar.GetDim(1), fvar.GetDim(2), fvar.GetDim(3), vsize};
  return VariablePack<decltype(cv)>(cv, cv_size);
}

template <typename T>
VarList<T> MakeList(const Container<T> &c, const std::vector<std::string> &names) {
  VarList<T> vars;
  auto var_map = c.GetCellVariableMap();
  auto sparse_map = c.GetSparseMap();
  // reverse iterator to end up with a list in the same order as requested
  for (auto it = names.rbegin(); it != names.rend(); ++it) {
    bool found = false;
    auto v = var_map.find(*it);
    if (v != var_map.end()) {
      vars.push_front(v->second);
      found = true;
    }
    auto sv = sparse_map.find(*it);
    if (sv != sparse_map.end()) {
      if (found) {
        // that's weird, found the name in both???
        std::cerr << *it << " found in both var_map and sparse_map in PackVariables"
                  << std::endl;
        std::exit(1);
      }
      found = true;
      auto svec = sv->second->GetVector();
      for (auto its = svec.rbegin(); its != svec.rend(); ++its) {
        vars.push_front(*its);
      }
    }
    if (!found) {
      std::cerr << *it << " not found in var_map or sparse_map in PackVariables"
                << std::endl;
      std::exit(1);
    }
  }
  return vars;
}

template <typename T>
VarList<T> MakeList(const Container<T> &c, const std::vector<MetadataFlag> &flags) {
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
  return vars;
}

template <typename T>
auto PackVariablesAndFluxes(const Container<T> &c,
                            const std::vector<std::string> &var_names,
                            const std::vector<std::string> &flx_names) {
  assert(var_names.size() == flx_names.size());
  VarList<T> vars = MakeList(c, var_names);
  VarList<T> fvars = MakeList(c, flx_names);
  return MakeFluxPack<T>(vars, fvars);
}

template <typename T>
auto PackVariables(const Container<T> &c, const std::vector<std::string> &names) {
  VarList<T> vars = MakeList(c, names);
  return MakePack<T>(vars);
}

template <typename T>
auto PackVariablesAndFluxes(const Container<T> &c,
                            const std::vector<std::string> &var_names,
                            const std::vector<std::string> &flx_names,
                            PackIndexMap &vmap) {
  assert(var_names.size() == flx_names.size());
  VarList<T> vars = MakeList(c, var_names);
  VarList<T> fvars = MakeList(c, flx_names);
  return MakeFluxPack<T>(vars, fvars);
}

template <typename T>
auto PackVariables(const Container<T> &c, const std::vector<std::string> &names,
                   PackIndexMap &vmap) {
  VarList<T> vars = MakeList(c, names);
  return MakePack<T>(vars, &vmap);
}

template <typename T>
auto PackVariablesAndFluxes(const Container<T> &c,
                            const std::vector<MetadataFlag> &flags) {
  VarList<T> vars = MakeList(c, flags);
  return MakeFluxPack<T>(vars, vars);
}

template <typename T>
auto PackVariables(const Container<T> &c, const std::vector<MetadataFlag> &flags) {
  VarList<T> vars = MakeList(c, flags);
  return MakePack<T>(vars);
}

template <typename T>
auto PackVariables(const Container<T> &c, PackIndexMap &vmap) {
  VarList<T> vars;
  for (const auto &v : c.GetCellVariableVector()) {
    vars.push_front(v);
  }
  for (const auto &sv : c.GetSparseVector()) {
    for (const auto &v : sv->GetVector()) {
      vars.push_front(v);
    }
  }

  return MakePack<T>(vars, &vmap);
}

template <typename T>
auto PackVariables(const Container<T> &c) {
  VarList<T> vars;
  for (const auto &v : c.GetCellVariableVector()) {
    vars.push_front(v);
  }
  for (const auto &sv : c.GetSparseVector()) {
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
    // faces not active yet    allFaceVars_ = c.faceVars();
    // edges not active yet    allEdgeVars_ = c.edgeVars();
    setMask(allVars, flags); // fill subset based on mask vector
  }

  //~ContainerIterator<T>() {
  //  emptyVars_();
  //}
  /// Changes the mask for the iterator and resets the iterator
  /// @param flagArray: a vector of MetadataFlag that you want to match
  void setMask(const CellVariableVector<T> &allVars,
               const std::vector<MetadataFlag> &flags) {
    // 1: clear out variables stored so far
    emptyVars_();

    // 2: fill in the subset of variables that match mask
    for (auto pv : allVars) {
      if (pv->metadata().AnyFlagsSet(flags)) {
        vars.push_back(pv);
      }
    }
  }

 private:
  uint64_t mask_;
  FaceVector<T> allFaceVars_ = {};
  // EdgeVector<T> allEdgeVars_ = {};
  // CellVariableVector<T> allVars_ = {};
  void emptyVars_() {
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
