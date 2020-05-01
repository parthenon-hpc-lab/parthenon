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
#ifndef INTERFACE_VARIABLE_PACK_HPP_
#define INTERFACE_VARIABLE_PACK_HPP_

#include <array>
#include <forward_list>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "interface/container.hpp"
#include "interface/metadata.hpp"
#include "interface/variable.hpp"

namespace parthenon {

// some convenience aliases meant for internal use only
namespace vpack_types {
template <typename T>
using VarList = std::forward_list<std::shared_ptr<CellVariable<T>>>;
using IndexPair = std::pair<int, int>;
} // namespace vpack_types

using PackIndexMap = std::map<std::string, vpack_types::IndexPair>;
template <typename T>
using ViewOfParArrays = Kokkos::View<ParArray3D<T> *>;

// Try to keep these Variable*Pack classes as lightweight as possible.
// They go to the device.
template <typename T>
class VariablePack {
 public:
  VariablePack(const ViewOfParArrays<T> view, const std::array<int, 4> dims)
      : v_(view), dims_(dims) {}
  KOKKOS_FORCEINLINE_FUNCTION
  auto &operator()(const int n) const { return v_(n); }
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
  const ViewOfParArrays<T> v_;
  const std::array<int, 4> dims_;
};

template <typename T>
class VariableFluxPack : public VariablePack<T> {
 public:
  VariableFluxPack(const ViewOfParArrays<T> view, const ViewOfParArrays<T> f0,
                   const ViewOfParArrays<T> f1, const ViewOfParArrays<T> f2,
                   const std::array<int, 4> dims, const int nflux)
      : VariablePack<T>(view, dims), f_({f0, f1, f2}), nflux_(nflux) {}

  KOKKOS_FORCEINLINE_FUNCTION
  ViewOfParArrays<T> &flux(const int dir) const { return f_[dir]; }

  KOKKOS_FORCEINLINE_FUNCTION
  T &flux(const int dir, const int n, const int k, const int j, const int i) const {
    return f_[dir](n)(k, j, i);
  }

 private:
  const std::array<const ViewOfParArrays<T>, 3> f_;
  const int nflux_;
};

//
// define some helper functions
//
template <typename T>
VariableFluxPack<T> MakeFluxPack(vpack_types::VarList<T> &vars,
                                 vpack_types::VarList<T> &flux_vars,
                                 PackIndexMap *vmap = nullptr) {
  using vpack_types::IndexPair;
  // count up the size
  int vsize = 0;
  for (const auto &v : vars) {
    vsize += v->GetDim(6) * v->GetDim(5) * v->GetDim(4);
  }
  int fsize = 0;
  for (const auto &v : flux_vars) {
    fsize += v->GetDim(6) * v->GetDim(5) * v->GetDim(4);
  }

  auto fvar = vars.front()->data;
  auto slice = fvar.Get(0, 0, 0);
  // make the outer view
  ViewOfParArrays<T> cv("MakeFluxPack::cv", vsize);
  ViewOfParArrays<T> f0("MakeFluxPack::f0", fsize);
  ViewOfParArrays<T> f1("MakeFluxPack::f1", fsize);
  ViewOfParArrays<T> f2("MakeFluxPack::f2", fsize);
  auto host_view = Kokkos::create_mirror_view(cv);
  auto host_f0 = Kokkos::create_mirror_view(f0);
  auto host_f1 = Kokkos::create_mirror_view(f1);
  auto host_f2 = Kokkos::create_mirror_view(f2);
  // add variables to host view
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
    if (vmap != nullptr) {
      vmap->insert(
          std::pair<std::string, IndexPair>(v->label(), IndexPair(vstart, vindex - 1)));
    }
  }
  // add fluxes to host view
  vindex = 0;
  for (const auto &v : flux_vars) {
    int vstart = vindex;
    for (int k = 0; k < v->GetDim(6); k++) {
      for (int j = 0; j < v->GetDim(5); j++) {
        for (int i = 0; i < v->GetDim(4); i++) {
          host_f0(vindex) = v->flux[0].Get(k, j, i);
          host_f1(vindex) = v->flux[1].Get(k, j, i);
          host_f2(vindex++) = v->flux[2].Get(k, j, i);
        }
      }
    }
    if (vmap != nullptr) {
      vmap->insert(
          std::pair<std::string, IndexPair>(v->label(), IndexPair(vstart, vindex - 1)));
    }
  }
  Kokkos::deep_copy(cv, host_view);
  Kokkos::deep_copy(f0, host_f0);
  Kokkos::deep_copy(f1, host_f1);
  Kokkos::deep_copy(f2, host_f2);
  std::array<int, 4> cv_size = {fvar.GetDim(1), fvar.GetDim(2), fvar.GetDim(3), vsize};
  return VariableFluxPack<T>(cv, f0, f1, f2, cv_size, fsize);
}

template <typename T>
VariablePack<T> MakePack(vpack_types::VarList<T> &vars, PackIndexMap *vmap = nullptr) {
  using vpack_types::IndexPair;
  // count up the size
  int vsize = 0;
  for (const auto &v : vars) {
    vsize += v->GetDim(6) * v->GetDim(5) * v->GetDim(4);
  }

  auto fvar = vars.front()->data;
  auto slice = fvar.Get(0, 0, 0);
  // make the outer view
  ViewOfParArrays<T> cv("MakePack::cv", vsize);
  auto host_view = Kokkos::create_mirror_view(cv);
  int vindex = 0;
  int sparse_start;
  std::string sparse_name = "";
  for (const auto v : vars) {
    if (v->IsSet(Metadata::Sparse)) {
      if (sparse_name == "") {
        sparse_name = v->label();
        sparse_name.erase(sparse_name.find_last_of("_"));
        sparse_start = vindex;
      }
    } else if (!(sparse_name == "")) {
      vmap->insert(std::pair<std::string, IndexPair>(
          sparse_name, IndexPair(sparse_start, vindex - 1)));
      sparse_name = "";
    }
    int vstart = vindex;
    for (int k = 0; k < v->GetDim(6); k++) {
      for (int j = 0; j < v->GetDim(5); j++) {
        for (int i = 0; i < v->GetDim(4); i++) {
          host_view(vindex++) = v->data.Get(k, j, i);
        }
      }
    }
    if (vmap != nullptr) {
      vmap->insert(
          std::pair<std::string, IndexPair>(v->label(), IndexPair(vstart, vindex - 1)));
    }
  }
  if (!(sparse_name == "")) {
    vmap->insert(std::pair<std::string, IndexPair>(sparse_name,
                                                   IndexPair(sparse_start, vindex - 1)));
  }

  Kokkos::deep_copy(cv, host_view);
  std::array<int, 4> cv_size = {fvar.GetDim(1), fvar.GetDim(2), fvar.GetDim(3), vsize};
  return VariablePack<T>(cv, cv_size);
}

template <typename T>
vpack_types::VarList<T> MakeList(const Container<T> &c,
                                 const std::vector<std::string> &names,
                                 const std::vector<int> sparse_ids = {}) {
  vpack_types::VarList<T> vars;
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
      if (sparse_ids.size() > 0) { // grab specific sparse variabes
        auto smap = sv->second->GetMap();
        for (auto its = sparse_ids.rbegin(); its != sparse_ids.rend(); ++its) {
          vars.push_front(smap[*its]);
        }
      } else {
        auto svec = sv->second->GetVector();
        for (auto its = svec.rbegin(); its != svec.rend(); ++its) {
          vars.push_front(*its);
        }
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
vpack_types::VarList<T> MakeList(const Container<T> &c,
                                 const std::vector<MetadataFlag> &flags) {
  vpack_types::VarList<T> vars;
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

//
// factory-like functions to generate Variable*Packs
//

// pull out variables and fluxes by name
template <typename T>
VariableFluxPack<T> PackVariablesAndFluxes(const Container<T> &c,
                                           const std::vector<std::string> &var_names,
                                           const std::vector<std::string> &flx_names) {
  vpack_types::VarList<T> vars = MakeList(c, var_names);
  vpack_types::VarList<T> fvars = MakeList(c, flx_names);
  return MakeFluxPack<T>(vars, fvars);
}

// pull out variables and fluxes by name and fill in an index map
template <typename T>
VariableFluxPack<T>
PackVariablesAndFluxes(const Container<T> &c, const std::vector<std::string> &var_names,
                       const std::vector<std::string> &flx_names, PackIndexMap &vmap) {
  vpack_types::VarList<T> vars = MakeList(c, var_names);
  vpack_types::VarList<T> fvars = MakeList(c, flx_names);
  return MakeFluxPack<T>(vars, fvars);
}

// pull out variables and fluxes based on Metadata flags
template <typename T>
VariableFluxPack<T> PackVariablesAndFluxes(const Container<T> &c,
                                           const std::vector<MetadataFlag> &flags) {
  vpack_types::VarList<T> vars = MakeList(c, flags);
  return MakeFluxPack<T>(vars, vars);
}

// pull out variables by name, including a particular set of sparse ids
template <typename T>
VariablePack<T> PackVariables(const Container<T> &c,
                              const std::vector<std::string> &names,
                              const std::vector<int> &sparse_ids) {
  vpack_types::VarList<T> vars = MakeList(c, names, sparse_ids);
  return MakePack<T>(vars);
}

// pull out variables by name, including a particular set of sparse ids, and fill in an
// index map
template <typename T>
VariablePack<T> PackVariables(const Container<T> &c,
                              const std::vector<std::string> &names,
                              const std::vector<int> &sparse_ids, PackIndexMap &vmap) {
  vpack_types::VarList<T> vars = MakeList(c, names, sparse_ids);
  return MakePack<T>(vars, &vmap);
}

// pull out variables by name and fill in an index map
template <typename T>
VariablePack<T> PackVariables(const Container<T> &c,
                              const std::vector<std::string> &names, PackIndexMap &vmap) {
  vpack_types::VarList<T> vars = MakeList(c, names);
  return MakePack<T>(vars, &vmap);
}

// pull out variables by name
template <typename T>
VariablePack<T> PackVariables(const Container<T> &c,
                              const std::vector<std::string> &names) {
  vpack_types::VarList<T> vars = MakeList(c, names);
  return MakePack<T>(vars);
}

// pull out variables based on Metadata
template <typename T>
VariablePack<T> PackVariables(const Container<T> &c,
                              const std::vector<MetadataFlag> &flags) {
  vpack_types::VarList<T> vars = MakeList(c, flags);
  return MakePack<T>(vars);
}

// pull out all variables and fill in an index map
template <typename T>
VariablePack<T> PackVariables(const Container<T> &c, PackIndexMap &vmap) {
  vpack_types::VarList<T> vars;
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

// pull out all variables
template <typename T>
VariablePack<T> PackVariables(const Container<T> &c) {
  vpack_types::VarList<T> vars;
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

} // namespace parthenon

#endif // INTERFACE_VARIABLE_PACK_HPP_
