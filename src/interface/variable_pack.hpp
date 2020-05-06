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
//=======================================================================================
#ifndef INTERFACE_VARIABLE_PACK_HPP_
#define INTERFACE_VARIABLE_PACK_HPP_

#include <array>
#include <forward_list>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "interface/metadata.hpp"
#include "interface/variable.hpp"

namespace parthenon {

// some convenience aliases
namespace vpack_types {
template <typename T>
using VarList = std::forward_list<std::shared_ptr<CellVariable<T>>>;
using IndexPair = std::pair<int, int>;
using StringPair = std::pair<std::vector<std::string>,
                               std::vector<std::string>>;
} // namespace vpack_types

using PackIndexMap = std::map<std::string, vpack_types::IndexPair>;
template <typename T>
using ViewOfParArrays = ParArrayND<ParArray3D<T>>;

// Try to keep these Variable*Pack classes as lightweight as possible.
// They go to the device.
template <typename T>
class VariablePack {
 public:
  VariablePack(const ViewOfParArrays<T> view, const std::array<int, 4> dims)
      : v_(view), dims_(dims) {}
  KOKKOS_FORCEINLINE_FUNCTION
  ParArray3D<T> &operator()(const int n) const { return v_(n); }
  KOKKOS_FORCEINLINE_FUNCTION
  T &operator()(const int n, const int k, const int j, const int i) const {
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

// mapping to a tuple instead of using multiple maps reduces # of lookups
// this wouldn't be super important if lookup time was constant,
// but std::maps are trees, not hash tables and have an O(log(N)) lookup.
template <typename T>
using PackIndxPair<T> = std::pair<VariablePack<T>,PackIndexMap>;
template <typename T>
using FluxPackIndxPair<T> = std::pair<VariableFluxPack<T>,PackIndexMap>;
template <typename T>
using MapToVariablePack<T> = std::map<std::vector<std::string>,
                                      PackIndxPair<T>>;
template <typename T>
using MapToVariableFluxPack<T> = std::map<vpack_types::StringPair,
                                          FluxPackIndxPair>

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

  // make the outer view
  ViewOfParArrays<T> cv("MakeFluxPack::cv", vsize);
  ViewOfParArrays<T> f0("MakeFluxPack::f0", fsize);
  ViewOfParArrays<T> f1("MakeFluxPack::f1", fsize);
  ViewOfParArrays<T> f2("MakeFluxPack::f2", fsize);
  auto host_view = cv.GetHostMirror();
  auto host_f0 = f0.GetHostMirror();
  auto host_f1 = f1.GetHostMirror();
  auto host_f2 = f2.GetHostMirror();
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
  cv.DeepCopy(host_view);
  f0.DeepCopy(host_f0);
  f1.DeepCopy(host_f1);
  f2.DeepCopy(host_f2);
  auto fvar = vars.front()->data;
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

  // make the outer view
  ViewOfParArrays<T> cv("MakePack::cv", vsize);
  auto host_view = cv.GetHostMirror();
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

  cv.DeepCopy(host_view);
  auto fvar = vars.front()->data;
  std::array<int, 4> cv_size = {fvar.GetDim(1), fvar.GetDim(2), fvar.GetDim(3), vsize};
  return VariablePack<T>(cv, cv_size);
}

} // namespace parthenon

#endif // INTERFACE_VARIABLE_PACK_HPP_
