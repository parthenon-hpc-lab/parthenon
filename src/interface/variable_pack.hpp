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

#include <Kokkos_Core.hpp>

#include "interface/metadata.hpp"

namespace parthenon {

// Forward declarations
template <typename T>
class CellVariable;

// some convenience aliases
namespace vpack_types {
template <typename T>
using VarList = std::forward_list<std::shared_ptr<CellVariable<T>>>;
// Sparse and/or scalar variables are multiple indices in the outer view of a pack
// the pairs represent interval (inclusive) of thos indices
using IndexPair = std::pair<int, int>;
// Flux packs require a set of names for the variables and a set of names for the strings
// and order matters. So StringPair forms the keys for the FluxPack cache.
using StringPair = std::pair<std::vector<std::string>, std::vector<std::string>>;
} // namespace vpack_types

// using PackIndexMap = std::map<std::string, vpack_types::IndexPair>;
class PackIndexMap {
 public:
  PackIndexMap() = default;
  vpack_types::IndexPair &operator[](const std::string &key) {
    if (!map_.count(key)) {
      map_[key] = vpack_types::IndexPair(0, -1);
    }
    return map_[key];
  }
  void insert(std::pair<std::string, vpack_types::IndexPair> keyval) {
    map_[keyval.first] = keyval.second;
  }

 private:
  std::map<std::string, vpack_types::IndexPair> map_;
};
template <typename T>
using ViewOfParArrays = ParArray1D<ParArray3D<T>>;

// Try to keep these Variable*Pack classes as lightweight as possible.
// They go to the device.
template <typename T>
class VariablePack {
 public:
  VariablePack() = default;
  VariablePack(const ViewOfParArrays<T> view,
               const ParArray1D<int> sparse_ids,
               const ParArray1D<bool> is_vector,
               const std::array<int, 4> dims)
    : v_(view), sparse_ids_(sparse_ids), si_vector_(is_vector), dims_(dims),
      ndim_((dims[2] > 1 ? 3 : (dims[1] > 1 ? 2 : 1))) {}
  KOKKOS_FORCEINLINE_FUNCTION
  ParArray3D<T> &operator()(const int n) const { return v_(n); }
  KOKKOS_FORCEINLINE_FUNCTION
  T &operator()(const int n, const int k, const int j, const int i) const {
    return v_(n)(k, j, i);
  }
  KOKKOS_FORCEINLINE_FUNCTION
  int GetDim(const int i) const {
    assert(i > 0 && i < 5);
    return dims_[i - 1];
  }
  KOKKOS_FORCEINLINE_FUNCTION
  int GetSparse(const int n) const {
    assert(0 <= n && n < dims_[3]);
    return sparse_ids_(n);
  }
  KOKKOS_FORCEINLINE_FUNCTION
  bool IsVector(const int n) const {
    assert(0 <= n && n < dims_[3]);
    return is_vector_(n);
  }
  KOKKOS_FORCEINLINE_FUNCTION
  int GetNdim() const { return ndim_; }

 protected:
  ViewOfParArrays<T> v_;
  ParArray1D<int> sparse_ids_;
  ParArray1D<bool> is_vector_;
  std::array<int, 4> dims_;
  int ndim_;
};

template <typename T>
class VariableFluxPack : public VariablePack<T> {
 public:
  VariableFluxPack() = default;
  VariableFluxPack(const ViewOfParArrays<T> view, const ViewOfParArrays<T> f0,
                   const ViewOfParArrays<T> f1, const ViewOfParArrays<T> f2,
                   const ParArray1D<int> sparse_ids,
                   const ParArray1D<int> is_vector,
                   const std::array<int, 4> dims)
    : VariablePack<T>(view, sparse_ids, is_vector, dims), f_({f0, f1, f2}) {}

  KOKKOS_FORCEINLINE_FUNCTION
  ViewOfParArrays<T> &flux(const int dir) const {
    assert(dir > 0 && dir <= this->ndim_);
    return f_[dir - 1];
  }

  KOKKOS_FORCEINLINE_FUNCTION
  T &flux(const int dir, const int n, const int k, const int j, const int i) const {
    assert(dir > 0 && dir <= this->ndim_);
    return f_[dir - 1](n)(k, j, i);
  }

 private:
  std::array<ViewOfParArrays<T>, 3> f_;
};

// Using std::map, not std::unordered_map because the key
// would require a custom hashing function. Note this is slower: O(log(N))
// instead of O(1).
// Unfortunately, std::pair doesn't work. So I have to roll my own.
// It appears to be an interaction caused by a std::map<key,std::pair>
// Possibly it's a compiler bug. gcc/7.4.0
// ~JMM
template <typename PackType>
struct PackAndIndexMap {
  PackType pack;
  PackIndexMap map;
};
template <typename T>
using PackIndxPair = PackAndIndexMap<VariablePack<T>>;
template <typename T>
using FluxPackIndxPair = PackAndIndexMap<VariableFluxPack<T>>;
template <typename T>
using MapToVariablePack = std::map<std::vector<std::string>, PackIndxPair<T>>;
template <typename T>
using MapToVariableFluxPack = std::map<vpack_types::StringPair, FluxPackIndxPair<T>>;

template <typename T>
void FillVarView(const vpack_types::VarList<T> &vars, PackIndexMap *vmap,
                 ViewOfParArrays<T> &cv, ParArray1D<int> &sparse_assoc,
                 ParArray1D<bool> &is_vector) {
  using vpack_types::IndexPair;

  auto host_view = Kokkos::create_mirror_view(Kokkos::HostSpace(), cv);
  auto host_sp = Kokkos::create_mirror_view(Kokkos::HostSpace(), sparse_assoc);
  auto host_iv = Kokkos::create_mirror_view(Kokkos::HostSPace, is_vector);

  int vindex = 0;
  int sparse_start;
  int sparse_id;
  std::string sparse_name;
  for (const auto v : vars) {
    sparse_id = v->metadata().GetSparseId();
    if (vmap != nullptr) {
      if (v->IsSet(Metadata::Sparse)) {
        std::string sparse_trim = v->label();
        sparse_trim.erase(sparse_trim.find_last_of("_"));
        if (sparse_name == "") {
          sparse_name = sparse_trim;
          sparse_start = vindex;
        }
        if (sparse_name != sparse_trim) {
          vmap->insert(std::pair<std::string, IndexPair>(
              sparse_name, IndexPair(sparse_start, vindex - 1)));
          sparse_name = sparse_trim;
          sparse_start = vindex;
        }
      } else if (!(sparse_name == "")) {
        vmap->insert(std::pair<std::string, IndexPair>(
            sparse_name, IndexPair(sparse_start, vindex - 1)));
        sparse_name = "";
      }
    }
    int vstart = vindex;
    for (int k = 0; k < v->GetDim(6); k++) {
      for (int j = 0; j < v->GetDim(5); j++) {
        for (int i = 0; i < v->GetDim(4); i++) {
          host_sp(vindex) = sparse_id;
          host_iv(vindex) = v->IsSet(Metadata::Vector);
          host_view(vindex) = v->data.Get(k, j, i);
          vindex++;
        }
      }
    }
    if (vmap != nullptr) {
      vmap->insert(
          std::pair<std::string, IndexPair>(v->label(), IndexPair(vstart, vindex - 1)));
    }
  }
  if (vmap != nullptr && sparse_name != "") {
    vmap->insert(std::pair<std::string, IndexPair>(sparse_name,
                                                   IndexPair(sparse_start, vindex - 1)));
  }

  Kokkos::deep_copy(cv, host_view);
  Kokkos::deep_copy(sparse_assoc, host_sp);
  Kokkos::deep_copy(is_vector, host_iv);
}

template <typename T>
void FillFluxViews(const vpack_types::VarList<T> &vars, PackIndexMap *vmap,
                   const int ndim, ViewOfParArrays<T> &f1, ViewOfParArrays<T> &f2,
                   ViewOfParArrays<T> &f3) {
  using vpack_types::IndexPair;

  auto host_f1 = Kokkos::create_mirror_view(Kokkos::HostSpace(), f1);
  auto host_f2 = Kokkos::create_mirror_view(Kokkos::HostSpace(), f2);
  auto host_f3 = Kokkos::create_mirror_view(Kokkos::HostSpace(), f3);

  int vindex = 0;
  int sparse_start;
  std::string sparse_name;
  for (const auto &v : vars) {
    if (vmap != nullptr) {
      if (v->IsSet(Metadata::Sparse)) {
        std::string sparse_trim = v->label();
        sparse_trim.erase(sparse_trim.find_last_of("_"));
        if (sparse_name == "") {
          sparse_name = sparse_trim;
          sparse_start = vindex;
        }
        if (sparse_name != sparse_trim) {
          vmap->insert(std::pair<std::string, IndexPair>(
              sparse_name, IndexPair(sparse_start, vindex - 1)));
          sparse_name = sparse_trim;
          sparse_start = vindex;
        }
      } else if (!(sparse_name == "")) {
        vmap->insert(std::pair<std::string, IndexPair>(
            sparse_name, IndexPair(sparse_start, vindex - 1)));
        sparse_name = "";
      }
    }
    int vstart = vindex;
    for (int k = 0; k < v->GetDim(6); k++) {
      for (int j = 0; j < v->GetDim(5); j++) {
        for (int i = 0; i < v->GetDim(4); i++) {
          host_f1(vindex) = v->flux[X1DIR].Get(k, j, i);
          if (ndim >= 2) host_f2(vindex) = v->flux[X2DIR].Get(k, j, i);
          if (ndim >= 3) host_f3(vindex) = v->flux[X3DIR].Get(k, j, i);
          vindex++;
        }
      }
    }
    if (vmap != nullptr) {
      vmap->insert(
          std::pair<std::string, IndexPair>(v->label(), IndexPair(vstart, vindex - 1)));
    }
  }
  if (vmap != nullptr && sparse_name != "") {
    vmap->insert(std::pair<std::string, IndexPair>(sparse_name,
                                                   IndexPair(sparse_start, vindex - 1)));
  }

  Kokkos::deep_copy(f1, host_f1);
  Kokkos::deep_copy(f2, host_f2);
  Kokkos::deep_copy(f3, host_f3);
}

template <typename T>
VariableFluxPack<T> MakeFluxPack(const vpack_types::VarList<T> &vars,
                                 const vpack_types::VarList<T> &flux_vars,
                                 PackIndexMap *vmap = nullptr) {
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
  std::array<int, 4> cv_size = {fvar.GetDim(1), fvar.GetDim(2), fvar.GetDim(3), vsize};
  const int ndim = (cv_size[2] > 1 ? 3 : (cv_size[1] > 1 ? 2 : 1));

  // make the outer view
  ViewOfParArrays<T> cv("MakeFluxPack::cv", vsize);
  ViewOfParArrays<T> f1("MakeFluxPack::f1", fsize);
  ViewOfParArrays<T> f2("MakeFluxPack::f2", fsize);
  ViewOfParArrays<T> f3("MakeFluxPack::f3", fsize);
  ParArray1D<int> sparse_assoc("MakeFluxPack::sparse_assoc", vsize);
  ParArray1D<bool> is_vector("MakeFluxPack::is_vector", vsize);
  // add variables to host view
  FillVarView(vars, vmap, cv, sparse_assoc, is_vector);
  // add fluxes to host view
  FillFluxViews(flux_vars, vmap, ndim, f1, f2, f3);

  return VariableFluxPack<T>(cv, f1, f2, f3, sparse_assoc, is_vector, cv_size);
}

template <typename T>
VariablePack<T> MakePack(const vpack_types::VarList<T> &vars,
                         PackIndexMap *vmap = nullptr) {
  // count up the size
  int vsize = 0;
  for (const auto &v : vars) {
    vsize += v->GetDim(6) * v->GetDim(5) * v->GetDim(4);
  }

  // make the outer view
  ViewOfParArrays<T> cv("MakePack::cv", vsize);
  ParArray1D<int> sparse_assoc("MakePack::sparse_assoc", vsize);
  ParArray1D<int> is_vector("MakePack::is_vector", vsize);

  FillVarView(vars, vmap, cv, sparse_assoc, is_vector);

  auto fvar = vars.front()->data;
  std::array<int, 4> cv_size = {fvar.GetDim(1), fvar.GetDim(2), fvar.GetDim(3), vsize};
  return VariablePack<T>(cv, sparse_assoc, is_vector, cv_size);
}

} // namespace parthenon

#endif // INTERFACE_VARIABLE_PACK_HPP_
