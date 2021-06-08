//========================================================================================
// (C) (or copyright) 2020-2021. Triad National Security, LLC. All rights reserved.
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

#include <algorithm>
#include <array>
#include <forward_list>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "defs.hpp"
#include "interface/metadata.hpp"
#include "interface/variable.hpp"
#include "kokkos_abstraction.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

// Forward declarations
template <typename T>
class CellVariable;
template <typename T>
class ParticleVariable;

// some convenience aliases
namespace vpack_types {
template <typename T>
using SwarmVarList = std::forward_list<std::shared_ptr<ParticleVariable<T>>>;

// Sparse and/or scalar variables are multiple indices in the outer view of a pack
// the pairs represent interval (inclusive) of those indices
using IndexPair = std::pair<int, int>;

// Flux packs require a set of names for the variables and a set of names for the strings
// and order matters. So StringPair forms the keys for the FluxPack cache.
using StringPair = std::pair<std::vector<std::string>, std::vector<std::string>>;

} // namespace vpack_types

// helper class to make lists of variables with labels
template <typename T>
class VarListWithLabels {
 public:
  VarListWithLabels<T>() = default;

  // Adds a variable to the list if one of the following is true:
  // a) The variable is not sparse
  // b) The set of sparse_ids is empty
  // c) The sparse id of the variable is contained in the set of sparse_ids
  void Add(const std::shared_ptr<CellVariable<T>> &var,
           const std::unordered_set<int> &sparse_ids = {}) {
    if (!var->IsSparse() || sparse_ids.empty() ||
        (sparse_ids.count(var->GetSparseID()) > 0)) {
      vars_.push_back(var);
      // The label here is used in the key for variable pack caching, so it has to include
      // the allocation status of the variable, since we cannot reuse a variable pack if
      // one of its variables has a changed allocation status
      labels_.push_back(var->label() + (var->IsAllocated() ? "_A" : ""));
    }
  }

  const auto &vars() const { return vars_; }
  const auto &labels() const { return labels_; }

 private:
  CellVariableVector<T> vars_;
  std::vector<std::string> labels_;
};

// using PackIndexMap = std::unordered_map<std::string, vpack_types::IndexPair>;
class PackIndexMap {
 public:
  PackIndexMap() = default;

  const auto &get(const std::string &base_name, int sparse_id = InvalidSparseID) const {
    const auto &key = MakeVarLabel(base_name, sparse_id);
    auto itr = map_.find(key);
    if (itr == map_.end()) {
      auto err = "PackIndexMap does not have key '" + key + "'";
      PARTHENON_THROW(err.c_str());
    }

    return itr->second;
  }

  auto &get(const std::string &base_name, int sparse_id = InvalidSparseID) {
    // to avoid code duplication, call const version and cast away const
    const auto &res = static_cast<const PackIndexMap&>(*this).get(base_name, sparse_id);
    return const_cast<vpack_types::IndexPair&>(res);
  }

  [[deprecated("Use PackIndexMap::get() instead")]] vpack_types::IndexPair &
  operator[](const std::string &key) {
    // this is too dangerous, we won't notice that we don't have a requested field if
    // misspelled or sparse id not allocated

    // if (!Has(key)) {
    //   map_.insert({key, vpack_types::IndexPair(0, -1)});
    // }
    // return map_.at(key);

    return get(key);
  }

  void insert(std::pair<std::string, vpack_types::IndexPair> keyval) {
    map_.insert(keyval);
  }

  bool Has(std::string const &base_name, int sparse_id = InvalidSparseID) const {
    return map_.count(MakeVarLabel(base_name, sparse_id)) > 0;
  }

  // for debugging
  void print() const {
    for (const auto itr : map_) {
      printf("%s: %i - %i\n", itr.first.c_str(), itr.second.first, itr.second.second);
    }
  }

 private:
  std::unordered_map<std::string, vpack_types::IndexPair> map_;
};

template <typename T>
using ViewOfParArrays = ParArray1D<ParArray3D<T>>;

// Try to keep these Variable*Pack classes as lightweight as possible.
// They go to the device.
template <typename T>
class VariablePack {
 public:
  VariablePack() = default;

  VariablePack(const ViewOfParArrays<T> &view, const ParArray1D<int> &sparse_ids,
               const ParArray1D<int> &vector_component, const std::array<int, 4> &dims)
      : v_(view), sparse_ids_(sparse_ids), vector_component_(vector_component),
        dims_(dims), ndim_((dims[2] > 1 ? 3 : (dims[1] > 1 ? 2 : 1))) {}

  KOKKOS_FORCEINLINE_FUNCTION
  bool IsAllocated(const int n) const {
    assert(0 <= n && n < dims_[3]);
    return v_(n).size() > 0;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  ParArray3D<T> &operator()(const int n) const {
    assert(IsAllocated(n));
    return v_(n);
  }

  KOKKOS_FORCEINLINE_FUNCTION
  T &operator()(const int n, const int k, const int j, const int i) const {
    assert(IsAllocated(n));
    return v_(n)(k, j, i);
  }

  // TODO(JL) This is here so code templated on VariablePack and MeshBlockPack doesn't
  // need to change, but maybe we should add assert(m == 0)?
  KOKKOS_FORCEINLINE_FUNCTION
  T &operator()(const int m, const int n, const int k, const int j, const int i) const {
    assert(IsAllocated(n));
    return v_(n)(k, j, i);
  }

  KOKKOS_FORCEINLINE_FUNCTION
  int GetDim(const int i) const {
    assert(i > 0 && i < 6);
    return (i == 5 ? 1 : dims_[i - 1]);
  }

  KOKKOS_FORCEINLINE_FUNCTION
  int GetSparseID(const int n) const {
    assert(0 <= n && n < dims_[3]);
    return sparse_ids_(n);
  }

  KOKKOS_FORCEINLINE_FUNCTION
  bool IsSparse(const int n) const { return GetSparseID() != InvalidSparseID; }

  KOKKOS_FORCEINLINE_FUNCTION
  int VectorComponent(const int n) const {
    assert(0 <= n && n < dims_[3]);
    return vector_component_(n);
  }

  KOKKOS_FORCEINLINE_FUNCTION
  bool IsVector(const int n) const { return VectorComponent(n) != NODIR; }

  KOKKOS_FORCEINLINE_FUNCTION
  int GetNdim() const { return ndim_; }

 protected:
  ViewOfParArrays<T> v_;
  ParArray1D<int> sparse_ids_;
  ParArray1D<int> vector_component_;
  std::array<int, 4> dims_;
  int ndim_;
};
template <typename T>
class SwarmVariablePack {
 public:
  SwarmVariablePack() = default;
  SwarmVariablePack(const ViewOfParArrays<T> view, const std::array<int, 2> dims)
      : v_(view), dims_(dims) {}
  KOKKOS_FORCEINLINE_FUNCTION
  ParArray3D<T> &operator()(const int n) const { return v_(n); }
  KOKKOS_FORCEINLINE_FUNCTION
  T &operator()(const int n, const int i) const { return v_(n)(0, 0, i); }

 private:
  ViewOfParArrays<T> v_;
  std::array<int, 2> dims_;
};

template <typename T>
class VariableFluxPack : public VariablePack<T> {
 public:
  VariableFluxPack() = default;
  VariableFluxPack(const ViewOfParArrays<T> &view, const ViewOfParArrays<T> &f0,
                   const ViewOfParArrays<T> &f1, const ViewOfParArrays<T> &f2,
                   const ParArray1D<int> &sparse_ids,
                   const ParArray1D<int> &vector_component,
                   const std::array<int, 4> &dims, int fsize)
      : VariablePack<T>(view, sparse_ids, vector_component, dims), f_({f0, f1, f2}),
        fsize_(fsize) {}

  KOKKOS_FORCEINLINE_FUNCTION
  const ViewOfParArrays<T> &flux(const int dir) const {
    assert(dir > 0 && dir <= this->ndim_);
    return f_[dir - 1];
  }

  KOKKOS_FORCEINLINE_FUNCTION
  bool IsFluxAllocated(const int n) const {
    assert(0 <= n && n < fsize_);
    // we can just check X1DIR, because it always exists and it's allocated iff all
    // used dirs are allcoated
    return flux(X1DIR)(n).size() > 0;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  T &flux(const int dir, const int n, const int k, const int j, const int i) const {
    assert(IsFluxAllocated(n));
    return flux(dir)(n)(k, j, i);
  }

 private:
  std::array<ViewOfParArrays<T>, 3> f_;
  int fsize_;
};

template <typename PackType>
struct PackAndIndexMap {
  PackType pack;
  PackIndexMap map;
};

template <typename T>
struct VarMetaPack {
  VariablePack<T> pack;
  PackIndexMap map;
  const std::vector<std::string> *key;
};

template <typename T>
struct FluxMetaPack {
  VariableFluxPack<T> pack;
  PackIndexMap map;
  const vpack_types::StringPair *key;
};

// Using std::map, not std::unordered_map because the key
// would require a custom hashing function. Note this is slower: O(log(N))
// instead of O(1).

template <typename T>
using SwarmPackIndxPair = PackAndIndexMap<SwarmVariablePack<T>>;

template <typename T>
using MapToSwarmVariablePack = std::map<std::vector<std::string>, SwarmPackIndxPair<T>>;

template <typename T>
using MapToVariablePack = std::map<std::vector<std::string>, VarMetaPack<T>>;

template <typename T>
using MapToVariableFluxPack = std::map<vpack_types::StringPair, FluxMetaPack<T>>;

template <typename T>
void FillVarView(const CellVariableVector<T> &vars, bool coarse,
                 ViewOfParArrays<T> &cv_out, ParArray1D<int> &sparse_id_out,
                 ParArray1D<int> &vector_component_out, PackIndexMap *vmap_out) {
  using vpack_types::IndexPair;

  assert(cv_out.size() == sparse_id_out.size());
  assert(cv_out.size() == vector_component_out.size());

  auto host_cv = Kokkos::create_mirror_view(Kokkos::HostSpace(), cv_out);
  auto host_sp = Kokkos::create_mirror_view(Kokkos::HostSpace(), sparse_id_out);
  auto host_vc = Kokkos::create_mirror_view(Kokkos::HostSpace(), vector_component_out);

  int vindex = 0;
  for (const auto &v : vars) {
    int vstart = vindex;
    for (int k = 0; k < v->GetDim(6); k++) {
      for (int j = 0; j < v->GetDim(5); j++) {
        for (int i = 0; i < v->GetDim(4); i++) {
          host_sp(vindex) = v->GetSparseID();

          // returns 1 for X1DIR, 2 for X2DIR, 3 for X3DIR
          // for tensors, returns flattened index.
          // for scalar-objects, returns NODIR.
          const bool is_vec = v->IsSet(Metadata::Vector) || v->IsSet(Metadata::Tensor);
          host_vc(vindex) = is_vec ? vindex - vstart + 1 : NODIR;

          if (v->IsAllocated()) {
            host_cv(vindex) = coarse ? v->coarse_s.Get(k, j, i) : v->data.Get(k, j, i);
          } else {
            host_cv(vindex) = ParArray3D<T>("unallocated_" + v->label(), 0, 0, 0);
          }

          vindex++;
        }
      }
    }

    if (vmap_out != nullptr) {
      vmap_out->insert(
          std::pair<std::string, IndexPair>(v->label(), IndexPair(vstart, vindex - 1)));
    }
  }

  Kokkos::deep_copy(cv_out, host_cv);
  Kokkos::deep_copy(sparse_id_out, host_sp);
  Kokkos::deep_copy(vector_component_out, host_vc);
}

template <typename T>
void FillSwarmVarView(const vpack_types::SwarmVarList<T> &vars, PackIndexMap *vmap,
                      ViewOfParArrays<T> &cv) {
  using vpack_types::IndexPair;

  auto host_view = Kokkos::create_mirror_view(Kokkos::HostSpace(), cv);

  int vindex = 0;
  int sparse_start;
  std::string sparse_name;
  // TODO(BRR) Remove the logic for sparse variables
  for (const auto v : vars) {
    if (vmap != nullptr) {
      vmap->insert(std::pair<std::string, IndexPair>(
          sparse_name, IndexPair(sparse_start, vindex - 1)));
      sparse_name = "";
    }
    int vstart = vindex;
    // Reusing ViewOfParArrays which expects 3D slices
    host_view(vindex++) = v->data.Get(0, 0, 0);
    if (vmap != nullptr) {
      vmap->insert(
          std::pair<std::string, IndexPair>(v->label(), IndexPair(vstart, vindex - 1)));
    }
  }

  Kokkos::deep_copy(cv, host_view);
}

template <typename T>
void FillFluxViews(const CellVariableVector<T> &vars, const int ndim,
                   ViewOfParArrays<T> &f1_out, ViewOfParArrays<T> &f2_out,
                   ViewOfParArrays<T> &f3_out, PackIndexMap *vmap_out) {
  using vpack_types::IndexPair;

  auto host_f1 = Kokkos::create_mirror_view(Kokkos::HostSpace(), f1_out);
  auto host_f2 = Kokkos::create_mirror_view(Kokkos::HostSpace(), f2_out);
  auto host_f3 = Kokkos::create_mirror_view(Kokkos::HostSpace(), f3_out);

  int vindex = 0;
  for (const auto &v : vars) {
    int vstart = vindex;
    for (int k = 0; k < v->GetDim(6); k++) {
      for (int j = 0; j < v->GetDim(5); j++) {
        for (int i = 0; i < v->GetDim(4); i++) {
          if (v->IsAllocated()) {
            host_f1(vindex) = v->flux[X1DIR].Get(k, j, i);
            if (ndim >= 2) host_f2(vindex) = v->flux[X2DIR].Get(k, j, i);
            if (ndim >= 3) host_f3(vindex) = v->flux[X3DIR].Get(k, j, i);
          } else {
            host_f1(vindex) = ParArray3D<T>("unallocated_f1_" + v->label(), 0, 0, 0);
            if (ndim >= 2) {
              host_f2(vindex) = ParArray3D<T>("unallocated_f2_" + v->label(), 0, 0, 0);
            }
            if (ndim >= 3) {
              host_f3(vindex) = ParArray3D<T>("unallocated_f3_" + v->label(), 0, 0, 0);
            }
          }

          vindex++;
        }
      }
    }

    if (vmap_out != nullptr) {
      vmap_out->insert(
          std::pair<std::string, IndexPair>(v->label(), IndexPair(vstart, vindex - 1)));
    }
  }

  Kokkos::deep_copy(f1_out, host_f1);
  Kokkos::deep_copy(f2_out, host_f2);
  Kokkos::deep_copy(f3_out, host_f3);
}

template <typename T>
VariableFluxPack<T> MakeFluxPack(const CellVariableVector<T> &vars,
                                 const CellVariableVector<T> &flux_vars,
                                 PackIndexMap *vmap_out) {
  // count up the size
  int vsize = 0;
  for (const auto &v : vars) {
    // we also count unallocated vars because the total size needs to be uniform across
    // meshblocks that meshblock packs will work
    vsize += v->NumComponents();
  }
  int fsize = 0;
  for (const auto &v : flux_vars) {
    // we also count unallocated vars because the total size needs to be uniform across
    // meshblocks that meshblock packs will work
    fsize += v->NumComponents();
  }

  // make the outer view
  ViewOfParArrays<T> cv("MakeFluxPack::cv", vsize);
  ViewOfParArrays<T> f1("MakeFluxPack::f1", fsize);
  ViewOfParArrays<T> f2("MakeFluxPack::f2", fsize);
  ViewOfParArrays<T> f3("MakeFluxPack::f3", fsize);
  ParArray1D<int> sparse_id("MakeFluxPack::sparse_id", vsize);
  ParArray1D<int> vector_component("MakeFluxPack::vector_component", vsize);

  std::array<int, 4> cv_size{0, 0, 0, 0};
  if (vsize > 0) {
    // add variables
    auto fvar = vars.front()->data;
    cv_size = {fvar.GetDim(1), fvar.GetDim(2), fvar.GetDim(3), vsize};
    FillVarView(vars, false, cv, sparse_id, vector_component, vmap_out);

    if (fsize > 0) {
      // add fluxes
      const int ndim = (cv_size[2] > 1 ? 3 : (cv_size[1] > 1 ? 2 : 1));
      FillFluxViews(flux_vars, ndim, f1, f2, f3, vmap_out);
    }
  }

  return VariableFluxPack<T>(cv, f1, f2, f3, sparse_id, vector_component, cv_size, fsize);
}

template <typename T>
VariablePack<T> MakePack(const CellVariableVector<T> &vars, bool coarse,
                         PackIndexMap *vmap_out) {
  // count up the size
  int vsize = 0;
  for (const auto &v : vars) {
    // we also count unallocated vars because the total size needs to be uniform across
    // meshblocks that meshblock packs will work
    vsize += v->NumComponents();
  }

  // make the outer view
  ViewOfParArrays<T> cv("MakePack::cv", vsize);
  ParArray1D<int> sparse_id("MakePack::sparse_id", vsize);
  ParArray1D<int> vector_component("MakePack::vector_component", vsize);

  std::array<int, 4> cv_size{0, 0, 0, 0};
  if (vsize > 0) {
    const auto &fvar = coarse ? vars.front()->coarse_s : vars.front()->data;
    cv_size = {fvar.GetDim(1), fvar.GetDim(2), fvar.GetDim(3), vsize};
    FillVarView(vars, coarse, cv, sparse_id, vector_component, vmap_out);
  }

  return VariablePack<T>(cv, sparse_id, vector_component, cv_size);
}

template <typename T>
SwarmVariablePack<T> MakeSwarmPack(const vpack_types::SwarmVarList<T> &vars,
                                   PackIndexMap *vmap = nullptr) {
  // count up the size
  int vsize = 0;
  for (const auto &v : vars) {
    vsize++;
  }

  // make the outer view
  ViewOfParArrays<T> cv("MakeSwarmPack::cv", vsize);
  ParArray1D<int> sparse_assoc("MakeSwarmPack::sparse_assoc", vsize); // Unused

  FillSwarmVarView(vars, vmap, cv);

  // If no vars, return empty pack
  if (vars.empty()) {
    return SwarmVariablePack<T>();
  }

  auto fvar = vars.front()->data;
  std::array<int, 2> cv_size = {fvar.GetDim(1), vsize};
  return SwarmVariablePack<T>(cv, cv_size);
}

} // namespace parthenon

#endif // INTERFACE_VARIABLE_PACK_HPP_
