//========================================================================================
// (C) (or copyright) 2020-2023. Triad National Security, LLC. All rights reserved.
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
#ifndef INTERFACE_SPARSE_PACK_BASE_HPP_
#define INTERFACE_SPARSE_PACK_BASE_HPP_

#include <algorithm>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "coordinates/coordinates.hpp"
#include "interface/state_descriptor.hpp"
#include "interface/variable.hpp"
#include "interface/variable_state.hpp"
#include "utils/utils.hpp"

namespace parthenon {
class SparsePackCache;
namespace impl {
class PackDescriptor;
}

// Map for going from variable names to sparse pack variable indices
using SparsePackIdxMap = std::unordered_map<std::string, std::size_t>;

class StateDescriptor;

enum class PDOpt { WithFluxes, Coarse, Flatten };

class SparsePackBase {
 public:
  SparsePackBase() = default;
  virtual ~SparsePackBase() = default;

 protected:
  friend class SparsePackCache;

  using alloc_t = std::vector<int>;
  using pack_t = ParArray3D<ParArray3D<Real, VariableState>>;
  using bounds_t = ParArray3D<int>;
  using bounds_h_t = typename ParArray3D<int>::HostMirror;
  using coords_t = ParArray1D<ParArray0D<Coordinates_t>>;

  // Returns a SparsePackBase object that is either newly created or taken
  // from the cache in pmd. The cache itself handles the all of this logic
  template <class T>
  static SparsePackBase GetPack(T *pmd, const impl::PackDescriptor &desc) {
    auto &cache = pmd->GetSparsePackCache();
    return cache.Get(pmd, desc);
  }

  // Return a map from variable names to pack variable indices
  static SparsePackIdxMap GetIdxMap(const impl::PackDescriptor &desc);

  // Get a list of booleans of the allocation status of every variable in pmd matching the
  // PackDescriptor desc
  template <class T>
  static alloc_t GetAllocStatus(T *pmd, const impl::PackDescriptor &desc);

  // Actually build a `SparsePackBase` (i.e. create a view of views, fill on host, and
  // deep copy the view of views to device) from the variables specified in desc contained
  // from the blocks contained in pmd (which can either be MeshBlockData/MeshData).
  template <class T>
  static SparsePackBase Build(T *pmd, const impl::PackDescriptor &desc);

  pack_t pack_;
  bounds_t bounds_;
  bounds_h_t bounds_h_;
  coords_t coords_;

  bool with_fluxes_;
  bool coarse_;
  bool flat_;
  int nblocks_;
  int nvar_;
  int size_;
};

// Object for cacheing sparse packs in MeshData and MeshBlockData objects. This
// handles checking for a pre-existing pack and creating a new SparsePackBase if
// a cached pack is unavailable. Essentially, this operates as a map from
// `PackDescriptor` to `SparsePackBase`
class SparsePackCache {
 public:
  std::size_t size() const { return pack_map.size(); }

  void clear() { pack_map.clear(); }

 protected:
  template <class T>
  SparsePackBase &Get(T *pmd, const impl::PackDescriptor &desc);

  template <class T>
  SparsePackBase &BuildAndAdd(T *pmd, const impl::PackDescriptor &desc,
                              const std::string &ident);

  std::string GetIdentifier(const impl::PackDescriptor &desc) const;
  std::unordered_map<std::string, std::pair<SparsePackBase, SparsePackBase::alloc_t>>
      pack_map;

  friend class SparsePackBase;
};

namespace impl {
struct PackDescriptor {
  using VariableGroup_t = std::vector<std::pair<VarID, Uid_t>>;
  using SelectorFunction_t = std::function<bool(int, const VarID &, const Metadata &)>;
  using SelectorFunctionUid_t = std::function<bool(int, const Uid_t &, const Metadata &)>;

  void Print() const;

  // default constructor needed for certain use cases
  PackDescriptor()
      : nvar_groups(0), var_group_names({}), var_groups({}), with_fluxes(false),
        coarse(false), flat(false) {}

  template <class GROUP_t, class SELECTOR_t>
  PackDescriptor(StateDescriptor *psd, const std::vector<GROUP_t> &var_groups_in,
                 const SELECTOR_t &selector, const std::set<PDOpt> &options)
      : nvar_groups(var_groups_in.size()), var_group_names(MakeGroupNames(var_groups_in)),
        var_groups(BuildUids(var_groups_in.size(), psd, selector)),
        with_fluxes(options.count(PDOpt::WithFluxes)),
        coarse(options.count(PDOpt::Coarse)), flat(options.count(PDOpt::Flatten)) {
    PARTHENON_REQUIRE(!(with_fluxes && coarse),
                      "Probably shouldn't be making a coarse pack with fine fluxes.");
  }

  const int nvar_groups;
  const std::vector<std::string> var_group_names;
  const std::vector<VariableGroup_t> var_groups;
  const bool with_fluxes;
  const bool coarse;
  const bool flat;

 private:
  template <class FUNC_t>
  std::vector<PackDescriptor::VariableGroup_t>
  BuildUids(int nvgs, const StateDescriptor *const psd, const FUNC_t &selector) {
    auto fields = psd->AllFields();
    std::vector<VariableGroup_t> vgs(nvgs);
    for (auto [id, md] : fields) {
      for (int i = 0; i < nvgs; ++i) {
        auto uid = Variable<Real>::GetUniqueID(id.label());
        if constexpr (std::is_invocable<FUNC_t, int, VarID, Metadata>::value) {
          if (selector(i, id, md)) {
            vgs[i].push_back({id, uid});
          }
        } else if constexpr (std::is_invocable<FUNC_t, int, Uid_t, Metadata>::value) {
          if (selector(i, uid, md)) {
            vgs[i].push_back({id, uid});
          }
        } else {
          PARTHENON_FAIL("Passing the wrong sort of selector.");
        }
      }
    }
    // Ensure ordering in terms of value of sparse indices
    for (auto &vg : vgs) {
      std::sort(vg.begin(), vg.end(), [](const auto &a, const auto &b) {
        if (a.first.base_name == b.first.base_name)
          return a.first.sparse_id < b.first.sparse_id;
        return a.first.base_name < b.first.base_name;
      });
    }
    return vgs;
  }

  template <class base_t>
  std::vector<std::string> MakeGroupNames(const std::vector<base_t> &var_groups) {
    if constexpr (std::is_same<base_t, std::string>::value) {
      return var_groups;
    } else if constexpr (std::is_same<base_t, Uid_t>::value) {
      std::vector<std::string> var_group_names;
      for (auto &vg : var_groups)
        var_group_names.push_back(std::to_string(vg));
      return var_group_names;
    }
    // silence compiler warnings about no return statement
    return std::vector<std::string>();
  }
};
} // namespace impl

} // namespace parthenon

#endif // INTERFACE_SPARSE_PACK_BASE_HPP_
