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

#include "interface/meshblock_data.hpp"

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <set>
#include <sstream>
#include <unordered_set>
#include <utility>
#include <vector>

#include "globals.hpp"
#include "interface/metadata.hpp"
#include "interface/state_descriptor.hpp"
#include "interface/variable.hpp"
#include "interface/variable_pack.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "utils/error_checking.hpp"
#include "utils/utils.hpp"

namespace parthenon {

template <typename T>
void MeshBlockData<T>::Initialize(
    const std::shared_ptr<StateDescriptor> resolved_packages,
    const std::shared_ptr<MeshBlock> pmb) {
  SetBlockPointer(pmb);
  resolved_packages_ = resolved_packages;

  // clear all variables, maps, and pack caches
  varVector_.clear();
  varMap_.clear();
  varUidMap_.clear();
  flagsToVars_.clear();
  varPackMap_.clear();
  coarseVarPackMap_.clear();
  varFluxPackMap_.clear();

  for (auto const &q : resolved_packages->AllFields()) {
    AddField(q.first.base_name, q.second, q.first.sparse_id);
  }
}

///
/// The internal routine for adding a new field.  This subroutine
/// is topology aware and will allocate accordingly.
///
/// @param label the name of the variable
/// @param metadata the metadata associated with the variable
/// @param sparse_id the sparse id of the variable
template <typename T>
void MeshBlockData<T>::AddField(const std::string &base_name, const Metadata &metadata,
                                int sparse_id) {
  auto pvar =
      std::make_shared<CellVariable<T>>(base_name, metadata, sparse_id, pmy_block);
  Add(pvar);

  if (!Globals::sparse_config.enabled || !pvar->IsSparse()) {
    pvar->Allocate(pmy_block);
  }
}

// TODO(JMM): To use the set logic in GetVariablesFromFlags
// that logic would need to be generalized. Not bothering for now.
template <typename T>
void MeshBlockData<T>::CopyFrom(const MeshBlockData<T> &src, bool shallow_copy,
                                const std::vector<std::string> &names,
                                const std::vector<MetadataFlag> &flags,
                                const std::vector<int> &sparse_ids) {
  SetBlockPointer(src);
  resolved_packages_ = src.resolved_packages_;
  std::unordered_set<int> sparse_ids_set(sparse_ids.begin(), sparse_ids.end());

  auto add_var = [=, &flags, &sparse_ids](auto var) {
    if (!flags.empty() && !var->metadata().AnyFlagsSet(flags)) {
      return;
    }

    if (!sparse_ids.empty() && var->IsSparse() &&
        (sparse_ids_set.count(var->GetSparseID()) == 0)) {
      return;
    }

    if (shallow_copy || var->IsSet(Metadata::OneCopy)) {
      Add(var);
    } else {
      Add(var->AllocateCopy(pmy_block));
    }
  };

  if (names.empty()) {
    for (auto v : src.GetCellVariableVector()) {
      add_var(v);
    }
  } else {
    auto var_map = src.GetCellVariableMap();

    for (const auto &name : names) {
      bool found = false;
      auto v = var_map.find(name);
      if (v != var_map.end()) {
        found = true;
        add_var(v->second);
      }

      if (!found && (resolved_packages_ != nullptr)) {
        // check if this is a sparse base name, if so we get its pool of sparse_ids,
        // otherwise we get an empty pool
        const auto &sparse_pool = resolved_packages_->GetSparsePool(name);

        // add all sparse ids of the pool
        for (const auto iter : sparse_pool.pool()) {
          // this variable must exist, if it doesn't something is very wrong
          const auto &v = varMap_.at(MakeVarLabel(name, iter.first));
          add_var(v);
          found = true;
        }
      }

      PARTHENON_REQUIRE_THROWS(found, "MeshBlockData::CopyFrom: Variable '" + name +
                                          "' not found");
    }
  }
}

// Constructor for getting sub-containers
// the variables returned are all shallow copies of the src container.
// Optionally extract only some of the sparse ids of src variable.
template <typename T>
MeshBlockData<T>::MeshBlockData(const MeshBlockData<T> &src,
                                const std::vector<std::string> &names,
                                const std::vector<int> &sparse_ids) {
  CopyFrom(src, true, names, {}, sparse_ids);
}

// TODO(JMM): To use the set logic in GetVariablesFromFlags
// that logic would need to be generalized. Not bothering for now.
template <typename T>
MeshBlockData<T>::MeshBlockData(const MeshBlockData<T> &src,
                                const std::vector<MetadataFlag> &flags,
                                const std::vector<int> &sparse_ids) {
  CopyFrom(src, true, {}, flags, sparse_ids);
}

// provides a container that has a single sparse slice
template <typename T>
std::shared_ptr<MeshBlockData<T>>
MeshBlockData<T>::SparseSlice(const std::vector<int> &sparse_ids) const {
  auto c = std::make_shared<MeshBlockData<T>>();
  c->CopyFrom(*this, true, {}, {}, sparse_ids);
  return c;
}

/// Queries related to variable packs
/// This is a helper function that queries the cache for the given pack.
/// The strings are the keys and the lists are the values.
/// Inputs:
/// variables = forward list of shared pointers of vars to pack
/// fluxes = forward list of shared pointers of fluxes to pack
/// Returns:
/// A FluxMetaPack<T> that contains the actual VariableFluxPack, the PackIndexMap, and the
/// keys
template <typename T>
const VariableFluxPack<T> &MeshBlockData<T>::PackListedVariablesAndFluxes(
    const VarList &var_list, const VarList &flux_list, PackIndexMap *map,
    vpack_types::UidPair *key) {
  auto keys = std::make_pair(var_list.unique_ids(), flux_list.unique_ids());

  auto itr = varFluxPackMap_.find(keys);
  bool make_new_pack = false;
  if (itr == varFluxPackMap_.end()) {
    // we don't have a cached pack, need to make a new one
    make_new_pack = true;
  } else {
    // we have a cached pack, check allocation status
    if ((var_list.alloc_status() != itr->second.alloc_status) ||
        (flux_list.alloc_status() != itr->second.flux_alloc_status)) {
      // allocation statuses differ, need to make a new pack and remove outdated one
      make_new_pack = true;
      varFluxPackMap_.erase(itr);
    }
  }

  if (make_new_pack) {
    FluxPackIndxPair<T> new_item;
    new_item.alloc_status = var_list.alloc_status();
    new_item.flux_alloc_status = flux_list.alloc_status();
    new_item.pack = MakeFluxPack(var_list, flux_list, &new_item.map);
    new_item.pack.coords = GetParentPointer()->coords_device;
    itr = varFluxPackMap_.insert({keys, new_item}).first;

    // need to grab pointers here
    itr->second.pack.alloc_status_ = &itr->second.alloc_status;
    itr->second.pack.flux_alloc_status_ = &itr->second.flux_alloc_status;
  }

  if (map != nullptr) {
    *map = itr->second.map;
  }
  if (key != nullptr) {
    *key = itr->first;
  }

  return itr->second.pack;
}

/// This is a helper function that queries the cache for the given pack.
/// The strings are the key and the lists are the values.
/// Inputs:
/// vars = forward list of shared pointers of vars to pack
/// coarse = whether to use coarse pack map or not
/// Returns:
/// A VarMetaPack<T> that contains the actual VariablePack, the PackIndexMap, and the key
template <typename T>
const VariablePack<T> &
MeshBlockData<T>::PackListedVariables(const VarList &var_list, bool coarse,
                                      PackIndexMap *map,
                                      vpack_types::VPackKey_t *key_out) {
  const auto &key = var_list.unique_ids();
  auto &packmap = coarse ? coarseVarPackMap_ : varPackMap_;

  auto itr = packmap.find(key);
  bool make_new_pack = false;
  if (itr == packmap.end()) {
    // we don't have a cached pack, need to make a new one
    make_new_pack = true;
  } else {
    // we have a cached pack, check allocation status
    if (var_list.alloc_status() != itr->second.alloc_status) {
      // allocation statuses differ, need to make a new pack and remove outdated one
      make_new_pack = true;
      packmap.erase(itr);
    }
  }

  if (make_new_pack) {
    PackIndxPair<T> new_item;
    new_item.alloc_status = var_list.alloc_status();
    new_item.pack = MakePack<T>(var_list, coarse, &new_item.map);
    new_item.pack.coords = GetParentPointer()->coords_device;

    itr = packmap.insert({key, new_item}).first;

    // need to grab pointers after map insertion
    itr->second.pack.alloc_status_ = &itr->second.alloc_status;
  }

  if (map != nullptr) {
    *map = itr->second.map;
  }
  if (key_out != nullptr) {
    *key_out = itr->first;
  }

  return itr->second.pack;
}

/***********************************/
/* PACK VARIABLES INTERFACE        */
/***********************************/

/// Variables and fluxes by Name
template <typename T>
const VariableFluxPack<T> &MeshBlockData<T>::PackVariablesAndFluxesImpl(
    const std::vector<std::string> &var_names, const std::vector<std::string> &flx_names,
    const std::vector<int> &sparse_ids, PackIndexMap *map, vpack_types::UidPair *key) {
  return PackListedVariablesAndFluxes(GetVariablesByName(var_names, sparse_ids),
                                      GetVariablesByName(flx_names, sparse_ids), map,
                                      key);
}

/// Variables and fluxes by Metadata Flags
template <typename T>
const VariableFluxPack<T> &MeshBlockData<T>::PackVariablesAndFluxesImpl(
    const Metadata::FlagCollection &flags, const std::vector<int> &sparse_ids,
    PackIndexMap *map, vpack_types::UidPair *key) {
  return PackListedVariablesAndFluxes(GetVariablesByFlag(flags, sparse_ids),
                                      GetVariablesByFlag(flags, sparse_ids), map, key);
}

/// All variables and fluxes by Metadata Flags
template <typename T>
const VariableFluxPack<T> &MeshBlockData<T>::PackVariablesAndFluxesImpl(
    const std::vector<int> &sparse_ids, PackIndexMap *map, vpack_types::UidPair *key) {
  return PackListedVariablesAndFluxes(GetAllVariables(sparse_ids),
                                      GetAllVariables(sparse_ids), map, key);
}

/// Variables by Name
template <typename T>
const VariablePack<T> &
MeshBlockData<T>::PackVariablesImpl(const std::vector<std::string> &names,
                                    const std::vector<int> &sparse_ids, bool coarse,
                                    PackIndexMap *map, vpack_types::VPackKey_t *key) {
  return PackListedVariables(GetVariablesByName(names, sparse_ids), coarse, map, key);
}

/// Variables by Metadata Flags
template <typename T>
const VariablePack<T> &
MeshBlockData<T>::PackVariablesImpl(const Metadata::FlagCollection &flags,
                                    const std::vector<int> &sparse_ids, bool coarse,
                                    PackIndexMap *map, vpack_types::VPackKey_t *key) {
  return PackListedVariables(GetVariablesByFlag(flags, sparse_ids), coarse, map, key);
}

/// All variables
template <typename T>
const VariablePack<T> &
MeshBlockData<T>::PackVariablesImpl(const std::vector<int> &sparse_ids, bool coarse,
                                    PackIndexMap *map, vpack_types::VPackKey_t *key) {
  return PackListedVariables(GetAllVariables(sparse_ids), coarse, map, key);
}

// Get variables with the given names. The given name could either be a full variable
// label or a sparse base name. Optionally only extract sparse fields with a sparse id in
// the given set of sparse ids
template <typename T>
typename MeshBlockData<T>::VarList
MeshBlockData<T>::GetVariablesByName(const std::vector<std::string> &names,
                                     const std::vector<int> &sparse_ids) {
  typename MeshBlockData<T>::VarList var_list;
  std::unordered_set<int> sparse_ids_set(sparse_ids.begin(), sparse_ids.end());

  for (const auto &name : names) {
    const auto itr = varMap_.find(name);
    if (itr != varMap_.end()) {
      const auto &v = itr->second;
      // this name exists, add it
      var_list.Add(v, sparse_ids_set);
    } else if ((resolved_packages_ != nullptr) &&
               (resolved_packages_->SparseBaseNamePresent(name))) {
      const auto &sparse_pool = resolved_packages_->GetSparsePool(name);

      // add all sparse ids of the pool
      for (const auto iter : sparse_pool.pool()) {
        // this variable must exist, if it doesn't something is very wrong
        const auto &v = varMap_.at(MakeVarLabel(name, iter.first));
        var_list.Add(v, sparse_ids_set);
      }
    }
  }

  return var_list;
}

// From a given container, extract all variables whose Metadata matchs the all of the
// given flags (if the list of flags is empty, extract all variables), optionally only
// extracting sparse fields with an index from the given list of sparse indices
//
// JMM: This algorithm uses the map from metadata flags to variables
// to accelerate performance.
//
// The cost of this loop scales as O(Nflags * Nvars/flag) In worst
// case, this is linear in number of variables. However, on average,
// the number of vars with a desired flag will be much smaller than
// all vars. So average performance is much better than linear.
template <typename T>
typename MeshBlockData<T>::VarList
MeshBlockData<T>::GetVariablesByFlag(const Metadata::FlagCollection &flags,
                                     const std::vector<int> &sparse_ids) {
  Kokkos::Profiling::pushRegion("GetVariablesByFlag");

  typename MeshBlockData<T>::VarList var_list;
  std::unordered_set<int> sparse_ids_set(sparse_ids.begin(), sparse_ids.end());

  // Note that only intersections and unions count for flags.Empty()
  if (flags.Empty()) { // Easy. Just do them all.
    for (const auto &p : varMap_) {
      var_list.Add(p.second, sparse_ids_set);
    }
  } else {               // Use set logic.
    VariableSet<T> vars; // ensures a consistent ordering
    const auto &intersections = flags.GetIntersections();
    const auto &unions = flags.GetUnions();
    const auto &exclusions = flags.GetExclusions();
    const bool check_excludes = exclusions.size() > 0;

    if (intersections.size() > 0) {
      // Dirty trick to get literally any flag from the intersections set
      MetadataFlag first_required = *(intersections.begin());

      for (auto &v : flagsToVars_[first_required]) {
        const auto &m = v->metadata();
        // TODO(JMM): Note that AnyFlagsSet returns FALSE if the set of flags
        // it's asked about is empty.  Not sure that's desired
        // behaviour, but whatever, let's just guard against edge cases
        // here.
        if (m.AllFlagsSet(intersections) &&
            !(check_excludes && m.AnyFlagsSet(exclusions)) &&
            (unions.empty() || m.AnyFlagsSet(unions))) {
          // TODO(JMM): When dense sparse packing is moved to Parthenon
          // develop we need an extra check for IsAllocated here.
          vars.insert(v);
        }
      }
    } else { // unions.size() > 0.
      for (const auto &f : unions) {
        for (const auto &v : flagsToVars_[f]) {
          // we know intersections.size == 0
          if (!(check_excludes && (v->metadata()).AnyFlagsSet(exclusions))) {
            // TODO(JMM): see above regarding IsAllocated()
            vars.insert(v);
          }
        }
      }
    }
    // Construct the var_list from the set.
    for (auto &v : vars) {
      var_list.Add(v, sparse_ids_set);
    }
  }

  Kokkos::Profiling::popRegion();
  return var_list;
}

template<typename T>
typename MeshBlockData<T>::VarList
MeshBlockData<T>::GetVariablesByUid(const std::vector<Uid_t> &uids) {
  typename MeshBlockData<T>::VarList var_list;
  for (auto i : uids) {
    auto v = GetCellVarPtr(i);
    var_list.Add(v);
  }
  return var_list;
}

template <typename T>
void MeshBlockData<T>::Remove(const std::string &label) {
  throw std::runtime_error("MeshBlockData<T>::Remove not yet implemented");
}

template <typename T>
void MeshBlockData<T>::Print() {
  std::cout << "Variables are:\n";
  for (auto v : varVector_) {
    std::cout << " cell: " << v->info() << std::endl;
  }
}

template class MeshBlockData<double>;

} // namespace parthenon
