//========================================================================================
// (C) (or copyright) 2020-2025. Triad National Security, LLC. All rights reserved.
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

#include <memory>
#include <regex>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "mesh/meshblock.hpp"
#include "pack/make_pack_descriptor.hpp"

namespace parthenon {
namespace impl {
class PackDescCache : public PackDescriptorCacheBase {
 public:
  using key_t = std::tuple<std::vector<std::string>, std::vector<bool>,
                           std::vector<MetadataFlag>, std::set<PDOpt>>;
  std::unordered_map<key_t, impl::PackDescriptor> map;
};

PackDescriptor MakePackDescriptorBase(StateDescriptor *psd, Mesh *pmesh,
                                      const std::vector<std::string> &vars,
                                      const std::vector<bool> &use_regex,
                                      const std::vector<MetadataFlag> &flags,
                                      const std::set<PDOpt> &options) {
  const std::string cache_label{"normal"};
  if (pmesh) {
    if (!pmesh->pack_desc_cache_map.count(cache_label)) {
      // Create a cache for PackDescriptors created with this particular selector
      auto pcache = std::make_shared<PackDescCache>();
      pmesh->pack_desc_cache_map.emplace(
          cache_label, std::dynamic_pointer_cast<PackDescriptorCacheBase>(pcache));
    } else {
      // Check if a PackDescriptor already exists in the cache
      auto pcache = std::dynamic_pointer_cast<PackDescCache>(
          pmesh->pack_desc_cache_map[cache_label]);
      auto key = std::make_tuple(vars, use_regex, flags, options);
      if (pcache->map.count(key)) return pcache->map[key];
    }
  }

  PARTHENON_REQUIRE(vars.size() == use_regex.size(),
                    "Vargroup names and use_regex need to be the same size.");
  auto selector = [&](int vidx, const VarID &id, const Metadata &md) {
    if (flags.size() > 0) {
      for (const auto &flag : flags) {
        if (!md.IsSet(flag)) return false;
      }
    }

    if (use_regex[vidx]) {
      if (std::regex_match(std::string(id.label()), std::regex(vars[vidx]))) return true;
    } else {
      if (vars[vidx] == id.label()) return true;
      if (vars[vidx] == id.base_name && id.sparse_id != InvalidSparseID) return true;
    }
    return false;
  };

  auto pd = PackDescriptor(psd, vars, selector, options);

  if (pmesh) {
    // Store the newly created PackDescriptor in the cache
    auto pcache =
        std::dynamic_pointer_cast<PackDescCache>(pmesh->pack_desc_cache_map[cache_label]);
    auto key = std::make_tuple(vars, use_regex, flags, options);
    pcache->map.emplace(key, pd);
  }
  return pd;
}

class PackDescUidCache : public PackDescriptorCacheBase {
 public:
  using key_t =
      std::tuple<std::vector<Uid_t>, std::vector<MetadataFlag>, std::set<PDOpt>>;
  std::unordered_map<key_t, impl::PackDescriptor> map;
};

PackDescriptor MakePackDescriptorBase(StateDescriptor *psd, Mesh *pmesh,
                                      const std::vector<Uid_t> &var_ids,
                                      const std::vector<MetadataFlag> &flags,
                                      const std::set<PDOpt> &options) {
  const std::string cache_label{"uid"};
  if (pmesh) {
    if (!pmesh->pack_desc_cache_map.count(cache_label)) {
      auto pcache = std::make_shared<PackDescUidCache>();
      pmesh->pack_desc_cache_map.emplace(
          cache_label, std::dynamic_pointer_cast<PackDescriptorCacheBase>(pcache));
    }
    auto pcache = std::dynamic_pointer_cast<PackDescUidCache>(
        pmesh->pack_desc_cache_map[cache_label]);
    auto key = std::make_tuple(var_ids, flags, options);
    if (pcache->map.count(key)) return pcache->map[key];
  }

  auto selector = [&](int vidx, const VarID &id, const Metadata &md) {
    if (flags.size() > 0) {
      for (const auto &flag : flags) {
        if (!md.IsSet(flag)) return false;
      }
    }
    if (Variable<Real>::GetUniqueID(id.label()) == var_ids[vidx]) return true;
    return false;
  };

  auto pd = PackDescriptor(psd, var_ids, selector, options);
  if (pmesh) {
    auto pcache = std::dynamic_pointer_cast<PackDescUidCache>(
        pmesh->pack_desc_cache_map[cache_label]);
    auto key = std::make_tuple(var_ids, flags, options);
    pcache->map.emplace(key, pd);
  }

  return PackDescriptor(psd, var_ids, selector, options);
}

template <class MT>
void SetMeshAndStateDescriptor(MT *pmd, Mesh *&pmesh, StateDescriptor *&psd) {
  psd = nullptr;
  pmesh = nullptr;
  if constexpr (std::is_same_v<MT, MeshData<Real>> ||
                std::is_same_v<MT, MeshBlockData<Real>>) {
    pmesh = pmd->GetMeshPointer();
    psd = pmesh->resolved_packages.get();
  } else if constexpr (std::is_same_v<MT, MeshBlock>) {
    pmesh = pmd->pmy_mesh;
    psd = pmesh->resolved_packages.get();
  } else if constexpr (std::is_same_v<MT, Mesh>) {
    pmesh = pmd;
    psd = pmesh->resolved_packages.get();
  } else if constexpr (std::is_same_v<MT, StateDescriptor>) {
    psd = pmd;
  }
}

template void SetMeshAndStateDescriptor<MeshData<Real>>(MeshData<Real> *pmd, Mesh *&pmesh,
                                                        StateDescriptor *&psd);
template void SetMeshAndStateDescriptor<MeshBlockData<Real>>(MeshBlockData<Real> *pmd,
                                                             Mesh *&pmesh,
                                                             StateDescriptor *&psd);
template void SetMeshAndStateDescriptor<Mesh>(Mesh *pmd, Mesh *&pmesh,
                                              StateDescriptor *&psd);
template void SetMeshAndStateDescriptor<MeshBlock>(MeshBlock *pmd, Mesh *&pmesh,
                                                   StateDescriptor *&psd);
template void SetMeshAndStateDescriptor<StateDescriptor>(StateDescriptor *pmd,
                                                         Mesh *&pmesh,
                                                         StateDescriptor *&psd);

} // namespace impl
} // namespace parthenon
