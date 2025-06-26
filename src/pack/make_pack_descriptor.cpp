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

PackDescriptor MakePackDescriptorBase(StateDescriptor *psd,
                                      const std::vector<std::string> &vars,
                                      const std::vector<bool> &use_regex,
                                      const std::vector<MetadataFlag> &flags,
                                      const std::set<PDOpt> &options) {
  const std::string cache_label{"normal"};
  using PDCache = PackDescCache<std::vector<std::string>, std::vector<bool>,
                                std::vector<MetadataFlag>, std::set<PDOpt>>;
  auto optional_pd =
      PDCache::CheckForKey(psd, cache_label, vars, use_regex, flags, options);
  if (optional_pd) return *optional_pd;

  PARTHENON_REQUIRE(vars.size() == use_regex.size(),
                    "Vargroup names and use_regex need to be the same size.");
  auto selector = [&](int vidx, const VarID &id, const Metadata &md) {
    for (const auto &flag : flags) {
      if (!md.IsSet(flag)) return false;
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
  PDCache::CachePackDescriptor(psd, cache_label, pd, vars, use_regex, flags, options);
  return pd;
}

PackDescriptor MakePackDescriptorBase(StateDescriptor *psd,
                                      const std::vector<Uid_t> &var_ids,
                                      const std::vector<MetadataFlag> &flags,
                                      const std::set<PDOpt> &options) {
  const std::string cache_label{"uid"};
  using PDCache =
      PackDescCache<std::vector<Uid_t>, std::vector<MetadataFlag>, std::set<PDOpt>>;
  auto optional_pd = PDCache::CheckForKey(psd, cache_label, var_ids, flags, options);
  if (optional_pd) return *optional_pd;

  auto selector = [&](int vidx, const VarID &id, const Metadata &md) {
    for (const auto &flag : flags) {
      if (!md.IsSet(flag)) return false;
    }
    if (Variable<Real>::GetUniqueID(id.label()) == var_ids[vidx]) return true;
    return false;
  };

  auto pd = PackDescriptor(psd, var_ids, selector, options);
  PDCache::CachePackDescriptor(psd, cache_label, pd, var_ids, flags, options);
  return PackDescriptor(psd, var_ids, selector, options);
}

template <class MT>
StateDescriptor *GetStateDescriptor(MT *pmd) {
  if constexpr (std::is_same_v<MT, MeshData<Real>> ||
                std::is_same_v<MT, MeshBlockData<Real>>) {
    return pmd->GetMeshPointer()->resolved_packages.get();
  } else if constexpr (std::is_same_v<MT, MeshBlock>) {
    return pmd->pmy_mesh->resolved_packages.get();
  } else if constexpr (std::is_same_v<MT, Mesh>) {
    return pmd->resolved_packages.get();
  } else if constexpr (std::is_same_v<MT, StateDescriptor>) {
    return pmd;
  }
}

template StateDescriptor *GetStateDescriptor<MeshData<Real>>(MeshData<Real> *pmd);
template StateDescriptor *
GetStateDescriptor<MeshBlockData<Real>>(MeshBlockData<Real> *pmd);
template StateDescriptor *GetStateDescriptor<Mesh>(Mesh *pmd);
template StateDescriptor *GetStateDescriptor<MeshBlock>(MeshBlock *pmd);
template StateDescriptor *GetStateDescriptor<StateDescriptor>(StateDescriptor *pmd);

} // namespace impl
} // namespace parthenon
