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
#ifndef INTERFACE_MAKE_PACK_DESCRIPTOR_HPP_
#define INTERFACE_MAKE_PACK_DESCRIPTOR_HPP_

#include <algorithm>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <regex>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "interface/mesh_data.hpp"
#include "interface/meshblock_data.hpp"
#include "interface/metadata.hpp"
#include "interface/sparse_pack.hpp"
#include "interface/state_descriptor.hpp"
#include "interface/variable.hpp"
#include "mesh/mesh.hpp"

namespace parthenon {

inline auto MakePackDescriptor(StateDescriptor *psd, const std::vector<std::string> &vars,
                               const std::vector<bool> &use_regex,
                               const std::vector<MetadataFlag> &flags = {},
                               const std::set<PDOpt> &options = {}) {
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

  impl::PackDescriptor base_desc(psd, vars, selector, options);
  return typename SparsePack<>::Descriptor(base_desc);
}

template <class... Ts>
inline auto MakePackDescriptor(StateDescriptor *psd,
                               const std::vector<MetadataFlag> &flags = {},
                               const std::set<PDOpt> &options = {}) {
  static_assert(sizeof...(Ts) > 0, "Must have at least one variable type for type pack");

  std::vector<std::string> vars{Ts::name()...};
  std::vector<bool> use_regex{Ts::regex()...};

  return typename SparsePack<Ts...>::Descriptor(static_cast<impl::PackDescriptor>(
      MakePackDescriptor(psd, vars, use_regex, flags, options)));
}

inline auto MakePackDescriptor(StateDescriptor *psd, const std::vector<std::string> &vars,
                               const std::vector<MetadataFlag> &flags = {},
                               const std::set<PDOpt> &options = {}) {
  return MakePackDescriptor(psd, vars, std::vector<bool>(vars.size(), false), flags,
                            options);
}

template <class... Ts>
inline auto MakePackDescriptor(MeshBlockData<Real> *pmbd,
                               const std::vector<MetadataFlag> &flags = {},
                               const std::set<PDOpt> &options = {}) {
  return MakePackDescriptor<Ts...>(
      pmbd->GetBlockPointer()->pmy_mesh->resolved_packages.get(), flags, options);
}

template <class... Ts>
inline auto MakePackDescriptor(MeshData<Real> *pmd,
                               const std::vector<MetadataFlag> &flags = {},
                               const std::set<PDOpt> &options = {}) {
  return MakePackDescriptor<Ts...>(pmd->GetMeshPointer()->resolved_packages.get(), flags,
                                   options);
}

template <class... Ts>
inline auto MakePackDescriptor(SparsePack<Ts...> pack, StateDescriptor *psd,
                               const std::vector<MetadataFlag> &flags = {},
                               const std::set<PDOpt> &options = {}) {
  return parthenon::MakePackDescriptor<Ts...>(psd, flags, options);
}

inline auto MakePackDescriptor(
    StateDescriptor *psd, const std::vector<std::pair<std::string, bool>> &var_regexes,
    const std::vector<MetadataFlag> &flags = {}, const std::set<PDOpt> &options = {}) {
  std::vector<std::string> vars;
  std::vector<bool> use_regex;
  for (const auto &[v, r] : var_regexes) {
    vars.push_back(v);
    use_regex.push_back(r);
  }
  return MakePackDescriptor(psd, vars, use_regex, flags, options);
}

inline auto MakePackDescriptor(StateDescriptor *psd, const std::vector<Uid_t> &var_ids,
                               const std::vector<MetadataFlag> &flags = {},
                               const std::set<PDOpt> &options = {}) {
  auto selector = [&](int vidx, const VarID &id, const Metadata &md) {
    if (flags.size() > 0) {
      for (const auto &flag : flags) {
        if (!md.IsSet(flag)) return false;
      }
    }
    if (Variable<Real>::GetUniqueID(id.label()) == var_ids[vidx]) return true;
    return false;
  };

  impl::PackDescriptor base_desc(psd, var_ids, selector, options);
  return typename SparsePack<>::Descriptor(base_desc);
}

} // namespace parthenon

#endif // INTERFACE_MAKE_PACK_DESCRIPTOR_HPP_
