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

#include <regex>
#include <set>
#include <string>
#include <vector>

#include "pack/make_pack_descriptor.hpp"

namespace parthenon {
namespace impl {
PackDescriptor MakePackDescriptorBase(StateDescriptor *psd,
                                      const std::vector<std::string> &vars,
                                      const std::vector<bool> &use_regex,
                                      const std::vector<MetadataFlag> &flags,
                                      const std::set<PDOpt> &options) {
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

  return PackDescriptor(psd, vars, selector, options);
}

PackDescriptor MakePackDescriptorBase(StateDescriptor *psd,
                                      const std::vector<Uid_t> &var_ids,
                                      const std::vector<MetadataFlag> &flags,
                                      const std::set<PDOpt> &options) {
  auto selector = [&](int vidx, const VarID &id, const Metadata &md) {
    if (flags.size() > 0) {
      for (const auto &flag : flags) {
        if (!md.IsSet(flag)) return false;
      }
    }
    if (Variable<Real>::GetUniqueID(id.label()) == var_ids[vidx]) return true;
    return false;
  };

  return PackDescriptor(psd, var_ids, selector, options);
}

std::string GetDescIdentifierString(const std::vector<std::string> &vars,
                                    const std::vector<bool> &use_regex,
                                    const std::vector<MetadataFlag> &flags,
                                    const std::set<PDOpt> &options) {
  std::string s;
  for (auto &&var : vars) {
    s += var;
    s += " ";
  }
  for (auto &&reg : use_regex)
    s += std::to_string(reg);
  s += "((";
  for (auto &&flag : flags) {
    s += std::to_string(flag.Flag());
    s += " ";
  }
  s += "))";
  for (auto &&option : options)
    s += std::to_string(static_cast<int>(option));
  s += " ";
  return s;
}

std::string GetDescIdentifierString(const std::vector<Uid_t> &var_ids,
                                    const std::vector<MetadataFlag> &flags,
                                    const std::set<PDOpt> &options) {
  std::string s{"((UIDPACK)) "};
  for (auto &&var : var_ids) {
    s += std::to_string(var);
    s += " ";
  }
  s += "((";
  for (auto &&flag : flags) {
    s += std::to_string(flag.Flag());
    s += " ";
  }
  s += "))";
  for (auto &&option : options)
    s += std::to_string(static_cast<int>(option));
  s += " ";
  return s;
}

template <class MT>
void SetMeshAndStateDescriptor(MT *pmd, Mesh *pmesh, StateDescriptor *psd) {
  psd = nullptr;
  pmesh = nullptr;
  if constexpr (std::is_same_v<MT, MeshData<Real>> ||
                std::is_same_v<MT, MeshBlockData<Real>>) {
    pmesh = pmd->GetMeshPointer();
    psd = pmesh->resolved_packages.get();
  } else if constexpr (std::is_same_v<MT, Mesh>) {
    pmesh = pmd;
    psd = pmesh->resolved_packages.get();
  } else if constexpr (std::is_same_v<MT, StateDescriptor>) {
    psd = pmd;
  }
}

template void SetMeshAndStateDescriptor<MeshData<Real>>(MeshData<Real> *pmd, Mesh *pmesh,
                                                        StateDescriptor *psd);
template void SetMeshAndStateDescriptor<MeshBlockData<Real>>(MeshBlockData<Real> *pmd,
                                                             Mesh *pmesh,
                                                             StateDescriptor *psd);
template void SetMeshAndStateDescriptor<Mesh>(Mesh *pmd, Mesh *pmesh,
                                              StateDescriptor *psd);
template void SetMeshAndStateDescriptor<StateDescriptor>(StateDescriptor *pmd,
                                                         Mesh *pmesh,
                                                         StateDescriptor *psd);

} // namespace impl
} // namespace parthenon
