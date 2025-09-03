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
#ifndef PACK_PACK_DESCRIPTOR_HPP_
#define PACK_PACK_DESCRIPTOR_HPP_

#include <algorithm>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "interface/metadata.hpp"
#include "interface/state_descriptor.hpp"
#include "interface/var_id.hpp"
#include "utils/hash.hpp"
#include "utils/unique_id.hpp"

namespace parthenon {

enum class PDOpt { WithFluxes, Coarse, Flatten };

namespace impl {
struct PackDescriptor {
  using VariableGroup_t = std::vector<std::pair<VarID, Uid_t>>;
  using SelectorFunction_t = std::function<bool(int, const VarID &, const Metadata &)>;
  using SelectorFunctionUid_t = std::function<bool(int, const Uid_t &, const Metadata &)>;

  void Print() const;

  // default constructor needed for certain use cases
  PackDescriptor()
      : nvar_groups(0), var_group_names({}), var_groups({}), with_fluxes(false),
        coarse(false), flat(false), identifier(""), nvar_tot(0) {}

  template <class GROUP_t, class SELECTOR_t>
  PackDescriptor(StateDescriptor *psd, const std::vector<GROUP_t> &var_groups_in,
                 const SELECTOR_t &selector, const std::set<PDOpt> &options)
      : nvar_groups(var_groups_in.size()), var_group_names(MakeGroupNames(var_groups_in)),
        var_groups(BuildUids(var_groups_in.size(), psd, selector)),
        with_fluxes(options.count(PDOpt::WithFluxes)),
        coarse(options.count(PDOpt::Coarse)), flat(options.count(PDOpt::Flatten)),
        identifier(GetIdentifier()), nvar_tot(GetNVarsTotal(var_groups)) {
    PARTHENON_REQUIRE(!(with_fluxes && coarse),
                      "Probably shouldn't be making a coarse pack with fine fluxes.");
  }

  const int nvar_groups;
  const std::vector<std::string> var_group_names;
  const std::vector<VariableGroup_t> var_groups;
  const bool with_fluxes;
  const bool coarse;
  const bool flat;
  const std::string identifier;
  const std::size_t nvar_tot;

 private:
  static int GetNVarsTotal(const std::vector<VariableGroup_t> &var_groups) {
    int nvar_tot = 0;
    for (const auto &group : var_groups) {
      for (const auto &[a, b] : group) {
        nvar_tot++;
      }
    }
    return nvar_tot;
  }
  std::string GetIdentifier() {
    std::string ident("");
    for (const auto &vgroup : var_groups) {
      for (const auto &[vid, uid] : vgroup) {
        ident += std::to_string(uid) + "_";
      }
      ident += "|";
    }
    ident += std::to_string(with_fluxes);
    ident += std::to_string(coarse);
    ident += std::to_string(flat);
    return ident;
  }
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

#endif // PACK_PACK_DESCRIPTOR_HPP_
