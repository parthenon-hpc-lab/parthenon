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
#ifndef PACK_MAKE_PACK_DESCRIPTOR_HPP_
#define PACK_MAKE_PACK_DESCRIPTOR_HPP_

#include <algorithm>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "interface/mesh_data.hpp"
#include "interface/meshblock_data.hpp"
#include "interface/metadata.hpp"
#include "interface/state_descriptor.hpp"
#include "interface/variable.hpp"
#include "mesh/mesh.hpp"
#include "pack/sparse_pack.hpp"
#include "utils/type_list.hpp"

namespace parthenon {

namespace impl {
PackDescriptor MakePackDescriptorBase(StateDescriptor *psd,
                                      const std::vector<std::string> &vars,
                                      const std::vector<bool> &use_regex,
                                      const std::vector<MetadataFlag> &flags,
                                      const std::set<PDOpt> &options);
PackDescriptor MakePackDescriptorBase(StateDescriptor *psd,
                                      const std::vector<Uid_t> &var_ids,
                                      const std::vector<MetadataFlag> &flags,
                                      const std::set<PDOpt> &options);
std::string GetDescIdentifierString(const std::vector<std::string> &vars,
                                    const std::vector<bool> &use_regex,
                                    const std::vector<MetadataFlag> &flags,
                                    const std::set<PDOpt> &options);
std::string GetDescIdentifierString(const std::vector<Uid_t> &var_ids,
                                    const std::vector<MetadataFlag> &flags,
                                    const std::set<PDOpt> &options);
template <class MT>
void SetMeshAndStateDescriptor(MT *pmd, Mesh *pmesh, StateDescriptor *psd);
} // namespace impl

inline auto MakeDefaultPackDescriptor() { return typename SparsePack<>::Descriptor(); }

template <class MT>
inline auto MakePackDescriptor(MT *pmd, const std::vector<std::string> &vars,
                               const std::vector<bool> &use_regex,
                               const std::vector<MetadataFlag> &flags = {},
                               const std::set<PDOpt> &options = {}) {
  const auto identifier = impl::GetDescIdentifierString(vars, use_regex, flags, options);
  Mesh *pmesh{nullptr};
  StateDescriptor *psd{nullptr};
  SetMeshAndStateDescriptor<MT>(pmd, pmesh, psd);

  // Pull the descriptor base out of cache stored in mesh if possible
  if (pmesh && pmesh->pack_map.count(identifier))
    return typename SparsePack<>::Descriptor(pmesh->pack_map[identifier]);

  auto desc_base = impl::MakePackDescriptorBase(psd, vars, use_regex, flags, options);

  // Store this in the cache for next time around if possible
  if (pmesh) pmesh->pack_map.emplace(identifier, desc_base);
  return typename SparsePack<>::Descriptor(desc_base);
}

template <class MT>
inline auto MakePackDescriptor(MT *pmd, const std::vector<std::string> &vars,
                               const std::vector<MetadataFlag> &flags = {},
                               const std::set<PDOpt> &options = {}) {
  return MakePackDescriptor(pmd, vars, std::vector<bool>(vars.size(), false), flags,
                            options);
}

template <class... Ts, class MT>
inline auto MakePackDescriptor(MT *pmd, const std::vector<MetadataFlag> &flags = {},
                               const std::set<PDOpt> &options = {}) {
  const std::vector<std::string> vars{Ts::name()...};
  const std::vector<bool> use_regex{Ts::regex()...};
  return typename SparsePack<Ts...>::Descriptor(
      MakePackDescriptor(pmd, vars, use_regex, flags, options));
}

template <class... Ts, class MT>
inline auto MakePackDescriptor(SparsePack<Ts...> pack, MT *pmd,
                               const std::vector<MetadataFlag> &flags = {},
                               const std::set<PDOpt> &options = {}) {
  return parthenon::MakePackDescriptor<Ts...>(pmd, flags, options);
}

template <class MT>
inline auto
MakePackDescriptor(MT *psd, const std::vector<std::pair<std::string, bool>> &var_regexes,
                   const std::vector<MetadataFlag> &flags = {},
                   const std::set<PDOpt> &options = {}) {
  std::vector<std::string> vars;
  std::vector<bool> use_regex;
  for (const auto &[v, r] : var_regexes) {
    vars.push_back(v);
    use_regex.push_back(r);
  }
  return MakePackDescriptor(psd, vars, use_regex, flags, options);
}

template <class MT>
inline auto MakePackDescriptor(MT *pmd, const std::vector<Uid_t> &var_ids,
                               const std::vector<MetadataFlag> &flags = {},
                               const std::set<PDOpt> &options = {}) {
  const auto identifier = impl::GetDescIdentifierString(var_ids, flags, options);
  Mesh *pmesh{nullptr};
  StateDescriptor *psd{nullptr};
  SetMeshAndStateDescriptor<MT>(pmd, pmesh, psd);

  // Pull the descriptor base out of cache stored in mesh if possible
  if (pmesh && pmesh->pack_map.count(identifier))
    return typename SparsePack<>::Descriptor(pmesh->pack_map[identifier]);

  auto desc_base = impl::MakePackDescriptorBase(psd, var_ids, flags, options);

  // Store this in the cache for next time around if possible
  if (pmesh) pmesh->pack_map.emplace(identifier, desc_base);
  return typename SparsePack<>::Descriptor(desc_base);
}

template <template <class...> class TL, class... Types, class... Args>
inline auto MakePackDescriptorFromTypeList(TL<Types...>, Args &&...args) {
  return MakePackDescriptor<Types...>(std::forward<Args>(args)...);
}

template <class TL, class... Args>
inline auto MakePackDescriptorFromTypeList(Args &&...args) {
  return MakePackDescriptorFromTypeList(TL(), std::forward<Args>(args)...);
}
} // namespace parthenon

#endif // PACK_MAKE_PACK_DESCRIPTOR_HPP_
