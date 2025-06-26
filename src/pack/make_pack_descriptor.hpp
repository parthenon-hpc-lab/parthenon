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

#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "interface/metadata.hpp"
#include "interface/state_descriptor.hpp"
#include "mesh/mesh.hpp"
#include "pack/pack_descriptor.hpp"
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
template <class MT>
StateDescriptor *GetStateDescriptor(MT *pmd);
} // namespace impl

inline auto MakeDefaultPackDescriptor() { return typename SparsePack<>::Descriptor(); }

template <class MT>
inline auto MakePackDescriptor(MT *pmd, const std::vector<std::string> &vars,
                               const std::vector<bool> &use_regex,
                               const std::vector<MetadataFlag> &flags = {},
                               const std::set<PDOpt> &options = {}) {
  return typename SparsePack<>::Descriptor(impl::MakePackDescriptorBase(
      impl::GetStateDescriptor<MT>(pmd), vars, use_regex, flags, options));
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
  return typename SparsePack<Ts...>::Descriptor(static_cast<impl::PackDescriptor>(
      MakePackDescriptor(pmd, vars, use_regex, flags, options)));
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
  return typename SparsePack<>::Descriptor(impl::MakePackDescriptorBase(
      impl::GetStateDescriptor<MT>(pmd), var_ids, flags, options));
}

template <template <class...> class TL, class... Types, class... Args>
inline auto MakePackDescriptorFromTypeList(TL<Types...>, Args &&...args) {
  return MakePackDescriptor<Types...>(std::forward<Args>(args)...);
}

template <class TL, class... Args>
inline auto MakePackDescriptorFromTypeList(Args &&...args) {
  return MakePackDescriptorFromTypeList(TL(), std::forward<Args>(args)...);
}

struct PackDescriptorCacheBase {
  virtual ~PackDescriptorCacheBase() = default;
};

template <class... Ts>
class PackDescCache : public PackDescriptorCacheBase {
 public:
  using key_t = std::tuple<Ts...>;
  std::unordered_map<key_t, impl::PackDescriptor> map;

  static std::optional<impl::PackDescriptor>
  CheckForKey(StateDescriptor *pdesc, const std::string &cache_label, const Ts &...args) {
    if (pdesc) {
      if (!pdesc->pack_desc_cache_map.count(cache_label)) {
        // Create a cache for PackDescriptors created with this particular selector
        auto pcache = std::make_shared<PackDescCache>();
        pdesc->pack_desc_cache_map.emplace(
            cache_label, std::dynamic_pointer_cast<PackDescriptorCacheBase>(pcache));
      } else {
        // Check if a PackDescriptor already exists in the cache
        auto pcache = std::dynamic_pointer_cast<PackDescCache>(
            pdesc->pack_desc_cache_map[cache_label]);
        auto key = std::make_tuple(args...);
        if (pcache->map.count(key))
          return std::optional<impl::PackDescriptor>{pcache->map[key]};
      }
    }
    return std::nullopt;
  }

  static void CachePackDescriptor(StateDescriptor *pdesc, const std::string &cache_label,
                                  const impl::PackDescriptor &pd, const Ts &...args) {
    if (pdesc) {
      // Store the newly created PackDescriptor in the cache
      auto pcache = std::dynamic_pointer_cast<PackDescCache>(
          pdesc->pack_desc_cache_map[cache_label]);
      auto key = std::make_tuple(args...);
      pcache->map.emplace(key, pd);
    }
  }
};

} // namespace parthenon

#endif // PACK_MAKE_PACK_DESCRIPTOR_HPP_
