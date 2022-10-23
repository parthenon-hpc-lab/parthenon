//========================================================================================
// (C) (or copyright) 2020-2022. Triad National Security, LLC. All rights reserved.
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
#include <regex>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "coordinates/coordinates.hpp"
#include "interface/variable.hpp"
#include "utils/utils.hpp"

namespace parthenon {
class SparsePackCache;

// Map for going from variable names to sparse pack variable indices
using SparsePackIdxMap = std::unordered_map<std::string, std::size_t>;

namespace impl {
struct PackDescriptor {
  PackDescriptor(const std::vector<std::string> &vars, const std::vector<bool> &use_regex,
                 const std::vector<MetadataFlag> &flags, bool with_fluxes, bool coarse)
      : vars(vars), use_regex(use_regex), flags(flags), with_fluxes(with_fluxes),
        coarse(coarse) {
    PARTHENON_REQUIRE(use_regex.size() == vars.size(),
                      "Must have a regex flag for each variable.");
    PARTHENON_REQUIRE(!(with_fluxes && coarse),
                      "Probably shouldn't be making a coarse pack with fine fluxes.");
    for (const auto &var : vars)
      regexes.push_back(std::regex(var));
  }

  PackDescriptor(const std::vector<std::pair<std::string, bool>> &vars_in,
                 const std::vector<MetadataFlag> &flags, bool with_fluxes, bool coarse)
      : flags(flags), with_fluxes(with_fluxes), coarse(coarse) {
    for (auto var : vars_in) {
      vars.push_back(var.first);
      use_regex.push_back(var.second);
    }
    PARTHENON_REQUIRE(!(with_fluxes && coarse),
                      "Probably shouldn't be making a coarse pack with fine fluxes.");
    for (const auto &var : vars)
      regexes.push_back(std::regex(var));
  }

  PackDescriptor(const std::vector<std::string> &vars_in,
                 const std::vector<MetadataFlag> &flags, bool with_fluxes, bool coarse)
      : vars(vars_in), use_regex(vars_in.size(), false), flags(flags),
        with_fluxes(with_fluxes), coarse(coarse) {
    PARTHENON_REQUIRE(!(with_fluxes && coarse),
                      "Probably shouldn't be making a coarse pack with fine fluxes.");
    for (const auto &var : vars)
      regexes.push_back(std::regex(var));
  }

  // Method for determining if variable pv should be included in pack for this
  // PackDescriptor
  bool IncludeVariable(int vidx, const std::shared_ptr<Variable<Real>> &pv) const {
    // TODO(LFR): Check that the shapes agree
    if (flags.size() > 0) {
      for (const auto &flag : flags) {
        if (!pv->IsSet(flag)) {
          return false;
        }
      }
    }

    if (use_regex[vidx]) {
      if (std::regex_match(std::string(pv->label()), regexes[vidx])) return true;
    } else {
      if (vars[vidx] == pv->label()) return true;
    }
    return false;
  }

  std::vector<std::string> vars;
  std::vector<std::regex> regexes;
  std::vector<bool> use_regex;
  std::vector<MetadataFlag> flags;
  bool with_fluxes;
  bool coarse;
};
} // namespace impl

class SparsePackBase {
 public:
  SparsePackBase() = default;
  virtual ~SparsePackBase() = default;

 protected:
  friend class SparsePackCache;

  using alloc_t = std::vector<bool>;
  using pack_t = ParArray3D<ParArray3D<Real>>;
  using bounds_t = ParArray3D<int>;
  using coords_t = ParArray1D<ParArray0D<Coordinates_t>>;

  // Returns a SparsePackBase object that is either newly created or taken
  // from the cache in pmd. The cache itself handles the all of this logic
  template <class T>
  static SparsePackBase GetPack(T *pmd, const impl::PackDescriptor &desc) {
    auto &cache = pmd->GetSparsePackCache();
    return cache.Get(pmd, desc);
  }

  // Return a map from variable names to pack variable indices
  static SparsePackIdxMap GetIdxMap(const impl::PackDescriptor &desc) {
    SparsePackIdxMap map;
    std::size_t idx = 0;
    for (const auto &var : desc.vars) {
      map[var] = idx;
      ++idx;
    }
    return map;
  }

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
  coords_t coords_;

  bool with_fluxes_;
  bool coarse_;
  int nblocks_;
  int ndim_;
  int dims_[6];
  int nvar_;
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

} // namespace parthenon

#endif // INTERFACE_SPARSE_PACK_BASE_HPP_
