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
namespace impl {
struct PackDescriptor {
  std::vector<std::string> vars;
  std::vector<bool> use_regex;
  std::vector<MetadataFlag> flags;
  bool with_fluxes;
  bool coarse;

  PackDescriptor(const std::vector<std::string> &vars, const std::vector<bool> &use_regex,
                 const std::vector<MetadataFlag> &flags, bool with_fluxes, bool coarse)
      : vars(vars), use_regex(use_regex), flags(flags), with_fluxes(with_fluxes),
        coarse(coarse) {
    PARTHENON_REQUIRE(use_regex.size() == vars.size(),
                      "Must have a regex flag for each variable.");
    PARTHENON_REQUIRE(!(with_fluxes && coarse),
                      "Probably shouldn't be making a coarse pack with fine fluxes.");
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
  }

  PackDescriptor(const std::vector<std::string> &vars_in,
                 const std::vector<MetadataFlag> &flags, bool with_fluxes, bool coarse)
      : vars(vars_in), use_regex(vars_in.size(), false), flags(flags),
        with_fluxes(with_fluxes), coarse(coarse) {
    PARTHENON_REQUIRE(!(with_fluxes && coarse),
                      "Probably shouldn't be making a coarse pack with fine fluxes.");
  }
};
} // namespace impl

using namespace impl;

class PackIdx {
  std::size_t vidx;
  int off;

 public:
  KOKKOS_INLINE_FUNCTION
  explicit PackIdx(std::size_t var_idx) : vidx(var_idx), off(0) {}
  KOKKOS_INLINE_FUNCTION
  PackIdx &operator=(std::size_t var_idx) {
    vidx = var_idx;
    off = 0;
    return *this;
  }
  KOKKOS_INLINE_FUNCTION
  PackIdx(std::size_t var_idx, int off) : vidx(var_idx), off(off) {}
  KOKKOS_INLINE_FUNCTION
  std::size_t Vidx() { return vidx; }
  KOKKOS_INLINE_FUNCTION
  int Off() { return off; }
};

template <class T, class = typename std::enable_if<std::is_integral<T>::value>::type>
KOKKOS_INLINE_FUNCTION PackIdx operator+(PackIdx idx, T off) {
  return PackIdx(idx.Vidx(), idx.Off() + off);
}

template <class T, class = typename std::enable_if<std::is_integral<T>::value>::type>
KOKKOS_INLINE_FUNCTION PackIdx operator+(T off, PackIdx idx) {
  return idx + off;
}

using SparsePackIdxMap = std::unordered_map<std::string, std::size_t>;

class SparsePackBase {
 protected:
  friend class SparsePackCache;

  using test_func_t =
      std::function<bool(int, const std::shared_ptr<CellVariable<Real>> &)>;
  using pack_t = ParArray3D<ParArray3D<Real>>;
  using bounds_t = ParArray3D<int>;
  using alloc_t = std::vector<bool>;
  using coords_t = ParArray1D<ParArray0D<Coordinates_t>>;

  pack_t pack_;
  bounds_t bounds_;
  coords_t coords_;

  bool with_fluxes_;
  bool coarse_;
  int nblocks_;
  int ndim_;
  int dims_[6];
  int nvar_;

//Get a list of booleans of the allocation status of every variable in pmd matching the PackDescriptor desc
  template <class T>
  static alloc_t GetAllocStatus(T *pmd, const PackDescriptor &desc);

  //Build a usable `SparsePackBase` from the variables specified in desc contained in the blocks  in a MeshBlockData/MeshBlock with a variable pack allocated on the device.
  template <class T>
  static SparsePackBase Build(T *pmd, const PackDescriptor &desc);

  static test_func_t GetTestFunction(const PackDescriptor &desc);

 public:
  SparsePackBase() = default;
  virtual ~SparsePackBase() = default;

  // VAR_VEC can be:
  //   1) std::vector<std::string> of variable names (in which case they are all assumed
  //   not to be regexs)
  //   2) std::vector<std::pair<std::string, bool>> of (variable name, treat name as
  //   regex) pairs
  template <class T, class VAR_VEC>
  static std::tuple<SparsePackBase, SparsePackIdxMap>
  Make(T *pmd, const VAR_VEC &vars, const std::vector<MetadataFlag> &flags = {},
       bool fluxes = false, bool coarse = false) {
    auto &cache = pmd->GetSparsePackCache();
    auto desc = PackDescriptor(vars, flags, fluxes, coarse);
    SparsePackIdxMap map;
    std::size_t idx = 0;
    for (const auto &var : desc.vars) {
      map[var] = idx;
      ++idx;
    }
    return {cache.Get(pmd, desc), map};
  }

  template <class T, class VAR_VEC>
  static std::tuple<SparsePackBase, SparsePackIdxMap>
  MakeWithFluxes(T *pmd, const VAR_VEC &vars,
                 const std::vector<MetadataFlag> &flags = {}) {
    const bool fluxes = true;
    const bool coarse = false;
    return Make(pmd, vars, flags, fluxes, coarse);
  }

  template <class T, class VAR_VEC>
  static std::tuple<SparsePackBase, SparsePackIdxMap>
  MakeWithCoarse(T *pmd, const VAR_VEC &vars,
                 const std::vector<MetadataFlag> &flags = {}) {
    const bool fluxes = false;
    const bool coarse = true;
    return Make(pmd, vars, flags, fluxes, coarse);
  }
};
//Map of `PackDescriptor` to `SparsePackBase`
class SparsePackCache {
 public:
  template <class T>
  SparsePackBase &Get(T *pmd, const PackDescriptor &desc);

  std::size_t size() const { return pack_map.size(); }

  void clear() { pack_map.clear(); }

 protected:
  std::string GetIdentifier(const PackDescriptor &desc) const;

  std::unordered_map<std::string, std::pair<SparsePackBase, SparsePackBase::alloc_t>>
      pack_map;
};

} // namespace parthenon

#endif // INTERFACE_SPARSE_PACK_BASE_HPP_
