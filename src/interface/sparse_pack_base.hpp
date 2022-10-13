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

template <unsigned int NMAX>
class PackIdx {
 public:
  KOKKOS_INLINE_FUNCTION
  explicit PackIdx(int var_start_idx) : vidx(var_start_idx), offset(0) {}
  //KOKKOS_INLINE_FUNCTION
  //PackIdx(std::size_t var_idx, int off) : vidx(var_idx), offset(off) {}

//  KOKKOS_INLINE_FUNCTION
//  PackIdx &operator=(std::size_t var_idx) {
//    vidx = var_idx;
//    offset = 0;
//    return *this;
//  }

  PackIdx(std::vector<int> shape, int var_start_idx) : vidx_(vidx), ndim_(shape.size() {
    PARTHENON_REQUIRE_THROWS(shape.size() <= NMAX, "Requested rank too large");
    for (int i = 0; i < shape.size(); ++i) {
      shape_[i] = shape[i];
    }
  }

  KOKKOS_INLINE_FUNCTION
  int DimSize(int iDim) const {
    PARTHENON_DEBUG_REQUIRE(iDim <= ndim_, "Wrong number of dimensions.");
    return shape_[iDim - 1];
  }

  IndexRange GetBounds(int iDim) const {
    PARTHENON_REQUIRE_THROWS(iDim > ndim_"Dimension " + std::to_string(iDim) + " greater than rank " +
      std::to_string(ndim_) + ".");
    IndexRange rng;
    rng.s = 0;
    rng.e = shape_[iDim - 1] - 1;
    return rng;
  }

  KOKKOS_INLINE_FUNCTION
  bool IsValid() { return vidx_ >= 0; }

  KOKKOS_FORCEINLINE_FUNCTION
  int operator()() const {
    PARTHENON_DEBUG_REQUIRE(ndim_ == 0, "Wrong number of dimensions.");
    return vidx_;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  int operator()(const int idx1) const {
    PARTHENON_DEBUG_REQUIRE(ndim_ == 1, "Wrong number of dimensions.");
    PARTHENON_DEBUG_REQUIRE(idx1 < shape_[0], "Idx1 too large.");
    return vidx_ + idx1;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  int operator()(const int idx1, const int idx2) const {
    PARTHENON_DEBUG_REQUIRE(ndim_ == 2, "Wrong number of dimensions.");
    PARTHENON_DEBUG_REQUIRE(idx1 < shape_[0], "Idx1 too large.");
    PARTHENON_DEBUG_REQUIRE(idx2 < shape_[1], "Idx2 too large.");
    return vidx_ + idx1 + shape_[0] * idx2;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  int operator()(const int idx1, const int idx2, const int idx3) const {
    PARTHENON_DEBUG_REQUIRE(ndim_ == 3, "Wrong number of dimensions.");
    PARTHENON_DEBUG_REQUIRE(idx1 < shape_[0], "Idx1 too large.");
    PARTHENON_DEBUG_REQUIRE(idx2 < shape_[1], "Idx2 too large.");
    PARTHENON_DEBUG_REQUIRE(idx3 < shape_[2], "Idx3 too large.");
    return vidx_ + idx1 + shape_[0] * (idx2 + shape_[1] * idx3);
  }

  KOKKOS_INLINE_FUNCTION
  std::size_t VariableIdx() { return vidx; }
//  KOKKOS_INLINE_FUNCTION
//  int Offset() { return offset; }



 private:
  int vidx_;
  int shape_[NMAX];
  int ndim_;
  //int offset;
};

// Map for going from variable names to sparse pack variable indices
using SparsePackIdxMap = std::unordered_map<std::string, std::size_t>;
//class SparsePackIdxMap {
// public:
//  SparsePackIdxMap() = default;
//
//  const auto &Map() const { return map_; }
//
//  const PackIdx &operator[](const std::string &key) const {
//    //static const PackIdx invalid_indices(-2
//  }
//
// private:
//  std::unordered_map<std::string, std::size_t> map_;
//};

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
  bool IncludeVariable(int vidx, const std::shared_ptr<CellVariable<Real>> &pv) const {
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
  // TODO(BRR) store type here? Maybe unnecessary
};

struct SwarmPackDescriptor {
  SwarmPackDescriptor(const std::string &swarm_name, const std::vector<std::string> &vars)
      : swarm_name(swarm_name), vars(vars) {}

  // Method for determining if variable pv should be included in pack for this
  // PackDescriptor
  bool IncludeVariable(int vidx, const std::shared_ptr<ParticleVariable<Real>> &pv) const {
    // TODO(LFR): Check that the shapes agree
    //if (flags.size() > 0) {
    //  for (const auto &flag : flags) {
    //    if (!pv->IsSet(flag)) {
    //      return false;
    //    }
    //  }
    //}

    //if (use_regex[vidx]) {
    //  if (std::regex_match(std::string(pv->label()), regexes[vidx])) return true;
    //} else {
      if (vars[vidx] == pv->label()) return true;
    //}
    return false;
  }

  std::string swarm_name;
  std::vector<std::string> vars;
};
} // namespace impl

struct BasePackType {
  static constexpr int Variable = 0;
  static constexpr int Swarm = 1;
};

// N is a switch for field vs swarm packing, 0 = fields 1 = swarm
template <unsigned int N = 0, typename TYPE = Real>
class SparsePackBase {
 public:
  SparsePackBase() = default;
  virtual ~SparsePackBase() = default;

 //protected:
  friend class SparsePackCache;

  using alloc_t = std::vector<bool>;
  using pack_t = typename std::tuple_element<
      N, std::tuple<ParArray3D<ParArray3D<TYPE>>, ParArray3D<ParArray1D<TYPE>>>>::type;
  // using pack_t = ParArray3D<ParArray3D<Real>>;
  using bounds_t = ParArray3D<int>;
  using coords_t = ParArray1D<ParArray0D<Coordinates_t>>;
  using desc_t = typename std::tuple_element<
    N, std::tuple<impl::PackDescriptor, impl::SwarmPackDescriptor>>::type;

  //// Return a map from variable names to pack variable indices
  static SparsePackIdxMap GetIdxMap(const desc_t &desc) {
    SparsePackIdxMap map;
    std::size_t idx = 0;
    for (const auto &var : desc.vars) {
      map[var] = idx;
      ++idx;
    }
    return map;
  }

  // Methods for getting parts of the shape of the pack
  KOKKOS_FORCEINLINE_FUNCTION
  int GetNBlocks() const { return nblocks_; }

  // TODO(BRR) public?
 public:
  pack_t pack_;
  bounds_t bounds_;
  coords_t coords_;

  bool with_fluxes_;
  bool coarse_;
  int nblocks_;
  int ndim_;
  int dims_[6];
  int nvar_;

  // Unused for compile-time version
  SparsePackIdxMap pack_map_;
};

template <typename T = Real>
using VariablePackBase = SparsePackBase<BasePackType::Variable, T>;
template <typename T = Real>
using SwarmPackBase = SparsePackBase<BasePackType::Swarm, T>;

// Object for cacheing sparse packs in MeshData and MeshBlockData objects. This
// handles checking for a pre-existing pack and creating a new SparsePackBase if
// a cached pack is unavailable. Essentially, this operates as a map from
// `PackDescriptor` to `SparsePackBase`
class SparsePackCache {
 public:
  std::size_t size() const { return pack_map.size(); }

  void clear() { pack_map.clear(); }

  // TODO(BRR) public?
  // protected:
  std::string GetIdentifier(const PackDescriptor &desc) const {
    std::string identifier("");
    for (const auto &flag : desc.flags)
      identifier += flag.Name();
    identifier += "____";
    for (int i = 0; i < desc.vars.size(); ++i)
      identifier += desc.vars[i] + std::to_string(desc.use_regex[i]);
    return identifier;
  }

  std::unordered_map<std::string, std::pair<SparsePackBase<>, SparsePackBase<>::alloc_t>>
      pack_map;

  // friend class SparsePackBase;
};

// TODO(BRR) stop copying code
class SwarmPackCache {
 public:
  std::size_t size() const { return pack_map.size(); }

  void clear() { pack_map.clear(); }

  // TODO(BRR) public?
  // protected:
  static std::string GetIdentifier(const SwarmPackDescriptor &desc) {
    std::string identifier("");
//    for (const auto &flag : desc.flags)
//      identifier += flag.Name();
//    identifier += "____";
    for (int i = 0; i < desc.vars.size(); ++i)
      identifier += desc.vars[i];// + std::to_string(desc.use_regex[i]);
    identifier += "____swarmname:";
    identifier += desc.swarm_name;
    return identifier;
  }

//  std::string GetIdentifier(const SwarmPackDescriptor &desc) const {
//    std::string identifier("");
//    for (const auto &flag : desc.flags)
//      identifier += flag.Name();
//    identifier += "____";
//    for (int i = 0; i < desc.vars.size(); ++i)
//      identifier += desc.vars[i] + std::to_string(desc.use_regex[i]);
//    return identifier;
//  }

  std::unordered_map<std::string, std::pair<SwarmPackBase<>, SwarmPackBase<>::alloc_t>>
      pack_map;

//  friend class SwarmPackBase<>;
};

} // namespace parthenon

#endif // INTERFACE_SPARSE_PACK_BASE_HPP_
