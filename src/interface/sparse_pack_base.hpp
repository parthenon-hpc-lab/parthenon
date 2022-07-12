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
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "coordinates/coordinates.hpp"
#include "interface/variable.hpp"
#include "utils/utils.hpp"

namespace parthenon {

struct PackDescriptor {
  std::vector<std::string> vars;
  std::vector<bool> use_regex;
  std::vector<MetadataFlag> flags;
  bool with_fluxes;
  bool coarse = false;
  PackDescriptor(const std::vector<std::string> &vars, const std::vector<bool> &use_regex,
                 const std::vector<MetadataFlag> &flags, bool with_fluxes, bool coarse)
      : vars(vars), use_regex(use_regex), flags(flags), with_fluxes(with_fluxes),
        coarse(coarse) {
    PARTHENON_REQUIRE(!(with_fluxes && coarse),
                      "Probably shouldn't be making a coarse pack with fine fluxes.");
  }
};

class SparsePackBase {
 protected:
  friend class SparsePackCache;
  using test_func_t =
      std::function<bool(int, const std::shared_ptr<CellVariable<Real>> &)>;
  using pack_t = ParArray3D<ParArray3D<Real>>;
  using bounds_t = ParArray3D<int>;
  using alloc_t = ParArray1D<bool>::host_mirror_type;
  using coords_t = ParArray1D<ParArray0D<Coordinates_t>>;

  pack_t pack_;
  bounds_t bounds_;
  coords_t coords_;

  ParArray1D<std::string>::host_mirror_type names_h_;
  ParArray1D<bool>::host_mirror_type use_regex_h_;
  alloc_t alloc_status_h_;

  bool with_fluxes_;
  bool coarse_;
  int nblocks_;
  int ndim_;

  template <class T>
  static alloc_t GetAllocStatus(T *pmd, const PackDescriptor &desc);

  template <class T>
  static SparsePackBase Build(T *pmd, const PackDescriptor &desc);

  static test_func_t GetTestFunction(const PackDescriptor &desc);

 public:
  SparsePackBase() = default;
  virtual ~SparsePackBase() = default;

  int GetNBlocks() const { return nblocks_; }
  int GetNDim() const { return ndim_; }

  KOKKOS_INLINE_FUNCTION
  const Coordinates_t &GetCoordinates(const int b) const { return coords_(b)(); }

  KOKKOS_INLINE_FUNCTION
  Real &operator()(const int b, const int idx, const int k, const int j,
                   const int i) const {
    return pack_(0, b, idx)(k, j, i);
  }

  KOKKOS_INLINE_FUNCTION
  Real &flux(const int b, const int dir, const int idx, const int k, const int j,
             const int i) const {
    PARTHENON_DEBUG_REQUIRE(dir > 0 && dir < 4 && with_fluxes_, "Bad input to flux call");
    return pack_(dir, b, idx)(k, j, i);
  }
};

class SparsePackCache {
 public:
  template <class T>
  SparsePackBase &Get(T *pmd, const PackDescriptor &desc) {
    std::string ident = GetIdentifier(desc);
    if (pack_map.count(ident) > 0) {
      auto &pack = pack_map[ident];
      if (desc.with_fluxes != pack.with_fluxes_) goto make_new_pack;
      if (desc.coarse != pack.coarse_) goto make_new_pack;
      auto alloc_status_h = SparsePackBase::GetAllocStatus(pmd, desc);
      if (alloc_status_h.size() != pack.alloc_status_h_.size()) goto make_new_pack;
      for (int i = 0; i < alloc_status_h.size(); ++i) {
        if (alloc_status_h(i) != pack.alloc_status_h_(i)) goto make_new_pack;
      }
      return pack_map[ident];
    }

  make_new_pack:
    pack_map[ident] = SparsePackBase::Build(pmd, desc);
    return pack_map[ident];
  }

  void clear() { pack_map.clear(); }

 protected:
  std::string GetIdentifier(const PackDescriptor &desc) const {
    std::string identifier("");
    for (const auto &flag : desc.flags)
      identifier += flag.Name();
    identifier += "____";
    for (int i = 0; i < desc.vars.size(); ++i)
      identifier += desc.vars[i] + std::to_string(desc.use_regex[i]);
    return identifier;
  }

  std::unordered_map<std::string, SparsePackBase> pack_map;
};

} // namespace parthenon

#endif // INTERFACE_SPARSE_PACK_BASE_HPP_
