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
#ifndef INTERFACE_SPARSE_PACK_HPP_
#define INTERFACE_SPARSE_PACK_HPP_

#include <algorithm>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <regex>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "coordinates/coordinates.hpp"
#include "interface/sparse_pack_base.hpp"
#include "interface/swarm.hpp"
#include "interface/variable.hpp"
#include "utils/concepts_lite.hpp"
#include "utils/utils.hpp"

namespace {
// SFINAE for block iteration so that sparse packs can work for MeshBlockData and MeshData
template <class T, class F>
inline auto ForEachBlock(T *pmd, F func) -> decltype(T().GetBlockData(0), void()) {
  for (int b = 0; b < pmd->NumBlocks(); ++b) {
    auto &pmbd = pmd->GetBlockData(b);
    func(b, pmbd.get());
  }
}

template <class T, class F>
inline auto ForEachBlock(T *pmbd, F func) -> decltype(T().GetBlockPointer(), void()) {
  func(0, pmbd);
}
} // namespace

namespace parthenon {

// Sparse pack index type which allows for relatively simple indexing
// into non-variable name type based SparsePacks (i.e. objects of
// type SparsePack<> which are created with a vector of variable
// names and/or regexes)
class PackIdx {
 public:
  KOKKOS_INLINE_FUNCTION
  explicit PackIdx(std::size_t var_idx) : vidx(var_idx), offset(0) {}
  KOKKOS_INLINE_FUNCTION
  PackIdx(std::size_t var_idx, int off) : vidx(var_idx), offset(off) {}

  KOKKOS_INLINE_FUNCTION
  PackIdx &operator=(std::size_t var_idx) {
    vidx = var_idx;
    offset = 0;
    return *this;
  }

  KOKKOS_INLINE_FUNCTION
  std::size_t VariableIdx() { return vidx; }
  KOKKOS_INLINE_FUNCTION
  int Offset() { return offset; }

 private:
  std::size_t vidx;
  int offset;
};

// Operator overloads to make calls like `my_pack(b, my_pack_idx + 3, k, j, i)` work
template <class T, REQUIRES(std::is_integral<T>::value)>
KOKKOS_INLINE_FUNCTION PackIdx operator+(PackIdx idx, T offset) {
  return PackIdx(idx.VariableIdx(), idx.Offset() + offset);
}

template <class T, REQUIRES(std::is_integral<T>::value)>
KOKKOS_INLINE_FUNCTION PackIdx operator+(T offset, PackIdx idx) {
  return idx + offset;
}

// Namespace in which to put variable name types that are used for
// indexing into SparsePack<[type list of variable name types]> on
// device
namespace variable_names {
// Struct that all variable_name types should inherit from
template <bool REGEX, int... NCOMP>
struct base_t {
  KOKKOS_INLINE_FUNCTION
  base_t() : idx(0) {}

  KOKKOS_INLINE_FUNCTION
  explicit base_t(int idx1) : idx(idx1) {}

  virtual ~base_t() = default;

  // All of these are just static methods so that there is no
  // extra storage in the struct
  static std::string name() {
    PARTHENON_FAIL("Need to implement your own name method.");
    return "error";
  }
  KOKKOS_INLINE_FUNCTION
  static bool regex() { return REGEX; }
  KOKKOS_INLINE_FUNCTION
  static int ndim() { return sizeof...(NCOMP); }
  KOKKOS_INLINE_FUNCTION
  static int size() { return multiply<NCOMP...>::value; }

  const int idx;
};

// An example variable name type that selects all variables available
// on Mesh*Data
struct any : public base_t<true> {
  template <class... Ts>
  KOKKOS_INLINE_FUNCTION any(Ts &&... args) : base_t<true>(std::forward<Ts>(args)...) {}
  static std::string name() { return ".*"; }
};
} // namespace variable_names

template <class... Ts>
class SwarmPack : public SwarmPackBase<> {
  public:
    SwarmPack() = default;

  explicit SwarmPack(const SwarmPackBase<> &spb) : SwarmPackBase<>(spb) {}

  template <class MBD, class T>
  static SwarmPack Get(MBD *pmd, const std::string &swarm_name) {
    const impl::SwarmPackDescriptor desc(swarm_name, std::vector<std::string>{Ts::name()...});
    return SwarmPack(GetPack<MBD, T>(pmd, desc));
  }

  template <class MBD, class T>
  //static SparsePackBase<1> GetPack(MBD *pmd, const impl::SwarmPackDescriptor &desc) {
  static SwarmPackBase<> GetPack(MBD *pmd, const impl::SwarmPackDescriptor &desc) {
    printf("%s:%i\n", __FILE__, __LINE__);
    return Get<MBD, T>(pmd, desc);
  }

  template <class MBD, class T>
  static SwarmPackBase<> Get(MBD *pmd, const impl::SwarmPackDescriptor &desc) {
    printf("%s:%i\n", __FILE__, __LINE__);
    std::string ident = GetIdentifier(desc);
    auto &pack_map = pmd->GetSwarmPackCache().pack_map;
    if (pack_map.count(ident) > 0) {
      auto &pack = pack_map[ident].first;
      // Cached version is not stale, so just return a reference to it
      return pack_map[ident].first;
    }
    return BuildAndAdd(pmd, desc, ident);
  }

  template <class T>
  static SwarmPackBase<> BuildAndAdd(T *pmd, const SwarmPackDescriptor &desc,
                                    const std::string &ident) {
    printf("%s:%i\n", __FILE__, __LINE__);
    auto &pack_map = pmd->GetSwarmPackCache().pack_map;
    printf("%s:%i\n", __FILE__, __LINE__);
    //pack_map[ident] = {Build(pmd, desc), GetAllocStatus(pmd, desc)};
    // TODO(BRR) hack
    pack_map[ident] = {Build(pmd, desc), alloc_t()};
    return pack_map[ident].first;
  }

  // Actually build a `SparsePackBase` (i.e. create a view of views, fill on host, and
  // deep copy the view of views to device) from the variables specified in desc contained
  // from the blocks contained in pmd (which can either be MeshBlockData/MeshData).
  template <class T>
  static SparsePackBase<1> Build(T *pmd, const SwarmPackDescriptor &desc) {
    using mbd_t = MeshBlockData<Real>;
    int nvar = desc.vars.size();

    SwarmPackBase<> pack;
    pack.nvar_ = desc.vars.size();

    // Count up the size of the array that is required
    int max_size = 0;
    int nblocks = 0;
    int ndim = 3;
    ForEachBlock(pmd, [&](int b, mbd_t *pmbd) {
      auto swarm = pmbd->GetSwarm(desc.swarm_name);
      int size = 0;
      nblocks++;
      for (auto &pv : swarm->GetParticleVariableVector<Real>()) {
        for (int i = 0; i < nvar; ++i) {
          if (desc.IncludeVariable(i, pv)) {
              size += pv->GetDim(6) * pv->GetDim(5) * pv->GetDim(4) * pv->GetDim(3) * pv->GetDim(2);
              ndim = 1;
          }
        }
      }
      max_size = std::max(size, max_size);
    });
    pack.nblocks_ = nblocks;

    // Allocate the views
    int leading_dim = 1;
    //if (desc.with_fluxes) leading_dim += 3;
    pack.pack_ = pack_t("data_ptr", leading_dim, nblocks, max_size);
    auto pack_h = Kokkos::create_mirror_view(pack.pack_);
    printf("%s:%i\n", __FILE__, __LINE__);

    pack.bounds_ = bounds_t("bounds", 2, nblocks, nvar);
    auto bounds_h = Kokkos::create_mirror_view(pack.bounds_);
    printf("%s:%i\n", __FILE__, __LINE__);

    pack.coords_ = coords_t("coords", nblocks);
    auto coords_h = Kokkos::create_mirror_view(pack.coords_);
    printf("%s:%i\n", __FILE__, __LINE__);

    // Fill the views
    ForEachBlock(pmd, [&](int b, mbd_t *pmbd) {
      int idx = 0;
      coords_h(b) = pmbd->GetBlockPointer()->coords_device;
      auto swarm = pmbd->GetSwarm(desc.swarm_name);

      for (int i = 0; i < nvar; ++i) {
        bounds_h(0, b, i) = idx;

        //for (auto &pv : pmbd->GetCellVariableVector()) {
        for (auto &pv : swarm->GetParticleVariableVector<Real>()) {
          if (desc.IncludeVariable(i, pv)) {
            for (int t = 0; t < pv->GetDim(6); ++t) {
              for (int u = 0; u < pv->GetDim(5); ++u) {
                for (int v = 0; v < pv->GetDim(4); ++v) {
                  for (int l = 0; l < pv->GetDim(3); ++l) {
                    for (int m = 0; m < pv->GetDim(2); ++m) {
                      pack_h(0, b, idx) = pv->data.Get(t, u, v, l, m);
                      PARTHENON_REQUIRE(
                          pack_h(0, b, idx).size() > 0,
                          "Seems like this variable might not actually be allocated.");
                      idx++;
                    }
                  }
                }
              }
            }
          }
        }

        bounds_h(1, b, i) = idx - 1;

        if (bounds_h(1, b, i) < bounds_h(0, b, i)) {
          // Did not find any allocated variables meeting our criteria
          bounds_h(0, b, i) = -1;
          // Make the upper bound more negative so a for loop won't iterate once
          bounds_h(1, b, i) = -2;
        }
      }
    });

//                    if (pack.coarse_) {
//                      pack_h(0, b, idx) = pv->coarse_s.Get(t, u, v);
//                    } else {
//                      pack_h(0, b, idx) = pv->data.Get(t, u, v);
//                    }
//                    PARTHENON_REQUIRE(
//                        pack_h(0, b, idx).size() > 0,
//                        "Seems like this variable might not actually be allocated.");
//                    if (desc.with_fluxes && pv->IsSet(Metadata::WithFluxes)) {
//                      pack_h(1, b, idx) = pv->flux[1].Get(t, u, v);
//                      PARTHENON_REQUIRE(pack_h(1, b, idx).size() ==
//                                            pack_h(0, b, idx).size(),
//                                        "Different size fluxes.");
//                      if (ndim > 1) {
//                        pack_h(2, b, idx) = pv->flux[2].Get(t, u, v);
//                        PARTHENON_REQUIRE(pack_h(2, b, idx).size() ==
//                                              pack_h(0, b, idx).size(),
//                                          "Different size fluxes.");
//                      }
//                      if (ndim > 2) {
//                        pack_h(3, b, idx) = pv->flux[3].Get(t, u, v);
//                        PARTHENON_REQUIRE(pack_h(3, b, idx).size() ==
//                                              pack_h(0, b, idx).size(),
//                                          "Different size fluxes.");
//                      }
//                    }
//                    idx++;
//                  }
//                }
//              }
//            }
//          }
//        }
//
//        bounds_h(1, b, i) = idx - 1;
//
//        if (bounds_h(1, b, i) < bounds_h(0, b, i)) {
//          // Did not find any allocated variables meeting our criteria
//          bounds_h(0, b, i) = -1;
//          // Make the upper bound more negative so a for loop won't iterate once
//          bounds_h(1, b, i) = -2;
//        }
//      }
//    });

    Kokkos::deep_copy(pack.pack_, pack_h);
    Kokkos::deep_copy(pack.bounds_, bounds_h);
    Kokkos::deep_copy(pack.coords_, coords_h);
    pack.ndim_ = ndim;
    pack.dims_[1] = pack.nblocks_;
    pack.dims_[2] = -1; // Not allowed to ask for the ragged dimension anyway
    pack.dims_[3] = pack_h(0, 0, 0).extent_int(0);
    //pack.dims_[4] = pack_h(0, 0, 0).extent_int(2);
    //pack.dims_[5] = pack_h(0, 0, 0).extent_int(3);

    return pack;
  }

  // template <class T>
  static std::string GetIdentifier(const SwarmPackDescriptor &desc) {
    printf("%s:%i\n", __FILE__, __LINE__);
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

  // Get a list of booleans of the allocation status of every variable in pmd matching the
  // PackDescriptor desc
  //template <class T>
  //static SwarmPackBase<>::alloc_t GetAllocStatus(T *pmd, const SwarmPackDescriptor &desc) {
  //  using mbd_t = MeshBlockData<Real>;

  //  int nvar = desc.vars.size();

  //  std::vector<bool> astat;
  //  ForEachBlock(pmd, [&](int b, mbd_t *pmbd) {
  //    auto swarm = pmbd->GetSwarm(desc.swarm_name);
  //    for (int i = 0; i < nvar; ++i) {
  //      //for (auto &pv : pmbd->GetCellVariableVector()) {
  //      for (auto &pv : swarm->GetParticleVariableVector<Real>()) {
  //        if (desc.IncludeVariable(i, pv)) {
  //          astat.push_back(pv->IsAllocated());
  //        }
  //      }
  //    }
  //  });
  //  return astat;
  //}

  // Make a `SwarmPack` with a corresponding `SwarmPackIdxMap` from the provided `vars`
  // and `flags`, creating the pack in `pmd->SparsePackCache` if it doesn't already exist.
  // The pack will be created and accessible on the device
  // VAR_VEC can be:
  //   1) std::vector<std::string> of variable names (in which case they are all assumed
  //   not to be regexs)
  //   2) std::vector<std::pair<std::string, bool>> of (variable name, treat name as
  //   regex) pairs
  template <class MBD, class T, class VAR_VEC>
  static std::tuple<SwarmPack, SparsePackIdxMap>
  Get(MBD *pmd, const std::string &swarm_name, const VAR_VEC &vars) {
    printf("%s:%i\n", __FILE__, __LINE__);
    static_assert(sizeof...(Ts) == 0, "Cannot create a string/type hybrid pack");
    impl::SwarmPackDescriptor desc(swarm_name, vars);
    return {SwarmPack(GetPack<MBD, T>(pmd, desc)), SwarmPackBase<>::GetIdxMap(desc)};
  }

  // Bound overloads
  KOKKOS_INLINE_FUNCTION int GetLowerBound(const int b) const { return bounds_(0, b, 0); }

  KOKKOS_INLINE_FUNCTION int GetUpperBound(const int b) const {
    return bounds_(1, b, nvar_ - 1);
  }

  KOKKOS_INLINE_FUNCTION int GetLowerBound(const int b, PackIdx idx) const {
    static_assert(sizeof...(Ts) == 0, "Cannot create a string/type hybrid pack");
    return bounds_(0, b, idx.VariableIdx());
  }

  KOKKOS_INLINE_FUNCTION int GetUpperBound(const int b, PackIdx idx) const {
    static_assert(sizeof...(Ts) == 0, "Cannot create a string/type hybrid pack");
    return bounds_(1, b, idx.VariableIdx());
  }

  template <class TIn, REQUIRES(IncludesType<TIn, Ts...>::value)>
  KOKKOS_INLINE_FUNCTION int GetLowerBound(const int b, const TIn &) const {
    const int vidx = GetTypeIdx<TIn, Ts...>::value;
    return bounds_(0, b, vidx);
  }

  template <class TIn, REQUIRES(IncludesType<TIn, Ts...>::value)>
  KOKKOS_INLINE_FUNCTION int GetUpperBound(const int b, const TIn &) const {
    const int vidx = GetTypeIdx<TIn, Ts...>::value;
    return bounds_(1, b, vidx);
  }

  // operator() overloads
  KOKKOS_INLINE_FUNCTION
  auto &operator()(const int b, const int idx) const { return pack_(0, b, idx); }

  KOKKOS_INLINE_FUNCTION
  Real &operator()(const int b, const int idx, const int n) const {
    return pack_(0, b, idx)(n);
  }

  KOKKOS_INLINE_FUNCTION
  Real &operator()(const int b, PackIdx idx, const int n) const {
    static_assert(sizeof...(Ts) == 0, "Cannot create a string/type hybrid pack");
    const int nidx = bounds_(0, b, idx.VariableIdx()) + idx.Offset();
    return pack_(0, b, nidx)(n);
  }

  template <class TIn, REQUIRES(IncludesType<TIn, Ts...>::value)>
  KOKKOS_INLINE_FUNCTION Real &operator()(const int b, const TIn &t, const int n) const {
    const int vidx = GetLowerBound(b, t) + t.idx;
    return pack_(0, b, vidx)(n);
  }

  private:
  using swarm_device_context_t = ParArray1D<ParArray0D<SwarmDeviceContext>>;

};

template <class... Ts>
class SparsePack : public SparsePackBase<> {
 public:
  SparsePack() = default;

  explicit SparsePack(const SparsePackBase &spb) : SparsePackBase(spb) {}

  // Make a `SparsePack` from variable_name types in the type list Ts..., creating the
  // pack in `pmd->SparsePackCache` if it doesn't already exist. Variables can be
  // accessed on device via instance of types in the type list Ts...
  // The pack will be created and accessible on the device
  template <class MBD, class T>
  static SparsePack Get(MBD *pmd, const std::vector<MetadataFlag> &flags = {},
                        bool fluxes = false, bool coarse = false) {
    const impl::PackDescriptor desc(std::vector<std::string>{Ts::name()...},
                                    std::vector<bool>{Ts::regex()...}, flags, fluxes,
                                    coarse);
    return SparsePack(GetPack<MBD, T>(pmd, desc));
  }

  // TODO(BRR) merge GetPack and Get?
  // Returns a SparsePackBase object that is either newly created or taken
  // from the cache in pmd.
  template <class MBD, class T>
  static SparsePackBase GetPack(MBD *pmd, const impl::PackDescriptor &desc) {
    return Get<MBD, T>(pmd, desc);
  }

  template <class MBD, class T>
  static SparsePackBase Get(MBD *pmd, const PackDescriptor &desc) {
    std::string ident = GetIdentifier(desc);
    auto &pack_map = pmd->GetSparsePackCache().pack_map;
    if (pack_map.count(ident) > 0) {
      auto &pack = pack_map[ident].first;
      if (desc.with_fluxes != pack.with_fluxes_) return BuildAndAdd(pmd, desc, ident);
      if (desc.coarse != pack.coarse_) return BuildAndAdd(pmd, desc, ident);
      auto alloc_status_in = GetAllocStatus(pmd, desc);
      auto &alloc_status = pack_map[ident].second;
      if (alloc_status.size() != alloc_status_in.size())
        return BuildAndAdd(pmd, desc, ident);
      for (int i = 0; i < alloc_status_in.size(); ++i) {
        if (alloc_status[i] != alloc_status_in[i]) return BuildAndAdd(pmd, desc, ident);
      }
      // Cached version is not stale, so just return a reference to it
      return pack_map[ident].first;
    }
    return BuildAndAdd(pmd, desc, ident);
  }

  template <class T>
  static SparsePackBase BuildAndAdd(T *pmd, const PackDescriptor &desc,
                                    const std::string &ident) {
    auto &pack_map = pmd->GetSparsePackCache().pack_map;
    pack_map[ident] = {Build(pmd, desc), GetAllocStatus(pmd, desc)};
    return pack_map[ident].first;
  }

  // Actually build a `SparsePackBase` (i.e. create a view of views, fill on host, and
  // deep copy the view of views to device) from the variables specified in desc contained
  // from the blocks contained in pmd (which can either be MeshBlockData/MeshData).
  template <class T>
  static SparsePackBase Build(T *pmd, const PackDescriptor &desc) {
    using mbd_t = MeshBlockData<Real>;
    int nvar = desc.vars.size();

    SparsePackBase pack;
    pack.with_fluxes_ = desc.with_fluxes;
    pack.coarse_ = desc.coarse;
    pack.nvar_ = desc.vars.size();

    // Count up the size of the array that is required
    int max_size = 0;
    int nblocks = 0;
    int ndim = 3;
    ForEachBlock(pmd, [&](int b, mbd_t *pmbd) {
      int size = 0;
      nblocks++;
      for (auto &pv : pmbd->GetCellVariableVector()) {
        for (int i = 0; i < nvar; ++i) {
          if (desc.IncludeVariable(i, pv)) {
            if (pv->IsAllocated()) {
              size += pv->GetDim(6) * pv->GetDim(5) * pv->GetDim(4);
              ndim = (pv->GetDim(1) > 1 ? 1 : 0) + (pv->GetDim(2) > 1 ? 1 : 0) +
                     (pv->GetDim(3) > 1 ? 1 : 0);
            }
          }
        }
      }
      max_size = std::max(size, max_size);
    });
    pack.nblocks_ = nblocks;

    // Allocate the views
    int leading_dim = 1;
    if (desc.with_fluxes) leading_dim += 3;
    pack.pack_ = pack_t("data_ptr", leading_dim, nblocks, max_size);
    auto pack_h = Kokkos::create_mirror_view(pack.pack_);

    pack.bounds_ = bounds_t("bounds", 2, nblocks, nvar);
    auto bounds_h = Kokkos::create_mirror_view(pack.bounds_);

    pack.coords_ = coords_t("coords", nblocks);
    auto coords_h = Kokkos::create_mirror_view(pack.coords_);

    // Fill the views
    ForEachBlock(pmd, [&](int b, mbd_t *pmbd) {
      int idx = 0;
      coords_h(b) = pmbd->GetBlockPointer()->coords_device;

      for (int i = 0; i < nvar; ++i) {
        bounds_h(0, b, i) = idx;

        for (auto &pv : pmbd->GetCellVariableVector()) {
          if (desc.IncludeVariable(i, pv)) {
            if (pv->IsAllocated()) {
              for (int t = 0; t < pv->GetDim(6); ++t) {
                for (int u = 0; u < pv->GetDim(5); ++u) {
                  for (int v = 0; v < pv->GetDim(4); ++v) {
                    if (pack.coarse_) {
                      pack_h(0, b, idx) = pv->coarse_s.Get(t, u, v);
                    } else {
                      pack_h(0, b, idx) = pv->data.Get(t, u, v);
                    }
                    PARTHENON_REQUIRE(
                        pack_h(0, b, idx).size() > 0,
                        "Seems like this variable might not actually be allocated.");
                    if (desc.with_fluxes && pv->IsSet(Metadata::WithFluxes)) {
                      pack_h(1, b, idx) = pv->flux[1].Get(t, u, v);
                      PARTHENON_REQUIRE(pack_h(1, b, idx).size() ==
                                            pack_h(0, b, idx).size(),
                                        "Different size fluxes.");
                      if (ndim > 1) {
                        pack_h(2, b, idx) = pv->flux[2].Get(t, u, v);
                        PARTHENON_REQUIRE(pack_h(2, b, idx).size() ==
                                              pack_h(0, b, idx).size(),
                                          "Different size fluxes.");
                      }
                      if (ndim > 2) {
                        pack_h(3, b, idx) = pv->flux[3].Get(t, u, v);
                        PARTHENON_REQUIRE(pack_h(3, b, idx).size() ==
                                              pack_h(0, b, idx).size(),
                                          "Different size fluxes.");
                      }
                    }
                    idx++;
                  }
                }
              }
            }
          }
        }

        bounds_h(1, b, i) = idx - 1;

        if (bounds_h(1, b, i) < bounds_h(0, b, i)) {
          // Did not find any allocated variables meeting our criteria
          bounds_h(0, b, i) = -1;
          // Make the upper bound more negative so a for loop won't iterate once
          bounds_h(1, b, i) = -2;
        }
      }
    });

    Kokkos::deep_copy(pack.pack_, pack_h);
    Kokkos::deep_copy(pack.bounds_, bounds_h);
    Kokkos::deep_copy(pack.coords_, coords_h);
    pack.ndim_ = ndim;
    pack.dims_[1] = pack.nblocks_;
    pack.dims_[2] = -1; // Not allowed to ask for the ragged dimension anyway
    pack.dims_[3] = pack_h(0, 0, 0).extent_int(0);
    pack.dims_[4] = pack_h(0, 0, 0).extent_int(1);
    pack.dims_[5] = pack_h(0, 0, 0).extent_int(2);

    return pack;
  }

  // Get a list of booleans of the allocation status of every variable in pmd matching the
  // PackDescriptor desc
  template <class T>
  static SparsePackBase::alloc_t GetAllocStatus(T *pmd, const PackDescriptor &desc) {
    using mbd_t = MeshBlockData<Real>;

    int nvar = desc.vars.size();

    std::vector<bool> astat;
    ForEachBlock(pmd, [&](int b, mbd_t *pmbd) {
      for (int i = 0; i < nvar; ++i) {
        for (auto &pv : pmbd->GetCellVariableVector()) {
          if (desc.IncludeVariable(i, pv)) {
            astat.push_back(pv->IsAllocated());
          }
        }
      }
    });
    return astat;
  }

  // template <class T>
  // TODO(BRR) duplicated with SparsePackCache?
  static std::string GetIdentifier(const PackDescriptor &desc) {
    std::string identifier("");
    for (const auto &flag : desc.flags)
      identifier += flag.Name();
    identifier += "____";
    for (int i = 0; i < desc.vars.size(); ++i)
      identifier += desc.vars[i] + std::to_string(desc.use_regex[i]);
    return identifier;
  }

  // Make a `SparsePack` with a corresponding `SparsePackIdxMap` from the provided `vars`
  // and `flags`, creating the pack in `pmd->SparsePackCache` if it doesn't already exist.
  // The pack will be created and accessible on the device
  // VAR_VEC can be:
  //   1) std::vector<std::string> of variable names (in which case they are all assumed
  //   not to be regexs)
  //   2) std::vector<std::pair<std::string, bool>> of (variable name, treat name as
  //   regex) pairs
  template <class MBD, class T, class VAR_VEC>
  static std::tuple<SparsePack, SparsePackIdxMap>
  Get(MBD *pmd, const VAR_VEC &vars, const std::vector<MetadataFlag> &flags = {},
      bool fluxes = false, bool coarse = false) {
    static_assert(sizeof...(Ts) == 0, "Cannot create a string/type hybrid pack");
    impl::PackDescriptor desc(vars, flags, fluxes, coarse);
    return {SparsePack(GetPack<MBD, T>(pmd, desc)), SparsePackBase::GetIdxMap(desc)};
  }

  // Some Get helper for functions for more readable code
  template <class T>
  static SparsePack GetWithFluxes(T *pmd, const std::vector<MetadataFlag> &flags = {}) {
    const bool coarse = false;
    const bool fluxes = true;
    return Get(pmd, flags, fluxes, coarse);
  }

  template <class T, class VAR_VEC>
  static std::tuple<SparsePack, SparsePackIdxMap>
  GetWithFluxes(T *pmd, const VAR_VEC &vars,
                const std::vector<MetadataFlag> &flags = {}) {
    const bool coarse = false;
    const bool fluxes = true;
    Get(pmd, vars, flags, fluxes, coarse);
  }

  template <class T>
  static SparsePack GetWithCoarse(T *pmd, const std::vector<MetadataFlag> &flags = {}) {
    const bool coarse = true;
    const bool fluxes = false;
    return Get(pmd, flags, fluxes, coarse);
  }

  template <class T, class VAR_VEC>
  static std::tuple<SparsePack, SparsePackIdxMap>
  GetWithCoarse(T *pmd, const VAR_VEC &vars,
                const std::vector<MetadataFlag> &flags = {}) {
    const bool coarse = true;
    const bool fluxes = false;
    Get(pmd, vars, flags, fluxes, coarse);
  }

  // Methods for getting parts of the shape of the pack
  KOKKOS_FORCEINLINE_FUNCTION
  int GetNBlocks() const { return nblocks_; }

  KOKKOS_FORCEINLINE_FUNCTION
  int GetNDim() const { return ndim_; }

  KOKKOS_INLINE_FUNCTION
  const Coordinates_t &GetCoordinates(const int b) const { return coords_(b)(); }

  // Bound overloads
  KOKKOS_INLINE_FUNCTION int GetLowerBound(const int b) const { return bounds_(0, b, 0); }

  KOKKOS_INLINE_FUNCTION int GetUpperBound(const int b) const {
    return bounds_(1, b, nvar_ - 1);
  }

  KOKKOS_INLINE_FUNCTION int GetLowerBound(const int b, PackIdx idx) const {
    static_assert(sizeof...(Ts) == 0, "Cannot create a string/type hybrid pack");
    return bounds_(0, b, idx.VariableIdx());
  }

  KOKKOS_INLINE_FUNCTION int GetUpperBound(const int b, PackIdx idx) const {
    static_assert(sizeof...(Ts) == 0, "Cannot create a string/type hybrid pack");
    return bounds_(1, b, idx.VariableIdx());
  }

  template <class TIn, REQUIRES(IncludesType<TIn, Ts...>::value)>
  KOKKOS_INLINE_FUNCTION int GetLowerBound(const int b, const TIn &) const {
    const int vidx = GetTypeIdx<TIn, Ts...>::value;
    return bounds_(0, b, vidx);
  }

  template <class TIn, REQUIRES(IncludesType<TIn, Ts...>::value)>
  KOKKOS_INLINE_FUNCTION int GetUpperBound(const int b, const TIn &) const {
    const int vidx = GetTypeIdx<TIn, Ts...>::value;
    return bounds_(1, b, vidx);
  }

  // operator() overloads
  KOKKOS_INLINE_FUNCTION
  auto &operator()(const int b, const int idx) const { return pack_(0, b, idx); }

  KOKKOS_INLINE_FUNCTION
  Real &operator()(const int b, const int idx, const int k, const int j,
                   const int i) const {
    return pack_(0, b, idx)(k, j, i);
  }

  KOKKOS_INLINE_FUNCTION
  Real &operator()(const int b, PackIdx idx, const int k, const int j,
                   const int i) const {
    static_assert(sizeof...(Ts) == 0, "Cannot create a string/type hybrid pack");
    const int n = bounds_(0, b, idx.VariableIdx()) + idx.Offset();
    return pack_(0, b, n)(k, j, i);
  }

  template <class TIn, REQUIRES(IncludesType<TIn, Ts...>::value)>
  KOKKOS_INLINE_FUNCTION Real &operator()(const int b, const TIn &t, const int k,
                                          const int j, const int i) const {
    const int vidx = GetLowerBound(b, t) + t.idx;
    return pack_(0, b, vidx)(k, j, i);
  }

  // flux() overloads
  KOKKOS_INLINE_FUNCTION
  auto &flux(const int b, const int dir, const int idx) const {
    PARTHENON_DEBUG_REQUIRE(dir > 0 && dir < 4 && with_fluxes_, "Bad input to flux call");
    return pack_(dir, b, idx);
  }

  KOKKOS_INLINE_FUNCTION
  Real &flux(const int b, const int dir, const int idx, const int k, const int j,
             const int i) const {
    PARTHENON_DEBUG_REQUIRE(dir > 0 && dir < 4 && with_fluxes_, "Bad input to flux call");
    return pack_(dir, b, idx)(k, j, i);
  }

  KOKKOS_INLINE_FUNCTION
  Real &flux(const int b, const int dir, PackIdx idx, const int k, const int j,
             const int i) const {
    static_assert(sizeof...(Ts) == 0, "Cannot create a string/type hybrid pack");
    PARTHENON_DEBUG_REQUIRE(dir > 0 && dir < 4 && with_fluxes_, "Bad input to flux call");
    const int n = bounds_(0, b, idx.VariableIdx()) + idx.Offset();
    return pack_(dir, b, n)(k, j, i);
  }

  template <class TIn, REQUIRES(IncludesType<TIn, Ts...>::value)>
  KOKKOS_INLINE_FUNCTION Real &flux(const int b, const int dir, const TIn &t, const int k,
                                    const int j, const int i) const {
    PARTHENON_DEBUG_REQUIRE(dir > 0 && dir < 4 && with_fluxes_, "Bad input to flux call");
    const int vidx = GetLowerBound(b, t) + t.idx;
    return pack_(dir, b, vidx)(k, j, i);
  }
};

} // namespace parthenon

#endif // INTERFACE_SPARSE_PACK_HPP_
