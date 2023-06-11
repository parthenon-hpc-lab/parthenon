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

#include <algorithm>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <regex>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "coordinates/coordinates.hpp"
#include "interface/mesh_data.hpp"
#include "interface/meshblock_data.hpp"
#include "interface/sparse_pack_base.hpp"
#include "interface/state_descriptor.hpp"
#include "interface/variable.hpp"
#include "utils/utils.hpp"
namespace parthenon {
namespace impl {
PackDescriptor::PackDescriptor(StateDescriptor *psd, const std::vector<std::string> &vars,
                               const std::vector<bool> &use_regex,
                               const std::vector<MetadataFlag> &flags, bool with_fluxes,
                               bool coarse, bool flat)
    : vars(vars), use_regex(use_regex), flags(flags), with_fluxes(with_fluxes),
      coarse(coarse), flat(flat) {
  PARTHENON_REQUIRE(use_regex.size() == vars.size(),
                    "Must have a regex flag for each variable.");
  PARTHENON_REQUIRE(!(with_fluxes && coarse),
                    "Probably shouldn't be making a coarse pack with fine fluxes.");
  for (const auto &var : vars)
    regexes.push_back(std::regex(var));
  BuildUids(psd);
}

PackDescriptor::PackDescriptor(StateDescriptor *psd,
                               const std::vector<std::pair<std::string, bool>> &vars_in,
                               const std::vector<MetadataFlag> &flags, bool with_fluxes,
                               bool coarse, bool flat)
    : flags(flags), with_fluxes(with_fluxes), coarse(coarse), flat(flat) {
  for (auto var : vars_in) {
    vars.push_back(var.first);
    use_regex.push_back(var.second);
  }
  PARTHENON_REQUIRE(!(with_fluxes && coarse),
                    "Probably shouldn't be making a coarse pack with fine fluxes.");
  for (const auto &var : vars)
    regexes.push_back(std::regex(var));
  BuildUids(psd);
}

PackDescriptor::PackDescriptor(StateDescriptor *psd,
                               const std::vector<std::string> &vars_in,
                               const std::vector<MetadataFlag> &flags, bool with_fluxes,
                               bool coarse, bool flat)
    : vars(vars_in), use_regex(vars_in.size(), false), flags(flags),
      with_fluxes(with_fluxes), coarse(coarse), flat(flat) {
  PARTHENON_REQUIRE(!(with_fluxes && coarse),
                    "Probably shouldn't be making a coarse pack with fine fluxes.");
  for (const auto &var : vars)
    regexes.push_back(std::regex(var));
  BuildUids(psd);
}

void PackDescriptor::BuildUids(const StateDescriptor *const psd) {
  auto fields = psd->AllFields();
  uids = std::vector<std::vector<Uid_t>>(vars.size());
  for (auto [id, md] : fields) {
    for (int i = 0; i < vars.size(); ++i) {
      if (IncludeVariable(i, id, md)) {
        uids[i].push_back(Variable<Real>::GetUniqueID(id.label()));
      }
    }
  }
}

// Method for determining if variable pv should be included in pack for this
// PackDescriptor
bool PackDescriptor::IncludeVariable(int vidx,
                                     const std::shared_ptr<Variable<Real>> &pv) const {
  return IncludeVariable(vidx, pv->GetVarID(), pv->metadata());
}

bool PackDescriptor::IncludeVariable(int vidx, const VarID &id,
                                     const Metadata &md) const {
  // TODO(LFR): Check that the shapes agree
  if (flags.size() > 0) {
    for (const auto &flag : flags) {
      if (!md.IsSet(flag)) return false;
    }
  }

  if (use_regex[vidx]) {
    if (std::regex_match(std::string(id.label()), regexes[vidx])) return true;
  } else {
    if (vars[vidx] == id.label()) return true;
    if (vars[vidx] == id.base_name && id.sparse_id != InvalidSparseID) return true;
  }
  return false;
}
} // namespace impl
} // namespace parthenon

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

using namespace impl;

template <class T>
SparsePackBase::alloc_t SparsePackBase::GetAllocStatus(T *pmd,
                                                       const PackDescriptor &desc) {
  using mbd_t = MeshBlockData<Real>;

  int nvar = desc.vars.size();

  std::vector<int> astat;
  ForEachBlock(pmd, [&](int b, mbd_t *pmbd) {
    auto &uid_map = pmbd->GetUidMap();
    for (int i = 0; i < nvar; ++i) {
      for (auto id : desc.uids[i]) {
        if (uid_map.count(id) > 0) {
          auto pv = uid_map[id];
          astat.push_back(pv->GetAllocationStatus());
        }
      }
    }
  });
  return astat;
}

// Specialize for the only two types this should work for
template SparsePackBase::alloc_t
SparsePackBase::GetAllocStatus<MeshBlockData<Real>>(MeshBlockData<Real> *,
                                                    const PackDescriptor &);
template SparsePackBase::alloc_t
SparsePackBase::GetAllocStatus<MeshData<Real>>(MeshData<Real> *, const PackDescriptor &);

template <class T>
SparsePackBase SparsePackBase::Build(T *pmd, const PackDescriptor &desc) {
  using mbd_t = MeshBlockData<Real>;
  int nvar = desc.vars.size();

  SparsePackBase pack;
  pack.with_fluxes_ = desc.with_fluxes;
  pack.coarse_ = desc.coarse;
  pack.nvar_ = desc.vars.size();
  pack.flat_ = desc.flat;
  pack.size_ = 0;

  // Count up the size of the array that is required
  int max_size = 0;
  int nblocks = 0;
  bool contains_face_or_edge = false;
  int size = 0; // local var used to compute size/block
  ForEachBlock(pmd, [&](int b, mbd_t *pmbd) {
    if (!desc.flat) {
      size = 0;
    }
    nblocks++;
    auto &uid_map = pmbd->GetUidMap();
    for (int i = 0; i < nvar; ++i) {
      for (auto id : desc.uids[i]) {
        if (uid_map.count(id) > 0) {
          auto pv = uid_map[id];
          if (pv->IsAllocated()) {
            if (pv->IsSet(Metadata::Face) || pv->IsSet(Metadata::Edge))
              contains_face_or_edge = true;
            int prod = pv->GetDim(6) * pv->GetDim(5) * pv->GetDim(4);
            size += prod;       // max size/block (or total size for flat)
            pack.size_ += prod; // total ragged size
          }
        }
      }
    }

    max_size = std::max(size, max_size);
  });
  pack.nblocks_ = desc.flat ? 1 : nblocks;

  // Allocate the views
  int leading_dim = 1;
  if (desc.with_fluxes) {
    leading_dim += 3;
  } else if (contains_face_or_edge) {
    leading_dim += 2;
  }
  pack.pack_ = pack_t("data_ptr", leading_dim, pack.nblocks_, max_size);
  auto pack_h = Kokkos::create_mirror_view(pack.pack_);

  // For non-flat packs, shape of pack is type x block x var x k x j x i
  // where type here might be a flux.
  // For flat packs, shape is type x (some var on some block)  x k x j x 1
  // in the latter case, coords indexes into the some var on some
  // block. Bounds provides the start and end index of a var in a block in the flat array.
  // Size is nvar + 1 to store the maximum idx for easy access
  pack.bounds_ = bounds_t("bounds", 2, nblocks, nvar + 1);
  pack.bounds_h_ = Kokkos::create_mirror_view(pack.bounds_);

  pack.coords_ = coords_t("coords", desc.flat ? max_size : nblocks);
  auto coords_h = Kokkos::create_mirror_view(pack.coords_);

  // Fill the views
  int idx = 0;
  ForEachBlock(pmd, [&](int block, mbd_t *pmbd) {
    int b = 0;
    auto &uid_map = pmbd->GetUidMap();
    if (!desc.flat) {
      idx = 0;
      b = block;
      // JMM: This line could be unified with the coords_h line below,
      // but it would imply unnecessary copies in the case of non-flat
      // packs.
      coords_h(b) = pmbd->GetBlockPointer()->coords_device;
    }

    for (int i = 0; i < nvar; ++i) {
      pack.bounds_h_(0, block, i) = idx;
      for (auto id : desc.uids[i]) {
        if (uid_map.count(id) > 0) {
          auto pv = uid_map[id];
          if (pv->IsAllocated()) {
            for (int t = 0; t < pv->GetDim(6); ++t) {
              for (int u = 0; u < pv->GetDim(5); ++u) {
                for (int v = 0; v < pv->GetDim(4); ++v) {
                  if (pv->IsSet(Metadata::Face) || pv->IsSet(Metadata::Edge)) {
                    if (pack.coarse_) {
                      pack_h(0, b, idx) = pv->coarse_s.Get(0, t, u, v);
                      pack_h(1, b, idx) = pv->coarse_s.Get(1, t, u, v);
                      pack_h(2, b, idx) = pv->coarse_s.Get(2, t, u, v);
                    } else {
                      pack_h(0, b, idx) = pv->data.Get(0, t, u, v);
                      pack_h(1, b, idx) = pv->data.Get(1, t, u, v);
                      pack_h(2, b, idx) = pv->data.Get(2, t, u, v);
                    }
                  } else { // This is a cell, node, or a variable that doesn't have
                           // topology information
                    if (pack.coarse_) {
                      pack_h(0, b, idx) = pv->coarse_s.Get(0, t, u, v);
                    } else {
                      pack_h(0, b, idx) = pv->data.Get(0, t, u, v);
                    }
                    if (desc.with_fluxes && pv->IsSet(Metadata::WithFluxes)) {
                      pack_h(1, b, idx) = pv->flux[X1DIR].Get(0, t, u, v);
                      pack_h(2, b, idx) = pv->flux[X2DIR].Get(0, t, u, v);
                      pack_h(3, b, idx) = pv->flux[X3DIR].Get(0, t, u, v);
                    }
                  }
                  PARTHENON_REQUIRE(
                      pack_h(0, b, idx).size() > 0,
                      "Seems like this variable might not actually be allocated.");

                  if (desc.flat) {
                    coords_h(idx) = pmbd->GetBlockPointer()->coords_device;
                  }
                  idx++;
                }
              }
            }
          }
        }
      }
      pack.bounds_h_(1, block, i) = idx - 1;
      if (pack.bounds_h_(1, block, i) < pack.bounds_h_(0, block, i)) {
        // Did not find any allocated variables meeting our criteria
        pack.bounds_h_(0, block, i) = -1;
        // Make the upper bound more negative so a for loop won't iterate once
        pack.bounds_h_(1, block, i) = -2;
      }
    }
    // Record the maximum for easy access
    pack.bounds_h_(1, block, nvar) = idx - 1;
  });

  Kokkos::deep_copy(pack.pack_, pack_h);
  Kokkos::deep_copy(pack.bounds_, pack.bounds_h_);
  Kokkos::deep_copy(pack.coords_, coords_h);
  pack.dims_[1] = pack.nblocks_;
  pack.dims_[2] = -1; // Not allowed to ask for the ragged dimension anyway
  pack.dims_[3] = pack_h(0, 0, 0).extent_int(0);
  pack.dims_[4] = pack_h(0, 0, 0).extent_int(2);
  pack.dims_[5] = pack_h(0, 0, 0).extent_int(3);

  return pack;
}

// Specialize for the only two types this should work for
template SparsePackBase
SparsePackBase::Build<MeshBlockData<Real>>(MeshBlockData<Real> *, const PackDescriptor &);
template SparsePackBase SparsePackBase::Build<MeshData<Real>>(MeshData<Real> *,
                                                              const PackDescriptor &);

template <class T>
SparsePackBase &SparsePackCache::Get(T *pmd, const PackDescriptor &desc) {
  std::string ident = GetIdentifier(desc);
  if (pack_map.count(ident) > 0) {
    auto &pack = pack_map[ident].first;
    auto alloc_status_in = SparsePackBase::GetAllocStatus(pmd, desc);
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
template SparsePackBase &SparsePackCache::Get<MeshData<Real>>(MeshData<Real> *,
                                                              const PackDescriptor &);
template SparsePackBase &
SparsePackCache::Get<MeshBlockData<Real>>(MeshBlockData<Real> *, const PackDescriptor &);

template <class T>
SparsePackBase &SparsePackCache::BuildAndAdd(T *pmd, const PackDescriptor &desc,
                                             const std::string &ident) {
  if (pack_map.count(ident) > 0) pack_map.erase(ident);
  pack_map[ident] = {SparsePackBase::Build(pmd, desc),
                     SparsePackBase::GetAllocStatus(pmd, desc)};
  return pack_map[ident].first;
}
template SparsePackBase &
SparsePackCache::BuildAndAdd<MeshData<Real>>(MeshData<Real> *, const PackDescriptor &,
                                             const std::string &);
template SparsePackBase &SparsePackCache::BuildAndAdd<MeshBlockData<Real>>(
    MeshBlockData<Real> *, const PackDescriptor &, const std::string &);

std::string SparsePackCache::GetIdentifier(const PackDescriptor &desc) const {
  std::string identifier("");
  for (const auto &flag : desc.flags)
    identifier += flag.Name();
  identifier += "____";
  for (int i = 0; i < desc.vars.size(); ++i)
    identifier += desc.vars[i] + std::to_string(desc.use_regex[i]);
  identifier += "____";
  identifier += std::to_string(desc.with_fluxes);
  identifier += std::to_string(desc.coarse);
  identifier += std::to_string(desc.flat);
  return identifier;
}

} // namespace parthenon
