//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2022 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2022. Triad National Security, LLC. All rights reserved.
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
#ifndef BVALS_CC_BVALS_UTILS_HPP_
#define BVALS_CC_BVALS_UTILS_HPP_

#include <algorithm>
#include <memory>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "bvals/cc/bnd_info.hpp"
#include "bvals/cc/bvals_cc_in_one.hpp"
#include "interface/variable.hpp"
#include "mesh/domain.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "utils/error_checking.hpp"
#include "utils/loop_utils.hpp"

namespace parthenon {
namespace cell_centered_bvars {

inline std::tuple<int, int, std::string, int>
SendKey(const std::shared_ptr<MeshBlock> &pmb, const NeighborBlock &nb,
        const std::shared_ptr<Variable<Real>> &pcv) {
  const int sender_id = pmb->gid;
  const int receiver_id = nb.snb.gid;
  const int location_idx = (1 + nb.ni.ox1) + 3 * (1 + nb.ni.ox2 + 3 * (1 + nb.ni.ox3));
  return {sender_id, receiver_id, pcv->label(), location_idx};
}

inline std::tuple<int, int, std::string, int>
ReceiveKey(const std::shared_ptr<MeshBlock> &pmb, const NeighborBlock &nb,
           const std::shared_ptr<Variable<Real>> &pcv) {
  const int receiver_id = pmb->gid;
  const int sender_id = nb.snb.gid;
  const int location_idx = (1 - nb.ni.ox1) + 3 * (1 - nb.ni.ox2 + 3 * (1 - nb.ni.ox3));
  return {sender_id, receiver_id, pcv->label(), location_idx};
}

// Build a vector of pointers to all of the sending or receiving communication buffers on
// MeshData md. This cache is important for performance, since this elides a map look up
// for the buffer every time the bvals code iterates over boundaries.
//
// The buffers in the cache do not necessarily need to be in the same order as the
// sequential order of the ForEachBoundary iteration. Therefore, this also builds a vector
// for indexing from the sequential boundary index defined by the iteration pattern of
// ForEachBoundary to the index of the buffer corresponding to this boundary in the buffer
// cache. This allows for reordering the calls to send and receive on the buffers, so that
// MPI_Isends and MPI_Irecvs get posted in the same order (approximately, due to the
// possibility of multiple MeshData per rank) on the sending and receiving ranks. In
// simple tests, this did not have a big impact on performance but I think it is useful to
// leave the machinery here since it doesn't seem to have a big overhead associated with
// it (LFR).
template <BoundaryType bound_type, class COMM_MAP, class F>
void InitializeBufferCache(std::shared_ptr<MeshData<Real>> &md, COMM_MAP *comm_map,
                           BvarsSubCache_t *pcache, F KeyFunc, bool initialize_flags) {
  using namespace loops;
  using namespace loops::shorthands;
  Mesh *pmesh = md->GetMeshPointer();

  using key_t = std::tuple<int, int, std::string, int>;
  std::vector<std::tuple<int, int, key_t>> key_order;

  int boundary_idx = 0;
  ForEachBoundary<bound_type>(md, [&](sp_mb_t pmb, sp_mbd_t rc, nb_t &nb, const sp_cv_t v,
                                      const OffsetIndices &) {
    // TODO(LFR): Remove temporary check that variables with FillGhost and/or WithFluxes 
    // are cell centered, since communication only is currently implemented for those types
    // of variables
    PARTHENON_REQUIRE(v->IsSet(Metadata::Cell), "Boundary communication only implemented for cell variables.");
    
    auto key = KeyFunc(pmb, nb, v);
    PARTHENON_DEBUG_REQUIRE(comm_map->count(key) > 0,
                            "Boundary communicator does not exist");
    // Create a unique index by combining receiver gid (second element of the key
    // tuple) and geometric element index (fourth element of the key tuple)
    int recvr_idx = 27 * std::get<1>(key) + std::get<3>(key);
    key_order.push_back({recvr_idx, boundary_idx, key});
    ++boundary_idx;
  });

  // If desired, sort the keys and boundary indices by receiver_idx
  // std::sort(key_order.begin(), key_order.end(),
  //          [](auto a, auto b) { return std::get<0>(a) < std::get<0>(b); });

  // Or, what the hell, you could put them in random order if you want, which
  // frighteningly seems to run faster in some cases
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(key_order.begin(), key_order.end(), g);

  int buff_idx = 0;
  pcache->buf_vec.clear();
  pcache->idx_vec = std::vector<std::size_t>(key_order.size());
  std::for_each(std::begin(key_order), std::end(key_order), [&](auto &t) {
    pcache->buf_vec.push_back(&((*comm_map)[std::get<2>(t)]));
    (pcache->idx_vec)[std::get<1>(t)] = buff_idx++;
  });

  const int nbound = pcache->buf_vec.size();
  if (initialize_flags && nbound > 0) {
    if (nbound != pcache->sending_non_zero_flags.size()) {
      pcache->sending_non_zero_flags = ParArray1D<bool>("sending_nonzero_flags", nbound);
      pcache->sending_non_zero_flags_h =
          Kokkos::create_mirror_view(pcache->sending_non_zero_flags);
    }
  }
}

template <BoundaryType BOUND_TYPE, bool SENDER>
inline auto CheckSendBufferCacheForRebuild(std::shared_ptr<MeshData<Real>> md) {
  using namespace loops;
  using namespace loops::shorthands;
  BvarsSubCache_t &cache = md->GetBvarsCache().GetSubCache(BOUND_TYPE, SENDER);

  bool rebuild = false;
  bool other_communication_unfinished = false;
  int nbound = 0;
  ForEachBoundary<BOUND_TYPE>(md, [&](sp_mb_t pmb, sp_mbd_t rc, nb_t &nb, const sp_cv_t v,
                                      const OffsetIndices &) {
    const std::size_t ibuf = cache.idx_vec[nbound];
    auto &buf = *(cache.buf_vec[ibuf]);

    if (!buf.IsAvailableForWrite()) other_communication_unfinished = true;

    if (v->IsAllocated()) {
      buf.Allocate();
    } else {
      buf.Free();
    }

    if (ibuf < cache.bnd_info_h.size()) {
      rebuild = rebuild || !UsingSameResource(cache.bnd_info_h(ibuf).buf, buf.buffer());
    } else {
      rebuild = true;
    }
    ++nbound;
  });
  return std::make_tuple(rebuild, nbound, other_communication_unfinished);
}

template <BoundaryType BOUND_TYPE, bool SENDER>
inline auto CheckReceiveBufferCacheForRebuild(std::shared_ptr<MeshData<Real>> md) {
  using namespace loops;
  using namespace loops::shorthands;
  BvarsSubCache_t &cache = md->GetBvarsCache().GetSubCache(BOUND_TYPE, SENDER);

  bool rebuild = false;
  int nbound = 0;

  ForEachBoundary<BOUND_TYPE>(md, [&](sp_mb_t pmb, sp_mbd_t rc, nb_t &nb, const sp_cv_t v,
                                      const OffsetIndices &) {
    const std::size_t ibuf = cache.idx_vec[nbound];
    auto &buf = *cache.buf_vec[ibuf];
    if (ibuf < cache.bnd_info_h.size()) {
      rebuild = rebuild || !UsingSameResource(cache.bnd_info_h(ibuf).buf, buf.buffer());
      if ((buf.GetState() == BufferState::received) &&
          !cache.bnd_info_h(ibuf).allocated) {
        rebuild = true;
      }
      if ((buf.GetState() == BufferState::received_null) &&
          cache.bnd_info_h(ibuf).allocated) {
        rebuild = true;
      }
    } else {
      rebuild = true;
    }
    ++nbound;
  });
  return std::make_tuple(rebuild, nbound);
}

template <BoundaryType BOUND_TYPE, bool SENDER>
inline auto CheckNoCommCacheForRebuild(std::shared_ptr<MeshData<Real>> md) {
  using namespace loops;
  using namespace loops::shorthands;
  BvarsSubCache_t &cache = md->GetBvarsCache().GetSubCache(BOUND_TYPE, SENDER);

  bool rebuild = false;
  int nbound = 0;
  ForEachBoundary<BOUND_TYPE>(md, [&](sp_mb_t pmb, sp_mbd_t rc, nb_t &nb, const sp_cv_t v,
                                      const OffsetIndices &) {
    if (nbound < cache.idx_vec.size()) {
      const std::size_t ibuf = cache.idx_vec[nbound];
      if (ibuf < cache.bnd_info_h.size()) {
        if (cache.bnd_info_h(ibuf).allocated != (v->IsAllocated())) {
          rebuild = true;
        }
      } else {
        rebuild = true;
      }
    } else {
      rebuild = true;
    }
    ++nbound;
  });
  rebuild = rebuild || (nbound == 0) || (nbound != cache.idx_vec.size());
  return std::make_tuple(rebuild, nbound);
}

using F_BND_INFO = std::function<BndInfo(
    std::shared_ptr<MeshBlock> pmb, const NeighborBlock &nb,
    std::shared_ptr<Variable<Real>> v, CommBuffer<buf_pool_t<Real>::owner_t> *buf,
    const OffsetIndices &no)>;

template <BoundaryType BOUND_TYPE, bool SENDER>
inline void RebuildBufferCache(std::shared_ptr<MeshData<Real>> md, int nbound,
                               F_BND_INFO BndInfoCreator) {
  using namespace loops;
  using namespace loops::shorthands;
  BvarsSubCache_t &cache = md->GetBvarsCache().GetSubCache(BOUND_TYPE, SENDER);
  cache.bnd_info = BufferCache_t("bnd_info", nbound);
  cache.bnd_info_h = Kokkos::create_mirror_view(cache.bnd_info);

  // prolongation/restriction sub-sets
  // TODO(JMM): Right now I exclude fluxcorrection boundaries but if
  // we eventually had custom fluxcorrection ops, you could remove
  // this.
  Mesh *pmesh = md->GetParentPointer();
  StateDescriptor *pkg = (pmesh->resolved_packages).get();
  if constexpr (!((BOUND_TYPE == BoundaryType::flxcor_send) ||
                  (BOUND_TYPE == BoundaryType::flxcor_recv))) {
    int nref_funcs = pkg->NumRefinementFuncs();
    // Note that assignment of Kokkos views resets them, but
    // buffer_subset_sizes is a std::vector. It must be cleared, then
    // re-filled.
    cache.buffer_subset_sizes.clear();
    cache.buffer_subset_sizes.resize(nref_funcs, 0);
    cache.buffer_subsets = ParArray2D<std::size_t>("buffer_subsets", nref_funcs, nbound);
    cache.buffer_subsets_h = Kokkos::create_mirror_view(cache.buffer_subsets);
  }

  int ibound = 0;
  ForEachBoundary<BOUND_TYPE>(md, [&](sp_mb_t pmb, sp_mbd_t rc, nb_t &nb, const sp_cv_t v,
                                      const OffsetIndices &no) {
    
    // TODO(LFR): Remove temporary check that variables with FillGhost and/or WithFluxes 
    // are cell centered, since communication only is currently implemented for those types
    // of variables
    PARTHENON_REQUIRE(v->IsSet(Metadata::Cell), "Boundary communication only implemented for cell variables.");
    
    // bnd_info
    const std::size_t ibuf = cache.idx_vec[ibound];
    cache.bnd_info_h(ibuf) = BndInfoCreator(pmb, nb, v, cache.buf_vec[ibuf], no);

    // subsets ordering is same as in cache.bnd_info
    // RefinementFunctions_t owns all relevant functionality, so
    // only one ParArray2D needed.
    if constexpr (!((BOUND_TYPE == BoundaryType::flxcor_send) ||
                    (BOUND_TYPE == BoundaryType::flxcor_recv))) {
      // var must be registered for refinement and this must be a coarse-fine boundary
      // note this condition means that each subset contains
      // both prolongation and restriction conditions. The
      // `RefinementOp_t` in `BndInfo` is assumed to
      // differentiate.
      if (v->IsRefined() && (nb.snb.level != pmb->loc.level)) {
        std::size_t rfid = pkg->RefinementFuncID((v->GetRefinementFunctions()));
        cache.buffer_subsets_h(rfid, cache.buffer_subset_sizes[rfid]++) = ibuf;
      }
    }

    ++ibound;
  });
  Kokkos::deep_copy(cache.bnd_info, cache.bnd_info_h);
  if constexpr (!((BOUND_TYPE == BoundaryType::flxcor_send) ||
                  (BOUND_TYPE == BoundaryType::flxcor_recv))) {
    Kokkos::deep_copy(cache.buffer_subsets, cache.buffer_subsets_h);
  }
}

} // namespace cell_centered_bvars
} // namespace parthenon

#endif // BVALS_CC_BVALS_UTILS_HPP_
