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

namespace parthenon {
namespace cell_centered_bvars {

namespace impl {

using sp_mb_t = std::shared_ptr<MeshBlock>;
using sp_mbd_t = std::shared_ptr<MeshBlockData<Real>>;
using sp_cv_t = std::shared_ptr<CellVariable<Real>>;
using nb_t = NeighborBlock;

enum class LoopControl { cont, break_out };

// Methods for wrapping a function that may or may not return a LoopControl
// object. The first is enabled if the function returns a LoopControl and
// just passes the returned object on. The second just calls the function,
// ignores its return, and returns a LoopControl continue. These wrap the
// function calls in the ForEachBoundary loop template to allow for breaking
// out of the loop if desired
template <class F, class... Args>
inline auto func_caller(F func, Args &&... args) -> typename std::enable_if<
    std::is_same<decltype(func(std::declval<Args>()...)), LoopControl>::value,
    LoopControl>::type {
  return func(std::forward<Args>(args)...);
}

template <class F, class... Args>
inline auto func_caller(F func, Args &&... args) -> typename std::enable_if<
    !std::is_same<decltype(func(std::declval<Args>()...)), LoopControl>::value,
    LoopControl>::type {
  func(std::forward<Args>(args)...);
  return LoopControl::cont;
}

// Loop over boundaries (or shared geometric elements) for blocks contained
// in MeshData, calling the passed function func for every boundary. Unifies
// boundary looping that occurs in many places in the boundary communication
// routines and allows for easy selection of a subset of the boundaries based
// on the template parameter BoundaryType. [Really, this probably does not
// need to be a template parameter, it could just be a function argument]
template <BoundaryType bound = BoundaryType::any, class F>
inline void ForEachBoundary(std::shared_ptr<MeshData<Real>> &md, F func) {
  for (int block = 0; block < md->NumBlocks(); ++block) {
    auto &rc = md->GetBlockData(block);
    auto pmb = rc->GetBlockPointer();
    for (auto &v : rc->GetCellVariableVector()) {
      if (v->IsSet(Metadata::FillGhost)) {
        for (int n = 0; n < pmb->pbval->nneighbor; ++n) {
          auto &nb = pmb->pbval->neighbor[n];
          if constexpr (bound == BoundaryType::local) {
            if (nb.snb.rank != Globals::my_rank) continue;
          } else if constexpr (bound == BoundaryType::nonlocal) {
            if (nb.snb.rank == Globals::my_rank) continue;
          } else if constexpr (bound == BoundaryType::flxcor_send) {
            // Check if this boundary requires flux correction
            if (nb.snb.level != pmb->loc.level - 1) continue;
            // No flux correction required unless boundaries share a face
            if (std::abs(nb.ni.ox1) + std::abs(nb.ni.ox2) + std::abs(nb.ni.ox3) != 1)
              continue;
          } else if constexpr (bound == BoundaryType::flxcor_recv) {
            // Check if this boundary requires flux correction
            if (nb.snb.level - 1 != pmb->loc.level) continue;
            // No flux correction required unless boundaries share a face
            if (std::abs(nb.ni.ox1) + std::abs(nb.ni.ox2) + std::abs(nb.ni.ox3) != 1)
              continue;
          } else if constexpr (bound == BoundaryType::restricted) {
	    // Check if restriction is required
	    if (nb.snb.level >= pmb->loc.level) continue;
	  }
	  if constexpr (bound == BoundaryType::restricted) {
	    IndexRange bni, bnj, bnk;
	    ComputeRestrictionBounds(ni, nj, nk, nb, pmb);
	    // This loop is only over {-1, 0, 1}^3 at most
	    for (int nk = bnk.s; nk <= bnk.e; ++nk) {
	      for (int nj = bnj.s; nj <= bnj.e; ++nj) {
		for (int ni = bni.s; ni <= bni.e; ++ni) {
		  int ntype = std::abs(ni) + std::abs(nj) + std::abs(nk);
		  // skip myself or coarse levels; only the same level must be restricted
		  if (nytpe == 0 || pmb->pbval->nblevel[nk + 1][nj + 1][ni + 1] != pmb->loc.level) continue;
		  OffsetIndices offsets(nk, nj, ni);
		  if (func_caller(func, pmb, rc, nb, v, offsets) == LoopControl::break_out) return;
		}
	      }
	    }
	  } else {
	    OffsetIndices junk;
	    if (func_caller(func, pmb, rc, nb, v, junk) == LoopControl::break_out) return;
	  }
        }
      }
    }
  }
}

inline std::tuple<int, int, std::string, int>
SendKey(const std::shared_ptr<MeshBlock> &pmb, const NeighborBlock &nb,
        const std::shared_ptr<CellVariable<Real>> &pcv) {
  const int sender_id = pmb->gid;
  const int receiver_id = nb.snb.gid;
  const int location_idx = (1 + nb.ni.ox1) + 3 * (1 + nb.ni.ox2 + 3 * (1 + nb.ni.ox3));
  return {sender_id, receiver_id, pcv->label(), location_idx};
}

inline std::tuple<int, int, std::string, int>
ReceiveKey(const std::shared_ptr<MeshBlock> &pmb, const NeighborBlock &nb,
           const std::shared_ptr<CellVariable<Real>> &pcv) {
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
  Mesh *pmesh = md->GetMeshPointer();

  using key_t = std::tuple<int, int, std::string, int>;
  std::vector<std::tuple<int, int, key_t>> key_order;

  int boundary_idx = 0;
  ForEachBoundary<bound_type>(
	md, [&](sp_mb_t pmb, sp_mbd_t rc, nb_t &nb, const sp_cv_t v, const OffsetIndices&) {
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
  BvarsSubCache_t &cache = md->GetBvarsCache().GetSubCache(BOUND_TYPE, SENDER);

  bool rebuild = false;
  bool other_communication_unfinished = false;
  int nbound = 0;
  ForEachBoundary<BOUND_TYPE>(md, [&](sp_mb_t pmb, sp_mbd_t rc, nb_t &nb,
                                      const sp_cv_t v, const OffsetIndices&) {
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
  BvarsSubCache_t &cache = md->GetBvarsCache().GetSubCache(BOUND_TYPE, SENDER);

  bool rebuild = false;
  int nbound = 0;

  ForEachBoundary<BOUND_TYPE>(md, [&](sp_mb_t pmb, sp_mbd_t rc, nb_t &nb,
                                      const sp_cv_t v, const OffsetIndices&) {
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
  BvarsSubCache_t &cache = md->GetBvarsCache().GetSubCache(BOUND_TYPE, SENDER);

  bool rebuild = false;
  int nbound = 0;
  ForEachBoundary<BOUND_TYPE>(md, [&](sp_mb_t pmb, sp_mbd_t rc, nb_t &nb,
                                      const sp_cv_t v, const OffsetIndices&) {
    if (!(v->IsAllocated())) continue;
    const std::size_t ibuf = cache.idx_vec[nbound++];
    rebuild = (ibuf < cache.bnd_info_h.size()) ? rebuild : true;
  });
  return std::make_tuple(rebuild, nbound);
}


using F_BND_INFO = std::function<BndInfo(
    std::shared_ptr<MeshBlock> pmb, const NeighborBlock &nb,
    std::shared_ptr<CellVariable<Real>> v, CommBuffer<buf_pool_t<Real>::owner_t> *buf,
    const OffsetIndices &no)>;

template <BoundaryType BOUND_TYPE, bool SENDER>
inline void RebuildBufferCache(std::shared_ptr<MeshData<Real>> md, int nbound,
                               F_BND_INFO BndInfoCreator) {
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
  ForEachBoundary<BOUND_TYPE>(
	md, [&](sp_mb_t pmb, sp_mbd_t rc, nb_t &nb, const sp_cv_t v, const OffsetIndices &no) {
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

} // namespace impl
} // namespace cell_centered_bvars
} // namespace parthenon

#endif // BVALS_CC_BVALS_UTILS_HPP_
