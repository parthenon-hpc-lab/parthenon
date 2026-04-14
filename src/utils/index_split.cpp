//========================================================================================
// (C) (or copyright) 2023. Triad National Security, LLC. All rights reserved.
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

#include <Kokkos_Core.hpp>

#include "utils/index_split.hpp"

#include "basic_types.hpp"
#include "defs.hpp"
#include "globals.hpp"
#include "interface/mesh_data.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/domain.hpp"
#include "mesh/mesh.hpp"

namespace parthenon {

struct DummyFunctor {
  DummyFunctor() = default;
  KOKKOS_INLINE_FUNCTION
  void operator()(team_mbr_t team_member) const {}
};

IndexSplit::IndexSplit(MeshData<Real> *md, const IndexRange &kb, const IndexRange &jb,
                       const IndexRange &ib, const int k_tiles, const int j_tiles)
    : nghost_(Globals::nghost), k_tiles_(k_tiles), j_tiles_(j_tiles), kbs_(kb.s), jbs_(jb.s), ibs_(ib.s),
      ibe_(ib.e) {
  // k_tiles_ and j_tiles_ define how the kj space is tiled into (k_tiles_ x j_tiles_) tiles. The k- and j- 
  // bounds of each of the tiles are returned by `GetBoundsK` and `GetBoundsJ`. 
  // The loop structure is:
  //   - Outermost loop over tiles
  //   - Middle loop over k range of the tile (since that can't be pulled into the inner contiguous memory loop)
  //   - Inner contiguous memory loop over i-range and j-range of tile, including ghosts where necessary 

  Init(md, kb.e, jb.e);
  ndim_ = md->GetNDim();
}

IndexSplit::IndexSplit(MeshData<Real> *md, IndexDomain domain, const int k_tiles,
                       const int j_tiles)
    : nghost_(Globals::nghost), k_tiles_(k_tiles), j_tiles_(j_tiles) {
  auto ib = md->GetBoundsI(domain);
  auto jb = md->GetBoundsJ(domain);
  auto kb = md->GetBoundsK(domain);
  kbs_ = kb.s;
  jbs_ = jb.s;
  ibs_ = ib.s;
  ibe_ = ib.e;
  Init(md, kb.e, jb.e);
  ndim_ = md->GetNDim();
}

void IndexSplit::Init(MeshData<Real> *md, const int kbe, const int jbe) {
  // Save the size of the logical domain (i.e. the requested index range)
  logical_nk_ = kbe - kbs_ + 1;
  logical_nj_ = jbe - jbs_ + 1;
  logical_ni_ = ibe_ - ibs_ + 1;
   
  logical_idxer_ = Indexer3D({kbs_, kbe}, {jbs_, jbe}, {ibs_, ibe_});
   
  // save the size of the memory domain of the block we are iterating over
  auto mib = md->GetBoundsI(IndexDomain::entire);
  auto mjb = md->GetBoundsJ(IndexDomain::entire);
  auto mkb = md->GetBoundsK(IndexDomain::entire);
  memory_ni_ = mib.e + 1;
  memory_nj_ = mjb.e + 1;
  memory_nk_ = mkb.e + 1;

  memory_idxer_ = Indexer3D({mkb.s, mkb.e}, {mjb.s, mjb.e}, {mib.s, mib.e});

  // Compute max parallelism (at outer loop level) from Kokkos
  // equivalent to NSMS in Kokkos
  // TODO(JMM): I'm not sure if this is really the best way to do
  // this. Based on discussion on Kokkos slack.
#ifdef PARTHENON_ENABLE_GPU
  const auto space = DevExecSpace();
  team_policy policy(space, (md->NumBlocks()) * logical_nk_, Kokkos::AUTO);
  // JMM: In principle, should pass a realistic functor here. Using a
  // dummy because we don't know what's available.
  // TODO(JMM): Should we expose the functor?
  policy.set_scratch_size(1, Kokkos::PerTeam(sizeof(Real) * logical_ni_ * logical_nj_));
  const int nteams =
      policy.team_size_recommended(DummyFunctor(), Kokkos::ParallelForTag());
  concurrency_ = space.concurrency() / nteams;
#else
  concurrency_ = 1;
#endif // PARTHENON_ENABLE_GPU

  if (k_tiles_ == all_outer)
    k_tiles_ = logical_nk_;
  else if (k_tiles_ == no_outer)
    k_tiles_ = 1;
  if (j_tiles_ == all_outer)
    j_tiles_ = logical_nj_;
  else if (j_tiles_ == no_outer)
    j_tiles_ = 1;

  if (k_tiles_ == 0) {
#ifdef PARTHENON_ENABLE_GPU
    k_tiles_ = logical_nk_;
#else
    k_tiles_ = 1;
#endif // PARTHENON_ENABLE_GPU
  } else if (k_tiles_ > logical_nk_) {
    k_tiles_ = logical_nk_;
  }
  if (j_tiles_ == 0) {
#ifdef PARTHENON_ENABLE_GPU
    // From Forrest Glines:
    // k_tiles_ * j_tiles_ >= number of SMs / number of streams
    // => j_tiles_ >= SMS / streams / k_tiles
    j_tiles_ = std::min(concurrency_ / (NSTREAMS_ * k_tiles_), logical_nj_);
#else
    j_tiles_ = 1;
#endif // PARTHENON_ENABLE_GPU
  } else if (j_tiles_ > logical_nj_) {
    j_tiles_ = logical_nj_;
  }
}

} // namespace parthenon
