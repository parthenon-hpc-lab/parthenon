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

IndexSplit::IndexSplit(MeshData<Real> *md, IndexDomain domain, const int k_tiles, const int j_tiles, TopologicalElement te, TopologicalElement te_mem)
    : IndexSplit(md, md->GetBoundsK(domain, te), md->GetBoundsJ(domain, te), md->GetBoundsI(domain, te), k_tiles, j_tiles, te_mem) {}

IndexSplit IndexSplit::RawMemIJ(IndexDomain domain, int halo, MeshData<Real> *md, TE logical_te, TE memory_te) { 
  auto pmesh = md->GetMeshPointer();
  const int ndim = pmesh->ndim;
  auto ib = md->GetBoundsI(domain, logical_te);
  auto jb = md->GetBoundsJ(domain, logical_te);
  auto kb = md->GetBoundsK(domain, logical_te); 
  ib.s -= halo;
  ib.e += halo;
  jb.s -= (ndim > 1) * halo;
  jb.e += (ndim > 1) * halo;
  kb.s -= (ndim > 2) * halo;
  kb.e += (ndim > 2) * halo;
  int k_tiles = kb.e - kb.s + 1; // Outer loop iterates over all k
  int j_tiles = 1; // Tile contains all j indices, so inner loops run over all i and j for a fixed k
  return IndexSplit(md, kb, jb, ib, kb.e - kb.s + 1, 1);
}

IndexSplit::IndexSplit(MeshData<Real> *md, const IndexRange &kb, const IndexRange &jb,
                       const IndexRange &ib, const int k_tiles, const int j_tiles, TopologicalElement te_mem)
    : k_tiles_(k_tiles), j_tiles_(j_tiles) {
  // k_tiles_ and j_tiles_ define how the kj space is tiled into (k_tiles_ x j_tiles_) tiles. The k- and j- 
  // bounds of each of the tiles are returned by `GetBoundsK` and `GetBoundsJ`. 
  // The loop structure is:
  //   - Outermost loop over tiles
  //   - Middle loop over k range of the tile (since that can't be pulled into the inner contiguous memory loop)
  //   - Inner contiguous memory loop over i-range and j-range of tile, including ghosts where necessary 

  // Save the size of the logical domain (i.e. the requested index range)
  logical_ = Indexer3D(kb, jb, ib);
   
  // save the size of the memory domain of the block we are iterating over
  using TE = TopologicalElement;
  PARTHENON_REQUIRE(te_mem == TE::CC || te_mem == TE::NN, "All memory layouts either are cell-centered or nodal, even for faces and edges.");
  auto mib = md->GetBoundsI(IndexDomain::entire, te_mem);
  auto mjb = md->GetBoundsJ(IndexDomain::entire, te_mem);
  auto mkb = md->GetBoundsK(IndexDomain::entire, te_mem);

  memory_ = Indexer3D(mkb, mjb, mib);

  // Compute max parallelism (at outer loop level) from Kokkos
  // equivalent to NSMS in Kokkos
  // TODO(JMM): I'm not sure if this is really the best way to do
  // this. Based on discussion on Kokkos slack.
  int concurrency{1};  //  = NSMs = 132 for NVIDIA H100
#ifdef PARTHENON_ENABLE_GPU
  const auto space = DevExecSpace();
  team_policy policy(space, (md->NumBlocks()) * logical_.Extent<KDIM>(), Kokkos::AUTO);
  // JMM: In principle, should pass a realistic functor here. Using a
  // dummy because we don't know what's available.
  // TODO(JMM): Should we expose the functor?
  policy.set_scratch_size(1, Kokkos::PerTeam(sizeof(Real) * logical_.Extent<IDIM>() * logical_.Extent<JDIM>()));
  const int nteams =
      policy.team_size_recommended(DummyFunctor(), Kokkos::ParallelForTag());
  concurrency = space.concurrency() / nteams;
#endif // PARTHENON_ENABLE_GPU

  if (k_tiles_ == all_outer)
    k_tiles_ = logical_.Extent<KDIM>();
  else if (k_tiles_ == no_outer)
    k_tiles_ = 1;
  if (j_tiles_ == all_outer)
    j_tiles_ = logical_.Extent<JDIM>();
  else if (j_tiles_ == no_outer)
    j_tiles_ = 1;

  if (k_tiles_ == 0) {
#ifdef PARTHENON_ENABLE_GPU
    k_tiles_ = logical_.Extent<KDIM>();
#else
    k_tiles_ = 1;
#endif // PARTHENON_ENABLE_GPU
  } else if (k_tiles_ > logical_.Extent<KDIM>()) {
    k_tiles_ = logical_.Extent<KDIM>();
  }
  if (j_tiles_ == 0) {
#ifdef PARTHENON_ENABLE_GPU
    // From Forrest Glines:
    // k_tiles_ * j_tiles_ >= number of SMs / number of streams
    // => j_tiles_ >= SMS / streams / k_tiles
    j_tiles_ = std::min(concurrency / (NSTREAMS_ * k_tiles_), logical_.Extent<JDIM>());
#else
    j_tiles_ = 1;
#endif // PARTHENON_ENABLE_GPU
  } else if (j_tiles_ > logical_.Extent<JDIM>()) {
    j_tiles_ = logical_.Extent<JDIM>();
  }
}

} // namespace parthenon
