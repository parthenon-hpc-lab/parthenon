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
  void operator(team_mbr_t team_member) const {}
};

IndexSplit::IndexSplit(MeshData<Real> *md, const IndexRange &kb, const IndexRange &jb,
                       const IndexRange &ib, const int nkp, const int njp)
    : nghost_(Globals::nghost), nkp_(nkp), njp_(njp), kbs_(kb.s), jbs_(jb.s), ibs_(ib.s),
      ibe_(ib.e) {
  Init(md, kb.e, jb.e);
  ndim_ = md->GetNDim();
}

IndexSplit::IndexSplit(MeshData<Real> *md, IndexDomain domain, const int nkp,
                       const int njp)
    : nghost_(Globals::nghost), nkp_(nkp), njp_(njp) {
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
  const int total_k = kbe - kbs_ + 1;
  const int total_j = jbe - jbs_ + 1;
  const int total_i = ibe_ - ibs_ + 1;

  // Compute max parallelism (at outer loop level) from Kokkos
  // equivalent to NSMS in Kokkos
  // TODO(JMM): I'm not sure if this is really the best way to do
  // this. Based on discussion on Kokkos slack.
#ifdef KOKKOS_ENABLE_CUDA
  const auto space = DevExecSpace();
  team_policy policy(space, total_k, Kokkos::AUTO);
  // JMM: In principle, should pass a realistic functor here. Using a
  // dummy because we don't know what's available.
  // TODO(JMM): Should we expose the functor?
  policy.set_scratch_size(1, Kokkos::PerTeam(sizeof(Real) * total_i * total_j));
  const int nteams = policy.team_size_recommended(DummyFunctor(),
                                                  Kokkos::ParallelForTag());
  concurrency_ = space.concurrency() / nteams;
#else
  concurrency_ = 1;
#endif // KOKKOS_ENABLE_CUDA

  if (nkp_ == all_outer)
    nkp_ = total_k;
  else if (nkp_ == no_outer)
    nkp_ = 1;
  if (njp_ == all_outer)
    njp_ = total_j;
  else if (njp_ == no_outer)
    njp_ = 1;

  if (nkp_ == 0) {
#ifdef KOKKOS_ENABLE_CUDA
    nkp_ = total_k;
#else
    nkp_ = 1;
#endif
  } else if (nkp_ > total_k) {
    nkp_ = total_k;
  }
  if (njp_ == 0) {
#ifdef KOKKOS_ENABLE_CUDA
    // From Forrest Glines:
    // nkp_ * njp_ >= number of SMs / number of streams
    njp_ = std::min(nkp_ * NSTREAMS_ / concurrency_, total_j);
#else
    njp_ = 1;
#endif
  } else if (njp_ > total_j) {
    njp_ = total_j;
  }

  // add a tiny bit to avoid round-off issues when we ultimately convert to int
  target_k_ = static_cast<int>((1.0 * total_k) / nkp_ + 1.e-6);
  target_j_ = static_cast<int>((1.0 * total_j) / njp_ + 1.e-6);

  // save the "entire" ranges
  // don't bother save ".s" since it's always zero
  auto ib = md->GetBoundsI(IndexDomain::entire);
  auto jb = md->GetBoundsJ(IndexDomain::entire);
  auto kb = md->GetBoundsK(IndexDomain::entire);
  kbe_entire_ = kb.e;
  jbe_entire_ = jb.e;
  ibe_entire_ = ib.e;
}

} // namespace parthenon
