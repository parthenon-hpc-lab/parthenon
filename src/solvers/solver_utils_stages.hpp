//========================================================================================
// (C) (or copyright) 2021-2024. Triad National Security, LLC. All rights reserved.
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
#ifndef SOLVERS_SOLVER_UTILS_STAGES_HPP_
#define SOLVERS_SOLVER_UTILS_STAGES_HPP_

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "kokkos_abstraction.hpp"

namespace parthenon {

namespace solvers {

namespace StageUtils {

template <bool only_fine_on_composite = true>
TaskStatus CopyData(const std::vector<std::string> &fields,
                    const std::shared_ptr<MeshData<Real>> &md_in,
                    const std::shared_ptr<MeshData<Real>> &md_out) {
  using TE = parthenon::TopologicalElement;
  TE te = TE::CC;
  IndexRange ib = md_in->GetBoundsI(IndexDomain::entire, te);
  IndexRange jb = md_in->GetBoundsJ(IndexDomain::entire, te);
  IndexRange kb = md_in->GetBoundsK(IndexDomain::entire, te);

  auto desc = parthenon::MakePackDescriptor(md_in.get(), fields);
  auto pack_in = desc.GetPack(md_in.get(), only_fine_on_composite);
  auto pack_out = desc.GetPack(md_out.get(), only_fine_on_composite);
  const int scratch_size = 0;
  const int scratch_level = 0;
  // Warning: This inner loop strategy only works because we are using IndexDomain::entire
  const int npoints_inner = (kb.e - kb.s + 1) * (jb.e - jb.s + 1) * (ib.e - ib.s + 1);
  parthenon::par_for_outer(
      DEFAULT_OUTER_LOOP_PATTERN, "CopyData", DevExecSpace(), scratch_size, scratch_level,
      0, pack_in.GetNBlocks() - 1, KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b) {
        const int nvars =
            pack_in.GetUpperBound(b) - pack_in.GetLowerBound(b) + 1;
        for (int c = 0; c < nvars; ++c) {
          Real *in = &pack_in(b, te, c, kb.s, jb.s, ib.s);
          Real *out = &pack_out(b, te, c, kb.s, jb.s, ib.s);
          parthenon::par_for_inner(DEFAULT_INNER_LOOP_PATTERN, member, 0,
                                   npoints_inner - 1,
                                   [&](const int idx) { out[idx] = in[idx]; });
        }
      });
  return TaskStatus::complete;
}

template <bool only_fine_on_composite = true>
TaskStatus AddFieldsAndStoreInteriorSelect(const std::vector<std::string> &fields,
                                           const std::shared_ptr<MeshData<Real>> &md_a,
                                           const std::shared_ptr<MeshData<Real>> &md_b,
                                           const std::shared_ptr<MeshData<Real>> &md_out,
                                           Real wa = 1.0, Real wb = 1.0,
                                           bool only_interior_blocks = false) {
  using TE = parthenon::TopologicalElement;
  TE te = TE::CC;
  IndexRange ib = md_a->GetBoundsI(IndexDomain::entire, te);
  IndexRange jb = md_a->GetBoundsJ(IndexDomain::entire, te);
  IndexRange kb = md_a->GetBoundsK(IndexDomain::entire, te);

  int nblocks = md_a->NumBlocks();
  std::vector<bool> include_block(nblocks, true);
  if (only_interior_blocks) {
    // The neighbors array will only be set for a block if its a leaf block
    for (int b = 0; b < nblocks; ++b)
      include_block[b] = md_a->GetBlockData(b)->GetBlockPointer()->neighbors.size() == 0;
  }
  
  auto desc = parthenon::MakePackDescriptor(md_a.get(), fields);
  auto pack_a = desc.GetPack(md_a.get(), include_block, only_fine_on_composite);
  auto pack_b = desc.GetPack(md_b.get(), include_block, only_fine_on_composite);
  auto pack_out = desc.GetPack(md_out.get(), include_block, only_fine_on_composite);
  const int scratch_size = 0;
  const int scratch_level = 0;
  // Warning: This inner loop strategy only works because we are using IndexDomain::entire
  const int npoints_inner = (kb.e - kb.s + 1) * (jb.e - jb.s + 1) * (ib.e - ib.s + 1);
  parthenon::par_for_outer(
      DEFAULT_OUTER_LOOP_PATTERN, "AddFieldsAndStore", DevExecSpace(), scratch_size,
      scratch_level, 0, pack_a.GetNBlocks() - 1,
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b) {
        const int nvars = pack_a.GetUpperBound(b) - pack_a.GetLowerBound(b) + 1;
        for (int c = 0; c < nvars; ++c) {
          Real *avar = &pack_a(b, te, c, kb.s, jb.s, ib.s);
          Real *bvar = &pack_b(b, te, c, kb.s, jb.s, ib.s);
          Real *out = &pack_out(b, te, c, kb.s, jb.s, ib.s);
          parthenon::par_for_inner(
              DEFAULT_INNER_LOOP_PATTERN, member, 0, npoints_inner - 1,
              [&](const int idx) { out[idx] = wa * avar[idx] + wb * bvar[idx]; });
        }
      });
  return TaskStatus::complete;
}

template <bool only_fine_on_composite = true>
TaskStatus AddFieldsAndStore(const std::vector<std::string> &fields,
                             const std::shared_ptr<MeshData<Real>> &md_a,
                             const std::shared_ptr<MeshData<Real>> &md_b,
                             const std::shared_ptr<MeshData<Real>> &md_out,
                             Real wa = 1.0, Real wb = 1.0) {
  return AddFieldsAndStoreInteriorSelect<only_fine_on_composite>(
      fields, md_a, md_b, md_out, wa, wb, false); 
}

template <bool only_fine_on_composite = true>
TaskStatus SetToZero(const std::vector<std::string> &fields, const std::shared_ptr<MeshData<Real>> &md) {
  int nblocks = md->NumBlocks();
  using TE = parthenon::TopologicalElement;
  TE te = TE::CC;
  auto desc = parthenon::MakePackDescriptor(md.get(), fields);
  auto pack = desc.GetPack(md.get(), only_fine_on_composite);
  const size_t scratch_size_in_bytes = 0;
  const int scratch_level = 1;
  const int ng = parthenon::Globals::nghost;
  parthenon::par_for_outer(
      DEFAULT_OUTER_LOOP_PATTERN, "SetFieldsToZero", DevExecSpace(),
      scratch_size_in_bytes, scratch_level, 0, pack.GetNBlocks() - 1,
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b) {
        auto cb = GetIndexShape(pack(b, te, 0), ng);
        const auto &coords = pack.GetCoordinates(b);
        IndexRange ib = cb.GetBoundsI(IndexDomain::interior, te);
        IndexRange jb = cb.GetBoundsJ(IndexDomain::interior, te);
        IndexRange kb = cb.GetBoundsK(IndexDomain::interior, te);
        const int nvars = pack.GetUpperBound(b) - pack.GetLowerBound(b) + 1;
        for (int c = 0; c < nvars; ++c) {
          parthenon::par_for_inner(
              parthenon::inner_loop_pattern_simdfor_tag, member, kb.s, kb.e, jb.s, jb.e,
              ib.s, ib.e,
              [&](int k, int j, int i) { pack(b, te, c, k, j, i) = 0.0; });
        }
      });
  return TaskStatus::complete;
}

inline TaskStatus ADividedByB(const std::vector<std::string> &fields,
                       const std::shared_ptr<MeshData<Real>> &md_a,
                       const std::shared_ptr<MeshData<Real>> &md_b,
                       const std::shared_ptr<MeshData<Real>> &md_out) {
  IndexRange ib = md_a->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md_a->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md_a->GetBoundsK(IndexDomain::interior);

  auto desc = parthenon::MakePackDescriptor(md_a.get(), fields);
  auto pack_a = desc.GetPack(md_a.get());
  auto pack_b = desc.GetPack(md_b.get());
  auto pack_out = desc.GetPack(md_out.get());
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "DotProduct", DevExecSpace(), 0, pack_a.GetNBlocks() - 1, kb.s,
      kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const int nvars = pack_a.GetUpperBound(b) - pack_a.GetLowerBound(b) + 1;
        for (int c = 0; c < nvars; ++c)
          pack_out(b, c, k, j, i) =
              pack_a(b, c, k, j, i) / pack_b(b, c, k, j, i);
      });
  return TaskStatus::complete;
}

inline TaskStatus DotProductLocal(const std::vector<std::string> &fields,
                           const std::shared_ptr<MeshData<Real>> &md_a,
                           const std::shared_ptr<MeshData<Real>> &md_b,
                           AllReduce<Real> *adotb) {
  using TE = parthenon::TopologicalElement;
  TE te = TE::CC;
  IndexRange ib = md_a->GetBoundsI(IndexDomain::interior, te);
  IndexRange jb = md_a->GetBoundsJ(IndexDomain::interior, te);
  IndexRange kb = md_a->GetBoundsK(IndexDomain::interior, te);

  auto desc = parthenon::MakePackDescriptor(md_a.get(), fields);
  auto pack_a = desc.GetPack(md_a.get());
  auto pack_b = desc.GetPack(md_b.get());
  Real gsum(0);
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "DotProduct", DevExecSpace(), 0,
      pack_a.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lsum) {
        const int nvars = pack_a.GetUpperBound(b) - pack_a.GetLowerBound(b) + 1;
        // TODO(LFR): If this becomes a bottleneck, exploit hierarchical parallelism and
        //            pull the loop over vars outside of the innermost loop to promote
        //            vectorization.
        for (int c = 0; c < nvars; ++c)
          lsum += pack_a(b, te, c, k, j, i) * pack_b(b, te, c, k, j, i);
      },
      Kokkos::Sum<Real>(gsum));
  adotb->val += gsum;
  return TaskStatus::complete;
}

inline TaskID DotProduct(TaskID dependency_in, TaskList &tl, AllReduce<Real> *adotb,
                  const std::vector<std::string> &fields,
                  const std::shared_ptr<MeshData<Real>> &md_a,
                  const std::shared_ptr<MeshData<Real>> &md_b) {
  using namespace impl;
  auto zero_adotb = tl.AddTask(
      TaskQualifier::once_per_region | TaskQualifier::local_sync, dependency_in,
      [](AllReduce<Real> *r) {
        r->val = 0.0;
        return TaskStatus::complete;
      },
      adotb);
  auto get_adotb = tl.AddTask(TaskQualifier::local_sync, zero_adotb,
                              DotProductLocal, fields, md_a, md_b, adotb);
  auto start_global_adotb = tl.AddTask(TaskQualifier::once_per_region, get_adotb,
                                       &AllReduce<Real>::StartReduce, adotb, MPI_SUM);
  auto finish_global_adotb =
      tl.AddTask(TaskQualifier::once_per_region | TaskQualifier::local_sync,
                 start_global_adotb, &AllReduce<Real>::CheckReduce, adotb);
  return finish_global_adotb;
}

} // namespace utils

} // namespace solvers

} // namespace parthenon

#endif // SOLVERS_SOLVER_UTILS_STAGES_HPP_
