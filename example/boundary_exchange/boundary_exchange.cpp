//========================================================================================
// (C) (or copyright) 2020-2024. Triad National Security, LLC. All rights reserved.
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

// Standard Includes
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// Parthenon Includes
#include <coordinates/coordinates.hpp>
#include <kokkos_abstraction.hpp>
#include <mesh/domain.hpp>
#include <parthenon/package.hpp>

// Local Includes
#include "boundary_exchange.hpp"
#include "bvals/neighbor_block.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"

using namespace parthenon::package::prelude;
using parthenon::IndexShape;

namespace boundary_exchange {

TaskStatus SetBlockValues(MeshData<Real> *md) {
  auto pmesh = md->GetMeshPointer();
  auto desc = parthenon::MakePackDescriptor<neighbor_info, linear_field>(
      pmesh->resolved_packages.get());
  auto pack = desc.GetPack(md);
  const bool set_linear =
      pmesh->packages.Get("boundary_exchange")->Param<bool>("check_faces_only");
  {
    IndexRange ib = md->GetBoundsI(IndexDomain::entire);
    IndexRange jb = md->GetBoundsJ(IndexDomain::entire);
    IndexRange kb = md->GetBoundsK(IndexDomain::entire);
    parthenon::par_for(
        parthenon::loop_pattern_mdrange_tag, "SetNaN", DevExecSpace(), 0,
        pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          for (int n = 0; n < 8; ++n)
            pack(b, neighbor_info(n), k, j, i) = std::numeric_limits<Real>::quiet_NaN();
          if (set_linear)
            pack(b, linear_field(), k, j, i) = std::numeric_limits<Real>::quiet_NaN();
        });
  }

  // Also reset the coarse buffers, so that stale coarse data from a previous exchange
  // cannot hide missing communication
  if (set_linear && pmesh->multilevel && md->NumBlocks() > 0) {
    auto cdesc = parthenon::MakePackDescriptor<neighbor_info, linear_field>(
        pmesh->resolved_packages.get(), {}, {parthenon::PDOpt::Coarse});
    auto cpack = cdesc.GetPack(md);
    auto &cbounds = md->GetBlockData(0)->GetBlockPointer()->c_cellbounds;
    IndexRange ib = cbounds.GetBoundsI(IndexDomain::entire);
    IndexRange jb = cbounds.GetBoundsJ(IndexDomain::entire);
    IndexRange kb = cbounds.GetBoundsK(IndexDomain::entire);
    parthenon::par_for(
        parthenon::loop_pattern_mdrange_tag, "SetCoarseNaN", DevExecSpace(), 0,
        cpack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          for (int n = 0; n < 8; ++n)
            cpack(b, neighbor_info(n), k, j, i) = std::numeric_limits<Real>::quiet_NaN();
          cpack(b, linear_field(), k, j, i) = std::numeric_limits<Real>::quiet_NaN();
        });
  }

  // Get the morton numbers of the blocks in the pack onto device
  parthenon::ParArray1D<int> x("x logical location", pack.GetNBlocks());
  parthenon::ParArray1D<int> y("y logical location", pack.GetNBlocks());
  parthenon::ParArray1D<int> z("z logical location", pack.GetNBlocks());
  parthenon::ParArray1D<int> tree("tree", pack.GetNBlocks());
  parthenon::ParArray1D<int> gid("gid", pack.GetNBlocks());
  auto x_h = Kokkos::create_mirror_view(x);
  auto y_h = Kokkos::create_mirror_view(y);
  auto z_h = Kokkos::create_mirror_view(z);
  auto tree_h = Kokkos::create_mirror_view(tree);
  auto gid_h = Kokkos::create_mirror_view(gid);
  for (int b = 0; b < md->NumBlocks(); ++b) {
    auto cpmb = md->GetBlockData(b)->GetBlockPointer();
    auto level = cpmb->loc.level();
    auto mx = cpmb->loc.lx1() << (2 - level);
    auto my = cpmb->loc.lx2() << (2 - level);
    auto mz = cpmb->loc.lx3() << (2 - level);
    x_h(b) = mx;
    y_h(b) = my;
    z_h(b) = mz;
    tree_h(b) = cpmb->loc.tree();
    gid_h(b) = cpmb->gid;
  }
  Kokkos::deep_copy(x, x_h);
  Kokkos::deep_copy(y, y_h);
  Kokkos::deep_copy(z, z_h);
  Kokkos::deep_copy(tree, tree_h);
  Kokkos::deep_copy(gid, gid_h);

  {
    IndexRange ib = md->GetBoundsI(IndexDomain::interior);
    IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
    IndexRange kb = md->GetBoundsK(IndexDomain::interior);
    parthenon::par_for(
        parthenon::loop_pattern_mdrange_tag, "SetMorton", DevExecSpace(), 0,
        pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          pack(b, neighbor_info(0), k, j, i) = tree(b);
          pack(b, neighbor_info(1), k, j, i) = x(b);
          pack(b, neighbor_info(2), k, j, i) = y(b);
          pack(b, neighbor_info(3), k, j, i) = z(b);
          pack(b, neighbor_info(4), k, j, i) = gid(b);
          pack(b, neighbor_info(5), k, j, i) = i;
          pack(b, neighbor_info(6), k, j, i) = j;
          pack(b, neighbor_info(7), k, j, i) = k;
          if (!set_linear) return;
          // A smooth profile plus a per-block jump, so that prolongation slopes are
          // sometimes limited and sometimes not
          auto &coords = pack.GetCoordinates(b);
          const Real x = coords.Xc<X1DIR>(k, j, i);
          const Real y = coords.Xc<X2DIR>(k, j, i);
          const Real z = coords.Xc<X3DIR>(k, j, i);
          pack(b, linear_field(), k, j, i) =
              x + 2.0 * y * y - 3.0 * z + x * y * z + 0.37 * gid(b);
        });
  }
  return TaskStatus::complete;
}

TaskStatus CopyToReference(MeshData<Real> *md) {
  auto pmesh = md->GetMeshPointer();
  auto desc =
      parthenon::MakePackDescriptor<neighbor_info, linear_field, neighbor_info_ref,
                                    linear_field_ref>(pmesh->resolved_packages.get());
  auto pack = desc.GetPack(md);
  IndexRange ib = md->GetBoundsI(IndexDomain::entire);
  IndexRange jb = md->GetBoundsJ(IndexDomain::entire);
  IndexRange kb = md->GetBoundsK(IndexDomain::entire);
  parthenon::par_for(
      parthenon::loop_pattern_mdrange_tag, "CopyToReference", DevExecSpace(), 0,
      pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        for (int n = 0; n < 8; ++n)
          pack(b, neighbor_info_ref(n), k, j, i) = pack(b, neighbor_info(n), k, j, i);
        pack(b, linear_field_ref(), k, j, i) = pack(b, linear_field(), k, j, i);
      });
  return TaskStatus::complete;
}

namespace {
// Accumulated over all partitions on this rank by CompareFaceGhosts
std::atomic<long> face_ghosts_checked{0};
std::atomic<long> face_ghost_mismatches{0};
std::atomic<long> other_ghosts_filled{0};
std::atomic<long> other_ghosts_total{0};

KOKKOS_INLINE_FUNCTION bool SameValue(const Real a, const Real b) {
  return (a == b) || (a != a && b != b);
}
} // namespace

// Compare the faces-only result with the reference BoundaryType::any result in every
// face ghost region. Edge and corner ghosts are only counted, not checked.
TaskStatus CompareFaceGhosts(MeshData<Real> *md) {
  auto pmesh = md->GetMeshPointer();
  auto desc =
      parthenon::MakePackDescriptor<neighbor_info, linear_field, neighbor_info_ref,
                                    linear_field_ref>(pmesh->resolved_packages.get());
  auto pack = desc.GetPack(md);
  const int ndim = pmesh->ndim;
  IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior);
  IndexRange ibe = md->GetBoundsI(IndexDomain::entire);
  IndexRange jbe = md->GetBoundsJ(IndexDomain::entire);
  IndexRange kbe = md->GetBoundsK(IndexDomain::entire);
  int nchecked{0}, nmismatch{0}, nfilled{0}, nother{0};
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "CompareFaceGhosts", DevExecSpace(), 0,
      pack.GetNBlocks() - 1, kbe.s, kbe.e, jbe.s, jbe.e, ibe.s, ibe.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, int &lchecked,
                    int &lmismatch, int &lfilled, int &lother) {
        const int nout = (i < ib.s || i > ib.e) + (ndim > 1 && (j < jb.s || j > jb.e)) +
                         (ndim > 2 && (k < kb.s || k > kb.e));
        if (nout == 0) return;
        if (nout == 1) {
          bool same = SameValue(pack(b, linear_field(), k, j, i),
                                pack(b, linear_field_ref(), k, j, i));
          for (int n = 0; n < 8; ++n)
            same = same && SameValue(pack(b, neighbor_info(n), k, j, i),
                                     pack(b, neighbor_info_ref(n), k, j, i));
          lchecked += 1;
          lmismatch += !same;
        } else {
          const Real v = pack(b, linear_field(), k, j, i);
          lother += 1;
          lfilled += (v == v);
        }
      },
      Kokkos::Sum<int>(nchecked), Kokkos::Sum<int>(nmismatch), Kokkos::Sum<int>(nfilled),
      Kokkos::Sum<int>(nother));
  face_ghosts_checked += nchecked;
  face_ghost_mismatches += nmismatch;
  other_ghosts_filled += nfilled;
  other_ghosts_total += nother;
  return TaskStatus::complete;
}

int CheckFacesOnlyConsistency(parthenon::Mesh *pmesh) {
  // Every (sender gid, receiver gid, location index) a block sends on under faces-only
  // must be matched by exactly one receive, using the same keys as the CommBuffers
  using key_t = std::tuple<int, int, int>;
  std::vector<int> local_send, local_recv;
  for (auto &pmb : pmesh->block_list) {
    for (auto &nb : pmb->GetNeighbors()) {
      if (nb.faces_only_send) {
        local_send.insert(local_send.end(), {pmb->gid, nb.gid, nb.offsets.GetIdx()});
      }
      if (nb.faces_only_recv) {
        local_recv.insert(
            local_recv.end(),
            {nb.gid, pmb->gid, nb.lcoord_trans.Transform(nb.offsets).GetReverseIdx()});
      }
    }
  }
  auto gather = [](const std::vector<int> &local) {
    std::vector<int> all = local;
#ifdef MPI_PARALLEL
    int nlocal = local.size();
    std::vector<int> counts(parthenon::Globals::nranks),
        displs(parthenon::Globals::nranks);
    MPI_Allgather(&nlocal, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    int tot = 0;
    for (int r = 0; r < parthenon::Globals::nranks; ++r) {
      displs[r] = tot;
      tot += counts[r];
    }
    all.resize(tot);
    MPI_Allgatherv(local.data(), nlocal, MPI_INT, all.data(), counts.data(),
                   displs.data(), MPI_INT, MPI_COMM_WORLD);
#endif
    std::vector<key_t> keys;
    for (std::size_t n = 0; n < all.size(); n += 3)
      keys.emplace_back(all[n], all[n + 1], all[n + 2]);
    std::sort(keys.begin(), keys.end());
    return keys;
  };
  auto send = gather(local_send);
  auto recv = gather(local_recv);
  long nany = 0;
  for (auto &pmb : pmesh->block_list)
    nany += pmb->GetNeighbors().size();
#ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE, &nany, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif
  if (parthenon::Globals::my_rank == 0)
    std::printf("faces_only uses %zu of %li neighbor channels used by any\n", send.size(),
                nany);
  std::vector<key_t> diff;
  std::set_symmetric_difference(send.begin(), send.end(), recv.begin(), recv.end(),
                                std::back_inserter(diff));
  if (parthenon::Globals::my_rank == 0) {
    for (auto &[s, r, l] : diff)
      std::printf("faces_only send/recv mismatch: sender %i receiver %i location %i\n", s,
                  r, l);
  }
  return diff.size();
}

bool ReportFacesOnlyCheck(parthenon::Mesh *pmesh) {
  long vals[4] = {face_ghosts_checked, face_ghost_mismatches, other_ghosts_filled,
                  other_ghosts_total};
#ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE, vals, 4, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif
  if (parthenon::Globals::my_rank == 0) {
    std::printf("faces_only check: %li face ghost cells checked, %li mismatches; %li of "
                "%li edge/corner ghost cells filled\n",
                vals[0], vals[1], vals[2], vals[3]);
    std::printf("faces_only check %s\n", vals[1] == 0 ? "PASSED" : "FAILED");
  }
  return vals[1] == 0;
}

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto package = std::make_shared<StateDescriptor>("boundary_exchange");
  Params &params = package->AllParams();

  Metadata m({Metadata::Cell, Metadata::Independent, Metadata::FillGhost},
             std::vector<int>{8});
  m.RegisterRefinementOps<parthenon::refinement_ops::ProlongatePiecewiseConstant,
                          parthenon::refinement_ops::RestrictAverage>();
  package->AddField(neighbor_info::name(), m);

  const bool check_faces_only =
      pin->GetOrAddBoolean("boundary_exchange", "check_faces_only", false);
  params.Add("check_faces_only", check_faces_only);
  if (check_faces_only) {
    Metadata ml({Metadata::Cell, Metadata::Independent, Metadata::FillGhost});
    ml.RegisterRefinementOps<parthenon::refinement_ops::ProlongateSharedMinMod,
                             parthenon::refinement_ops::RestrictAverage>();
    package->AddField(linear_field::name(), ml);
    package->AddField(neighbor_info_ref::name(),
                      Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy},
                               std::vector<int>{8}));
    package->AddField(linear_field_ref::name(),
                      Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy}));
  }

  return package;
}

} // namespace boundary_exchange
