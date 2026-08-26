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
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// Parthenon Includes
#include <coordinates/coordinates.hpp>
#include <kokkos_abstraction.hpp>
#include <mesh/domain.hpp>
#include <parthenon/package.hpp>

// Local Includes
#include "boundary_exchange.hpp"

using namespace parthenon::package::prelude;
using parthenon::IndexShape;

namespace boundary_exchange {

TaskStatus SetBlockValues(MeshData<Real> *md) {
  auto pmesh = md->GetMeshPointer();
  auto desc =
      parthenon::MakePackDescriptor<neighbor_info>(pmesh->resolved_packages.get());
  auto pack = desc.GetPack(md);
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
        });
  }
  return TaskStatus::complete;
}

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto package = std::make_shared<StateDescriptor>("boundary_exchange");
  Params &params = package->AllParams();

  Metadata m({Metadata::Cell, Metadata::Independent, Metadata::FillGhost},
             std::vector<int>{8});
  m.RegisterRefinementOps<parthenon::refinement_ops::ProlongatePiecewiseConstant,
                          parthenon::refinement_ops::RestrictAverage>();
  package->AddField(neighbor_info::name(), m);

  return package;
}

} // namespace boundary_exchange
