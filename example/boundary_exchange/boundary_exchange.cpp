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
#include <cstdio>
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

struct ParameterizedLine {
  enum class ltype { straight, arc };

  const Real x1, y1;
  const Real x2, y2;
  Real r, delta, phi;
  ltype type;

  using node = parthenon::forest::Node;
  ParameterizedLine(std::shared_ptr<node> start, std::shared_ptr<node> end)
      : x1{start->x[0]}, y1{start->x[1]}, x2{end->x[0]}, y2{end->x[1]},
        type{ltype::straight} {
    Real d1 = std::sqrt(x1 * x1 + y1 * y1);
    Real d2 = std::sqrt(x2 * x2 + y2 * y2);
    if (std::abs(d1 - d2) < 1.e-8 && d1 > 1.1) {
      type = ltype::arc;
      r = d1;
      delta = M_PI / 4.0;
      phi = 0.0;
      if (y1 > 1.e-5) phi = delta;
    }
  }

  KOKKOS_INLINE_FUNCTION
  Real GetX(Real u) const {
    if (type == ltype::straight)
      return x1 * (1.0 - u) + x2 * u;
    else
      return r * cos(delta * u + phi);
  }

  KOKKOS_INLINE_FUNCTION
  Real GetY(Real u) const {
    if (type == ltype::straight)
      return y1 * (1.0 - u) + y2 * u;
    else
      return r * sin(delta * u + phi);
  }
};

TaskStatus SetCoordinates(MeshData<Real> *md) {
  using TE = parthenon::TopologicalElement;
  auto pmesh = md->GetMeshPointer();
  auto desc = parthenon::MakePackDescriptor<position>(md);

  for (auto &ptree : pmesh->forest.GetTrees()) {
    auto tree_id = ptree->GetId();
    auto pack = desc.GetPack(md, parthenon::GetBlockSelector::OnTree(ptree->GetId()));
    int i{0};
    Real posx[4], posy[4];
    for (auto &pnode : ptree->forest_nodes) {
      posx[i] = pnode->x[0];
      posy[i] = pnode->x[1];
    }

    auto &pnodes = ptree->forest_nodes;
    ParameterizedLine c1(pnodes[0], pnodes[1]);
    ParameterizedLine c3(pnodes[2], pnodes[3]);
    ParameterizedLine c2(pnodes[0], pnodes[2]);
    ParameterizedLine c4(pnodes[1], pnodes[3]);

    IndexRange ib = md->GetBoundsI(IndexDomain::interior, TE::NN);
    IndexRange jb = md->GetBoundsJ(IndexDomain::interior, TE::NN);
    IndexRange kb = md->GetBoundsK(IndexDomain::interior, TE::NN);

    parthenon::par_for(
        parthenon::loop_pattern_mdrange_tag, "SetPosition", DevExecSpace(), 0,
        pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          auto &coords = pack.GetCoordinates(b);
          const Real u = coords.X<X1DIR, TE::NN>(k, j, i);
          const Real v = coords.X<X2DIR, TE::NN>(k, j, i);
          const Real mu = 1.0 - u;
          const Real mv = 1.0 - v;

          Real x = mv * c1.GetX(u) + v * c3.GetX(u) + mu * c2.GetX(v) + u * c4.GetX(v) -
                   (mu * mv * c1.GetX(0.0) + u * mv * c1.GetX(1.0) +
                    mu * v * c2.GetX(1.0) + u * v * c3.GetX(1.0));
          Real y = mv * c1.GetY(u) + v * c3.GetY(u) + mu * c2.GetY(v) + u * c4.GetY(v) -
                   (mu * mv * c1.GetY(0.0) + u * mv * c1.GetY(1.0) +
                    mu * v * c2.GetY(1.0) + u * v * c3.GetY(1.0));
          pack(b, TE::NN, position(0), k, j, i) = x; // x-position
          pack(b, TE::NN, position(1), k, j, i) = y; // y-position
        });
  }
  return TaskStatus::complete;
}

TaskStatus FixTrivalentNodes2D(MeshData<Real> *md) {
  using TE = parthenon::TopologicalElement;
  auto pmesh = md->GetMeshPointer();
  auto desc = parthenon::MakePackDescriptor<position>(md);

  IndexRange ib_in = md->GetBoundsI(IndexDomain::interior, TE::NN);
  IndexRange jb_in = md->GetBoundsJ(IndexDomain::interior, TE::NN);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior, TE::NN);

  for (auto &ptree : pmesh->forest.GetTrees()) {
    auto tree_id = ptree->GetId();
    int pos{0};
    for (auto &pnode : ptree->forest_nodes) {
      bool trivalent =
          (pnode->associated_faces.size() == 3) && !pnode->on_physical_boundary;
      if (trivalent) {
        parthenon::CellCentOffsets offset(2 * (pos % 2) - 1, 2 * (pos / 2) - 1, 0);
        auto pack =
            desc.GetPack(md, parthenon::GetBlockSelector::OnTree(ptree->GetId(), offset));
        if (pack.GetNBlocks() == 0) return TaskStatus::complete;
        // Copy shared boundary data

        // This pack has to have only one block
        auto gid = pack.GetGIDHost(0);
        parthenon::MeshBlock *pmb;
        for (auto &pmbd : md->GetAllBlockData())
          if (pmbd->GetParentPointer()->gid == gid) pmb = pmbd->GetParentPointer();

        // Now check which row of corner shared elements has been set by an owning block
        parthenon::CellCentOffsets offsetX1(2 * (pos % 2) - 1, 0, 0);
        parthenon::CellCentOffsets offsetX2(0, 2 * (pos / 2) - 1, 0);
        int dir = X2DIR;
        for (auto &neighbor : pmb->neighbors) {
          if (neighbor.offsets == offsetX1 && neighbor.origin_ownership(offsetX2))
            dir = X1DIR;
          if (neighbor.offsets == offsetX2 && neighbor.origin_ownership(offsetX1))
            dir = X2DIR;
        }

        // Select the reference location on the node shared by all three blocks
        const int icorner = (offset(X1DIR) == parthenon::Offset::Low) ? ib_in.s : ib_in.e;
        const int jcorner = (offset(X2DIR) == parthenon::Offset::Low) ? jb_in.s : jb_in.e;
        // Select the index space of elements that need to get overwritten
        IndexRange ib = offset(X1DIR) == parthenon::Offset::Low
                            ? IndexRange{0, ib_in.s - (dir == X2DIR)}
                            : IndexRange{ib_in.e + (dir == X2DIR),
                                         ib_in.e + parthenon::Globals::nghost};
        IndexRange jb = offset(X2DIR) == parthenon::Offset::Low
                            ? IndexRange{0, jb_in.s - (dir == X1DIR)}
                            : IndexRange{jb_in.e + (dir == X1DIR),
                                         ib_in.e + parthenon::Globals::nghost};
        parthenon::par_for(
            parthenon::loop_pattern_mdrange_tag, "SetPosition", DevExecSpace(), 0,
            pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
            KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
              const int enclosed_area = std::abs((i - icorner) * (j - jcorner));
              int ioff{0};
              int joff{0};
              // This is an element shared by both of the neighbors, but only one
              // communicated it, so copy it to the other side of the corner
              if (enclosed_area == 0) {
                joff = std::abs(icorner - i) * offset(X2DIR);
                ioff = std::abs(jcorner - j) * offset(X1DIR);
              } else {
                // These are non-existent elements that we fill with data copied from
                // their nearest neighbors
                int off = 1;
                while (off * off < enclosed_area)
                  off++;
                ioff = offset(X1DIR) * off * (dir == X1DIR);
                joff = offset(X2DIR) * off * (dir == X2DIR);
              }
              pack(b, TE::NN, position(0), k, j, i) =
                  pack(b, TE::NN, position(0), k, jcorner + joff,
                       icorner + ioff); // x-position
              pack(b, TE::NN, position(1), k, j, i) =
                  pack(b, TE::NN, position(1), k, jcorner + joff,
                       icorner + ioff); // y-position
            });
      }
      pos++;
    }
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
  package->AddField<neighbor_info>(m);

  Metadata m_node({Metadata::Node, Metadata::Independent, Metadata::FillGhost},
                  std::vector<int>{2});
  package->AddField<position>(m_node);
  return package;
}

} // namespace boundary_exchange
