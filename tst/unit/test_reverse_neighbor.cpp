//========================================================================================
// (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
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

// Step 6a: GetInverseTransform and the ReverseNeighbor descriptor. Verifies the transform
// inverse behaviorally, and -- on a single rank where the neighbor is itself a live block
// holding a back-pointing NeighborBlock -- that ReverseNeighbor(BlockInfo(A), nb_A->B)
// reproduces exactly the {BlockInfo(B), nb_B->A} the mesh built independently. Checked
// across a statically-refined mesh (coarse/fine neighbors) and a two-tree forest with a
// non-trivial logical coordinate transformation.

#include <array>
#include <memory>
#include <sstream>
#include <unordered_map>

#include <catch2/catch.hpp>

#include "application_input.hpp"
#include "bvals/comms/calc_indices.hpp"
#include "bvals/neighbor_block.hpp"
#include "globals.hpp"
#include "interface/packages.hpp"
#include "interface/state_descriptor.hpp"
#include "mesh/forest/forest.hpp"
#include "mesh/forest/forest_node.hpp"
#include "mesh/forest/forest_topology.hpp"
#include "mesh/forest/logical_coordinate_transformation.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"

using parthenon::ApplicationInput;
using parthenon::BlockInfo;
using parthenon::Mesh;
using parthenon::NeighborBlock;
using parthenon::Packages_t;
using parthenon::ParameterInput;
using parthenon::Real;
using parthenon::ReverseNeighbor;
using parthenon::StateDescriptor;
using parthenon::X1DIR;
using parthenon::X2DIR;
using parthenon::X3DIR;
namespace forest = parthenon::forest;

namespace {

constexpr int kNGhost = 2;

Packages_t MakePackages() {
  Packages_t packages;
  packages.Add(std::make_shared<StateDescriptor>("reverse_neighbor_test"));
  return packages;
}

// A 2x2-block periodic mesh with one corner statically refined a level, so blocks carry a
// mix of same-level, finer, and coarser neighbors.
std::shared_ptr<Mesh> MakeRefinedMesh(ApplicationInput *app_in, Packages_t &packages) {
  std::stringstream is;
  is << "<parthenon/mesh>\n";
  is << "refinement = static\n";
  is << "nghost = " << kNGhost << "\n";
  is << "nx1 = 8\nx1min = 0.0\nx1max = 1.0\nix1_bc = periodic\nox1_bc = periodic\n";
  is << "nx2 = 8\nx2min = 0.0\nx2max = 1.0\nix2_bc = periodic\nox2_bc = periodic\n";
  is << "nx3 = 1\nx3min = 0.0\nx3max = 1.0\nix3_bc = outflow\nox3_bc = outflow\n";
  is << "<parthenon/meshblock>\nnx1 = 4\nnx2 = 4\nnx3 = 1\n";
  is << "<parthenon/static_refinement0>\n";
  is << "level = 1\nx1min = 0.0\nx1max = 0.5\nx2min = 0.0\nx2max = 0.5\n";
  auto pin = std::make_shared<ParameterInput>();
  pin->LoadFromStream(is);
  return std::make_shared<Mesh>(pin.get(), app_in, packages, 0);
}

// A 2D two-tree forest whose second tree is glued on with a rotated/flipped orientation,
// so the cross-tree logical coordinate transformation is non-trivial. Mirrors the gold
// test's forest construction.
std::shared_ptr<Mesh> MakeForestMesh(ApplicationInput *app_in, Packages_t &packages) {
  using forest::Node;
  using ar3_t = std::array<Real, 3>;

  std::unordered_map<int, std::shared_ptr<Node>> n;
  n[0] = Node::create(0, {0.0, 0.0});
  n[1] = Node::create(1, {1.0, 0.0});
  n[2] = Node::create(2, {0.0, 1.0});
  n[3] = Node::create(3, {1.0, 1.0});
  n[4] = Node::create(4, {2.0, 0.0});
  n[5] = Node::create(5, {2.0, 1.0});

  forest::ForestDefinition forest_def;
  forest_def.AddFace(0, {n[0], n[1], n[2], n[3]}, ar3_t{0.0, 0.0, 0.0},
                     ar3_t{1.0, 1.0, 1.0});
  // Tree 1's shared edge {n1,n3} runs along its X0 axis (swapped vs tree 0's X1) and in
  // reversed order (a flip) -- a genuine rotation+flip transform.
  forest_def.AddFace(1, {n[3], n[1], n[5], n[4]}, ar3_t{1.0, 0.0, 0.0},
                     ar3_t{2.0, 1.0, 1.0});

  using edge_t = forest::Edge;
  forest_def.AddBC(edge_t({n[0], n[1]}));
  forest_def.AddBC(edge_t({n[0], n[2]}));
  forest_def.AddBC(edge_t({n[2], n[3]}));
  forest_def.AddBC(edge_t({n[1], n[4]}));
  forest_def.AddBC(edge_t({n[4], n[5]}));
  forest_def.AddBC(edge_t({n[3], n[5]}));

  std::stringstream is;
  is << "<parthenon/mesh>\nnghost = " << kNGhost << "\n";
  is << "nx1 = 8\nnx2 = 8\nnx3 = 1\n";
  is << "<parthenon/meshblock>\nnx1 = 4\nnx2 = 4\nnx3 = 1\n";
  auto pin = std::make_shared<ParameterInput>();
  pin->LoadFromStream(is);
  return std::make_shared<Mesh>(pin.get(), app_in, packages, forest_def);
}

bool HasNonTrivialTransform(const std::shared_ptr<Mesh> &mesh) {
  const forest::LogicalCoordinateTransformation identity;
  for (const auto &pmb : mesh->block_list)
    for (const auto &nb : pmb->GetNeighbors()) {
      const auto &t = nb.lcoord_trans;
      if (t.dir_connection != identity.dir_connection || t.dir_flip != identity.dir_flip)
        return true;
    }
  return false;
}

std::shared_ptr<parthenon::MeshBlock> FindBlock(const std::shared_ptr<Mesh> &mesh,
                                                int gid) {
  for (const auto &pmb : mesh->block_list)
    if (pmb->gid == gid) return pmb;
  return nullptr;
}

bool SameLct(const forest::LogicalCoordinateTransformation &a,
             const forest::LogicalCoordinateTransformation &b) {
  return a.dir_connection == b.dir_connection &&
         a.dir_connection_inverse == b.dir_connection_inverse &&
         a.dir_flip == b.dir_flip && a.offset == b.offset;
}

// Ground-truth check: single-rank, so the neighbor B is itself a live MeshBlock holding a
// NeighborBlock describing A. ReverseNeighbor(BlockInfo(A), nb_A->B) must reproduce exactly
// the {BlockInfo(B), nb_B->A} the mesh built independently for B. Returns #boundaries.
int CheckAgainstMesh(const std::shared_ptr<Mesh> &mesh) {
  int nchecked = 0;
  for (const auto &pmbA : mesh->block_list) {
    BlockInfo biA(pmbA.get());
    for (const auto &nbAB : pmbA->GetNeighbors()) {
      auto [biB_calc, nbBA_calc] = ReverseNeighbor(biA, nbAB);

      // Locate B and its actual NeighborBlock back at A (unique by gid + reversed offset).
      auto pmbB = FindBlock(mesh, nbAB.gid);
      REQUIRE(pmbB != nullptr);
      const NeighborBlock *nbBA = nullptr;
      for (const auto &cand : pmbB->GetNeighbors()) {
        if (cand.gid == pmbA->gid &&
            cand.offsets(X1DIR) == nbBA_calc.offsets(X1DIR) &&
            cand.offsets(X2DIR) == nbBA_calc.offsets(X2DIR) &&
            cand.offsets(X3DIR) == nbBA_calc.offsets(X3DIR)) {
          nbBA = &cand;
          break;
        }
      }
      INFO("A.gid=" << pmbA->gid << " B.gid=" << nbAB.gid << " off=(" << nbAB.offsets(X1DIR)
                    << "," << nbAB.offsets(X2DIR) << "," << nbAB.offsets(X3DIR) << ")");
      REQUIRE(nbBA != nullptr);

      // BlockInfo(B) reconstructed from the descriptor matches B's real geometry/ownership.
      BlockInfo biB(pmbB.get());
      REQUIRE(biB_calc.gid == biB.gid);
      REQUIRE(biB_calc.loc == biB.loc);
      REQUIRE(biB_calc.block_coarsenings == biB.block_coarsenings);
      for (auto dir : {X1DIR, X2DIR, X3DIR}) {
        REQUIRE(biB_calc.block_size.nx(dir) == biB.block_size.nx(dir));
        REQUIRE(biB_calc.block_size.symmetry(dir) == biB.block_size.symmetry(dir));
      }
      REQUIRE(biB_calc.ownership == biB.ownership);

      // The reverse NeighborBlock matches the one the mesh built for B.
      REQUIRE(nbBA_calc.gid == nbBA->gid);
      REQUIRE(nbBA_calc.loc == nbBA->loc);
      REQUIRE(nbBA_calc.origin_loc == nbBA->origin_loc);
      REQUIRE(nbBA_calc.block_coarsenings == nbBA->block_coarsenings);
      for (auto dir : {X1DIR, X2DIR, X3DIR})
        REQUIRE(nbBA_calc.offsets(dir) == nbBA->offsets(dir));
      REQUIRE(nbBA_calc.ownership == nbBA->ownership);
      REQUIRE(SameLct(nbBA_calc.lcoord_trans, nbBA->lcoord_trans));
      ++nchecked;
    }
  }
  return nchecked;
}

} // namespace

TEST_CASE("GetInverseTransform inverts a logical coordinate transformation", "[bvals]") {
  using forest::ComposeTransformations;
  using forest::GetInverseTransform;
  using forest::LogicalCoordinateTransformation;

  // A non-trivial transform: swap X1<->X2 and flip the (now-X1) direction.
  LogicalCoordinateTransformation t;
  t.SetDirection(X1DIR, X2DIR, /*reversed=*/true);
  t.SetDirection(X2DIR, X1DIR, /*reversed=*/false);
  t.SetDirection(X3DIR, X3DIR, /*reversed=*/false);
  t.ncell = 8;

  auto inv = GetInverseTransform(t);
  inv.ncell = t.ncell;

  const std::array<std::array<int, 3>, 4> pts{{{0, 0, 0}, {1, 2, 0}, {7, 3, 0}, {5, 6, 0}}};

  THEN("inv.Transform reproduces t.InverseTransform on index triples") {
    for (auto x : pts)
      REQUIRE(inv.Transform(x) == t.InverseTransform(x));
  }

  THEN("Composing t with its inverse round-trips index triples to identity") {
    auto rt = ComposeTransformations(t, GetInverseTransform(t));
    rt.ncell = t.ncell;
    for (auto x : pts)
      REQUIRE(rt.Transform(x) == x);
  }
}

TEST_CASE("ReverseNeighbor matches the mesh's own back-neighbor", "[bvals][mesh][MPI]") {
  parthenon::Globals::nghost = kNGhost;
  auto app_in = std::make_shared<ApplicationInput>();
  auto packages = MakePackages();

  GIVEN("A statically-refined periodic mesh (same-level, finer, and coarser neighbors)") {
    auto mesh = MakeRefinedMesh(app_in.get(), packages);
    THEN("The reverse descriptor matches the mesh's own back-neighbor for each boundary") {
      REQUIRE(CheckAgainstMesh(mesh) > 0);
    }
  }

  GIVEN("A two-tree forest with a non-trivial coordinate transformation") {
    auto mesh = MakeForestMesh(app_in.get(), packages);
    REQUIRE(HasNonTrivialTransform(mesh));
    THEN("The reverse descriptor matches the mesh's own back-neighbor for each boundary") {
      REQUIRE(CheckAgainstMesh(mesh) > 0);
    }
  }
}
