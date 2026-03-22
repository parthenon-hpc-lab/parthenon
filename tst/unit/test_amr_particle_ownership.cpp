//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2026 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
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

// This file was made in part with generative AI

#include <algorithm>
#include <memory>
#include <optional>
#include <sstream>
#include <vector>

#include <catch2/catch.hpp>

#include "mesh/amr_particle_ownership.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"

#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

using parthenon::ApplicationInput;
using parthenon::BoundaryFlag;
using parthenon::CoordinateDirection;
using parthenon::IndexDomain;
using parthenon::LogicalLocation;
using parthenon::Mesh;
using parthenon::MeshBlock;
using parthenon::Packages_t;
using parthenon::ParameterInput;
using parthenon::Real;
using parthenon::RegionSize;
using parthenon::StateDescriptor;

namespace {

std::shared_ptr<Mesh> MakeUniformMesh() {
  // This is deliberately the same logical shape as the tracer example: a 2D periodic
  // domain tiled by 16x16-cell meshblocks. The ownership test only cares about the mesh
  // geometry, so a single empty package is sufficient.
  std::stringstream is;
  is << "<parthenon/mesh>\n";
  is << "refinement = adaptive\n";
  is << "numlevel = 2\n";
  is << "nx1 = 64\n";
  is << "x1min = -0.5\n";
  is << "x1max = 0.5\n";
  is << "ix1_bc = periodic\n";
  is << "ox1_bc = periodic\n";
  is << "nx2 = 64\n";
  is << "x2min = -0.5\n";
  is << "x2max = 0.5\n";
  is << "ix2_bc = periodic\n";
  is << "ox2_bc = periodic\n";
  is << "nx3 = 1\n";
  is << "x3min = -0.5\n";
  is << "x3max = 0.5\n";
  is << "ix3_bc = periodic\n";
  is << "ox3_bc = periodic\n";
  is << "<parthenon/meshblock>\n";
  is << "nx1 = 16\n";
  is << "nx2 = 16\n";
  is << "nx3 = 1\n";

  auto pin = std::make_shared<ParameterInput>();
  pin->LoadFromStream(is);
  auto app_in = std::make_shared<ApplicationInput>();
  Packages_t packages;
  packages.Add(std::make_shared<StateDescriptor>("test"));

  return std::make_shared<Mesh>(pin.get(), app_in.get(), packages, 1);
}

std::vector<LogicalLocation> ReplaceLeaf(const std::vector<LogicalLocation> &leaves,
                                         const LogicalLocation &old_leaf,
                                         const std::vector<LogicalLocation> &new_leaves) {
  // Mirror a remesh event by replacing one existing leaf block with the leaf set that
  // would exist after refinement or derefinement.
  std::vector<LogicalLocation> out;
  out.reserve(leaves.size() - 1 + new_leaves.size());
  for (const auto &loc : leaves) {
    if (loc != old_leaf) out.push_back(loc);
  }
  out.insert(out.end(), new_leaves.begin(), new_leaves.end());
  return out;
}

Real CellCenter(const RegionSize &block_size, const CoordinateDirection dir,
                const int local_cell_index) {
  const Real dx = (block_size.xmax(dir) - block_size.xmin(dir)) /
                  static_cast<Real>(block_size.nx(dir));
  return block_size.xmin(dir) + (static_cast<Real>(local_cell_index) + 0.5) * dx;
}

LogicalLocation FindLeafWithSameLevelNeighbor(const std::vector<LogicalLocation> &leaves,
                                              const int ox1, const int ox2,
                                              const int ox3) {
  for (const auto &loc : leaves) {
    const auto neighbor = loc.GetSameLevelNeighbor(ox1, ox2, ox3);
    if (std::find(leaves.begin(), leaves.end(), neighbor) != leaves.end()) {
      return loc;
    }
  }
  REQUIRE(false);
  return {};
}

bool IntervalsOverlap(const Real a0, const Real a1, const Real b0, const Real b1) {
  return std::max(a0, b0) < std::min(a1, b1);
}

std::optional<std::pair<LogicalLocation, LogicalLocation>> FindSharedXFacePair(
    const std::shared_ptr<Mesh> &mesh, const std::vector<LogicalLocation> &leaves,
    const std::function<bool(const LogicalLocation &, const LogicalLocation &)>
        &predicate) {
  for (std::size_t i = 0; i < leaves.size(); ++i) {
    const auto a = leaves[i];
    const auto as = mesh->GetBlockSize(a);
    for (std::size_t j = i + 1; j < leaves.size(); ++j) {
      const auto b = leaves[j];
      if (!predicate(a, b)) continue;
      const auto bs = mesh->GetBlockSize(b);

      const bool a_left_b = as.xmax(parthenon::X1DIR) == bs.xmin(parthenon::X1DIR);
      const bool b_left_a = bs.xmax(parthenon::X1DIR) == as.xmin(parthenon::X1DIR);
      if (!(a_left_b || b_left_a)) continue;

      if (!IntervalsOverlap(as.xmin(parthenon::X2DIR), as.xmax(parthenon::X2DIR),
                            bs.xmin(parthenon::X2DIR), bs.xmax(parthenon::X2DIR))) {
        continue;
      }

      return a_left_b ? std::optional{std::make_pair(a, b)}
                      : std::optional{std::make_pair(b, a)};
    }
  }
  return std::nullopt;
}

std::optional<std::tuple<LogicalLocation, std::vector<LogicalLocation>,
                         std::pair<LogicalLocation, LogicalLocation>>>
FindRefinementWithFineCoarseSharedFace(const std::shared_ptr<Mesh> &mesh,
                                       const std::vector<LogicalLocation> &base_leaves) {
  for (const auto &candidate_parent : base_leaves) {
    const auto candidate_leaves = ReplaceLeaf(base_leaves, candidate_parent,
                                              candidate_parent.GetDaughters(mesh->ndim));
    const auto pair =
        FindSharedXFacePair(mesh, candidate_leaves, [](const auto &a, const auto &b) {
          return a.level() != b.level();
        });
    if (pair.has_value())
      return std::make_optional(
          std::make_tuple(candidate_parent, candidate_leaves, *pair));
  }
  return std::nullopt;
}

} // namespace

TEST_CASE("AMR particle ownership lookup matches refine and derefine expectations",
          "[AMR][Swarm][Ownership][MPI]") {
  auto mesh = MakeUniformMesh();
  const auto base_leaves = mesh->GetLocList();
  REQUIRE(!base_leaves.empty());

  // Any base leaf works in a periodic mesh. Using the first one keeps the test simple and
  // deterministic.
  const auto parent = base_leaves.front();
  const auto daughters = parent.GetDaughters(mesh->ndim);
  const auto refined_leaves = ReplaceLeaf(base_leaves, parent, daughters);

  // The ownership lookup only needs the logical mesh geometry from Mesh. For cell-index
  // checks we can use a standalone meshblock with the same 16x16 resolution as the test
  // mesh, which avoids depending on any driver-side block allocation details.
  MeshBlock reference_block(16, mesh->ndim);
  const auto &cellbounds = reference_block.cellbounds;
  const auto ib = cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = cellbounds.GetBoundsK(IndexDomain::interior);

  SECTION("Refinement assigns a coarse-block particle to the expected daughter block") {
    const auto expected_daughter = parent.GetDaughter(1, 0, 0);
    const auto daughter_size = mesh->GetBlockSize(expected_daughter);

    // Pick an unambiguous daughter-cell center well away from any daughter interface.
    const int local_i = 5;
    const int local_j = 7;
    const Real x = CellCenter(daughter_size, parthenon::X1DIR, local_i);
    const Real y = CellCenter(daughter_size, parthenon::X2DIR, local_j);
    const Real z = 0.0;

    const int owner =
        parthenon::amr::FindOwningBlock(mesh.get(), refined_leaves, x, y, z);
    REQUIRE(owner >= 0);
    REQUIRE(refined_leaves[owner] == expected_daughter);

    const auto ijk =
        parthenon::amr::FindCellIndices(daughter_size, cellbounds, x, y, z, mesh->ndim);
    REQUIRE(ijk[0] == ib.s + local_i);
    REQUIRE(ijk[1] == jb.s + local_j);
    REQUIRE(ijk[2] == kb.s);
  }

  SECTION("Derefine assigns a fine-block particle back to the expected parent cell") {
    const auto fine_owner = parent.GetDaughter(0, 1, 0);
    const auto fine_size = mesh->GetBlockSize(fine_owner);
    const auto coarse_size = mesh->GetBlockSize(parent);

    // Choose the center of a specific fine cell on the upper-left daughter. After
    // derefinement, the same physical point must map back to the matching coarse cell on
    // the parent block.
    const int fine_i = 6;
    const int fine_j = 11;
    const int expected_coarse_i = fine_i / 2;
    const int expected_coarse_j = (coarse_size.nx(parthenon::X2DIR) / 2) + fine_j / 2;
    const Real x = CellCenter(fine_size, parthenon::X1DIR, fine_i);
    const Real y = CellCenter(fine_size, parthenon::X2DIR, fine_j);
    const Real z = 0.0;

    const int refined_owner =
        parthenon::amr::FindOwningBlock(mesh.get(), refined_leaves, x, y, z);
    REQUIRE(refined_owner >= 0);
    REQUIRE(refined_leaves[refined_owner] == fine_owner);

    const auto fine_ijk =
        parthenon::amr::FindCellIndices(fine_size, cellbounds, x, y, z, mesh->ndim);
    REQUIRE(fine_ijk[0] == ib.s + fine_i);
    REQUIRE(fine_ijk[1] == jb.s + fine_j);
    REQUIRE(fine_ijk[2] == kb.s);

    const int derefined_owner =
        parthenon::amr::FindOwningBlock(mesh.get(), base_leaves, x, y, z);
    REQUIRE(derefined_owner >= 0);
    REQUIRE(base_leaves[derefined_owner] == parent);

    const auto coarse_ijk =
        parthenon::amr::FindCellIndices(coarse_size, cellbounds, x, y, z, mesh->ndim);
    REQUIRE(coarse_ijk[0] == ib.s + expected_coarse_i);
    REQUIRE(coarse_ijk[1] == jb.s + expected_coarse_j);
    REQUIRE(coarse_ijk[2] == kb.s);
  }

  SECTION("Ownership near a daughter interface stays on the expected side of the split") {
    const auto left_daughter = parent.GetDaughter(0, 0, 0);
    const auto right_daughter = parent.GetDaughter(1, 0, 0);
    const auto right_size = mesh->GetBlockSize(right_daughter);

    // This point sits just inside the right daughter, immediately to the right of the
    // internal fine/fine interface. That makes it a useful edge case for the geometric
    // ownership rule without relying on an exactly-on-the-interface convention.
    const Real dx =
        (right_size.xmax(parthenon::X1DIR) - right_size.xmin(parthenon::X1DIR)) /
        static_cast<Real>(right_size.nx(parthenon::X1DIR));
    const Real x = right_size.xmin(parthenon::X1DIR) + 1.0e-10 * dx;
    const Real y = CellCenter(right_size, parthenon::X2DIR, 3);
    const Real z = 0.0;

    const int owner =
        parthenon::amr::FindOwningBlock(mesh.get(), refined_leaves, x, y, z);
    REQUIRE(owner >= 0);
    REQUIRE(refined_leaves[owner] == right_daughter);
    REQUIRE(refined_leaves[owner] != left_daughter);

    const auto ijk =
        parthenon::amr::FindCellIndices(right_size, cellbounds, x, y, z, mesh->ndim);
    REQUIRE(ijk[0] == ib.s);
    REQUIRE(ijk[1] == jb.s + 3);
    REQUIRE(ijk[2] == kb.s);
  }

  SECTION(
      "Exact same-level shared-face ownership follows Parthenon tree/Morton priority") {
    const auto left = parent.GetDaughter(0, 0, 0);
    const auto right = parent.GetDaughter(1, 0, 0);
    const auto left_size = mesh->GetBlockSize(left);
    const auto right_size = mesh->GetBlockSize(right);
    const Real x = left_size.xmax(parthenon::X1DIR);
    const Real y =
        0.5 *
        (std::max(left_size.xmin(parthenon::X2DIR), right_size.xmin(parthenon::X2DIR)) +
         std::min(left_size.xmax(parthenon::X2DIR), right_size.xmax(parthenon::X2DIR)));
    const Real z = 0.0;

    const auto expected_owner =
        parthenon::amr::OwnershipLessThan(left, right) ? right : left;
    const auto expected_size = mesh->GetBlockSize(expected_owner);

    // The shared face under test is the internal face between two daughters created by
    // refining `parent`, so the owning block must be searched for on the refined leaf
    // set. Looking on the base leaf set would ask a different question entirely: which
    // coarse block owns the same physical position before refinement.
    const int owner =
        parthenon::amr::FindOwningBlock(mesh.get(), refined_leaves, x, y, z);
    REQUIRE(owner >= 0);
    REQUIRE(refined_leaves[owner] == expected_owner);

    const auto ijk =
        parthenon::amr::FindCellIndices(expected_size, cellbounds, x, y, z, mesh->ndim);
    REQUIRE(ijk[0] >= ib.s);
    REQUIRE(ijk[0] <= ib.e);
    REQUIRE(ijk[1] >= jb.s);
    REQUIRE(ijk[1] <= jb.e);
    REQUIRE(ijk[2] == kb.s);
  }

  SECTION("Exact fine-coarse shared-face ownership prefers the finer block") {
    const auto remesh_case = FindRefinementWithFineCoarseSharedFace(mesh, base_leaves);
    REQUIRE(remesh_case.has_value());
    const auto &[chosen_parent, chosen_leaves, pair] = *remesh_case;
    const auto left = pair.first;
    const auto right = pair.second;
    const auto fine = left.level() > right.level() ? left : right;
    const auto coarse = left.level() > right.level() ? right : left;
    const auto fine_size = mesh->GetBlockSize(fine);
    const auto coarse_size = mesh->GetBlockSize(coarse);

    const Real x = fine_size.xmax(parthenon::X1DIR) == coarse_size.xmin(parthenon::X1DIR)
                       ? fine_size.xmax(parthenon::X1DIR)
                       : coarse_size.xmax(parthenon::X1DIR);
    const Real y =
        0.5 *
        (std::max(fine_size.xmin(parthenon::X2DIR), coarse_size.xmin(parthenon::X2DIR)) +
         std::min(fine_size.xmax(parthenon::X2DIR), coarse_size.xmax(parthenon::X2DIR)));
    const Real z = 0.0;

    const int owner = parthenon::amr::FindOwningBlock(mesh.get(), chosen_leaves, x, y, z);
    REQUIRE(owner >= 0);
    REQUIRE(chosen_leaves[owner] == fine);

    const auto fine_ijk =
        parthenon::amr::FindCellIndices(fine_size, cellbounds, x, y, z, mesh->ndim);
    REQUIRE(fine_ijk[0] == ib.e);
    REQUIRE(fine_ijk[1] >= jb.s);
    REQUIRE(fine_ijk[1] <= jb.e);
    REQUIRE(fine_ijk[2] == kb.s);

    const auto coarse_ijk =
        parthenon::amr::FindCellIndices(coarse_size, cellbounds, x, y, z, mesh->ndim);
    REQUIRE(coarse_ijk[0] >= ib.s);
    REQUIRE(coarse_ijk[0] <= ib.e);
    REQUIRE(coarse_ijk[1] >= jb.s);
    REQUIRE(coarse_ijk[1] <= jb.e);
    REQUIRE(coarse_ijk[2] == kb.s);
  }
}
