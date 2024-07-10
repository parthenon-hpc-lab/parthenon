//========================================================================================
// (C) (or copyright) 2024. Triad National Security, LLC. All rights reserved.
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
#include <array>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stack>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "basic_types.hpp"
#include "defs.hpp"
#include "mesh/forest/forest_topology.hpp"
#include "mesh/forest/logical_coordinate_transformation.hpp"
#include "mesh/forest/logical_location.hpp"
#include "mesh/forest/tree.hpp"
#include "utils/bit_hacks.hpp"
#include "utils/indexer.hpp"

namespace parthenon {
namespace forest {

Tree::Tree(Tree::private_t, std::int64_t id, int ndim, int root_level)
    : my_id(id), ndim(ndim) {
  // Add internal and leaf nodes of the initial tree
  for (int l = 0; l <= root_level; ++l) {
    for (int k = 0; k < (ndim > 2 ? (1LL << l) : 1); ++k) {
      for (int j = 0; j < (ndim > 1 ? (1LL << l) : 1); ++j) {
        for (int i = 0; i < (ndim > 0 ? (1LL << l) : 1); ++i) {
          LogicalLocation loc(my_id, l, i, j, k);
          if (l == root_level) {
            leaves.emplace(LocMapEntry(loc, -1, -1));
          } else {
            internal_nodes.emplace(LocMapEntry(loc, -1, -1));
          }
        }
      }
    }
  }

  // Build in negative levels
  for (int l = -20; l < 0; ++l) {
    internal_nodes.emplace(LocMapEntry(LogicalLocation(my_id, l, 0, 0, 0), -1, -1));
  }
}

Tree::Tree(Tree::private_t, std::int64_t id, int ndim, int root_level,
           RegionSize domain_in, std::array<BoundaryFlag, BOUNDARY_NFACES> bcs_in)
    : Tree(Tree::private_t(), id, ndim, root_level) {
  domain = domain_in;
  boundary_conditions = bcs_in;
}

int Tree::AddMeshBlock(const LogicalLocation &loc, bool enforce_proper_nesting) {
  PARTHENON_REQUIRE(
      loc.tree() == my_id,
      "Trying to add a meshblock to a tree with a LogicalLocation on a different tree.");
  if (internal_nodes.count(loc)) return -1;
  if (leaves.count(loc)) return 0;

  std::stack<LogicalLocation> refinement_locs;
  auto parent = loc.GetParent();
  for (int l = loc.level() - 1; l >= 0; --l) {
    refinement_locs.push(parent);
    if (leaves.count(parent)) break;
    parent = parent.GetParent();
  }

  int added = 0;
  while (!refinement_locs.empty()) {
    added += Refine(refinement_locs.top(), enforce_proper_nesting);
    refinement_locs.pop();
  }

  return added;
}

int Tree::Refine(const LogicalLocation &ref_loc, bool enforce_proper_nesting) {
  PARTHENON_REQUIRE(
      ref_loc.tree() == my_id,
      "Trying to refine a tree with a LogicalLocation on a different tree.");
  // Check that this is a valid refinement location
  if (!leaves.count(ref_loc)) return 0; // Can't refine a block that doesn't exist

  // Perform the refinement for this block
  std::vector<LogicalLocation> daughters = ref_loc.GetDaughters(ndim);
  auto gid_parent = leaves[ref_loc].first;
  leaves.erase(ref_loc);
  internal_nodes.insert(LocMapEntry(ref_loc, -1, -1));
  for (auto &d : daughters) {
    leaves.insert(LocMapEntry(d, gid_parent, -1));
  }
  int nadded = daughters.size() - 1;

  if (enforce_proper_nesting) {
    LogicalLocation parent = ref_loc.GetParent();
    int ox1 = ref_loc.lx1() - (parent.lx1() << 1);
    int ox2 = ref_loc.lx2() - (parent.lx2() << 1);
    int ox3 = ref_loc.lx3() - (parent.lx3() << 1);

    for (int k = 0; k < (ndim > 2 ? 2 : 1); ++k) {
      for (int j = 0; j < (ndim > 1 ? 2 : 1); ++j) {
        for (int i = 0; i < (ndim > 0 ? 2 : 1); ++i) {
          LogicalLocation neigh = parent.GetSameLevelNeighbor(
              i + ox1 - 1, j + ox2 - (ndim > 1), k + ox3 - (ndim > 2));
          // Need to communicate this refinement action to possible neighboring tree(s)
          // and trigger refinement there
          int n_idx =
              neigh.NeighborTreeIndex(); // Note that this can point you back to this tree
          for (auto &[neighbor_tree, lcoord_trans] : neighbors[n_idx]) {
            nadded += neighbor_tree->Refine(
                lcoord_trans.Transform(neigh, neighbor_tree->GetId()));
          }
        }
      }
    }
  }
  return nadded;
}

std::vector<NeighborLocation> Tree::FindNeighbors(const LogicalLocation &loc,
                                                  GridIdentifier grid_id) const {
  const Indexer3D offsets({ndim > 0 ? -1 : 0, ndim > 0 ? 1 : 0},
                          {ndim > 1 ? -1 : 0, ndim > 1 ? 1 : 0},
                          {ndim > 2 ? -1 : 0, ndim > 2 ? 1 : 0});
  std::vector<NeighborLocation> neighbor_locs;
  for (int o = 0; o < offsets.size(); ++o) {
    auto [ox1, ox2, ox3] = offsets(o);
    if (std::abs(ox1) + std::abs(ox2) + std::abs(ox3) == 0) continue;
    FindNeighborsImpl(loc, ox1, ox2, ox3, &neighbor_locs, grid_id);
  }

  return neighbor_locs;
}

std::vector<NeighborLocation> Tree::FindNeighbors(const LogicalLocation &loc, int ox1,
                                                  int ox2, int ox3) const {
  std::vector<NeighborLocation> neighbor_locs;
  FindNeighborsImpl(loc, ox1, ox2, ox3, &neighbor_locs, GridIdentifier::leaf());
  return neighbor_locs;
}

void Tree::FindNeighborsImpl(const LogicalLocation &loc, int ox1, int ox2, int ox3,
                             std::vector<NeighborLocation> *neighbor_locs,
                             GridIdentifier grid_id) const {
  PARTHENON_REQUIRE(
      loc.tree() == my_id,
      "Trying to find neighbors in a tree with a LogicalLocation on a different tree.");
  PARTHENON_REQUIRE((leaves.count(loc) == 1 || internal_nodes.count(loc) == 1),
                    "Location must be in the tree to find neighbors.");
  auto neigh = loc.GetSameLevelNeighbor(ox1, ox2, ox3);
  int n_idx = neigh.NeighborTreeIndex();

  bool include_same, include_fine, include_internal, include_coarse;
  if (grid_id.type == GridType::leaf) {
    include_same = true;
    include_fine = true;
    include_internal = false;
    include_coarse = true;
  } else if (grid_id.type == GridType::two_level_composite) {
    if (loc.level() == grid_id.logical_level) {
      include_same = true;
      include_fine = false;
      include_internal = true;
      include_coarse = true;
    } else if (loc.level() == grid_id.logical_level - 1) {
      include_same = false;
      include_fine = true;
      include_internal = false;
      include_coarse = false;
    } else {
      PARTHENON_FAIL("Logic is wrong somewhere.");
    }
  }

  for (auto &[neighbor_tree, lcoord_trans] : neighbors[n_idx]) {
    auto tneigh = lcoord_trans.Transform(neigh, neighbor_tree->GetId());
    auto tloc = lcoord_trans.Transform(loc, neighbor_tree->GetId());
    PARTHENON_REQUIRE(lcoord_trans.InverseTransform(tloc, GetId()) == loc,
                      "Inverse transform not working.");
    if (neighbor_tree->leaves.count(tneigh) && include_same) {
      neighbor_locs->push_back(NeighborLocation(
          tneigh, lcoord_trans.InverseTransform(tneigh, GetId()), lcoord_trans));
    } else if (neighbor_tree->internal_nodes.count(tneigh)) {
      if (include_fine) {
        auto daughters = tneigh.GetDaughters(neighbor_tree->ndim);
        for (auto &n : daughters) {
          if (tloc.IsNeighbor(n))
            neighbor_locs->push_back(NeighborLocation(
                n, lcoord_trans.InverseTransform(n, GetId()), lcoord_trans));
        }
      } else if (include_internal) {
        neighbor_locs->push_back(NeighborLocation(
            tneigh, lcoord_trans.InverseTransform(tneigh, GetId()), lcoord_trans));
      }
    } else if (neighbor_tree->leaves.count(tneigh.GetParent()) && include_coarse) {
      auto neighp = lcoord_trans.InverseTransform(tneigh.GetParent(), GetId());
      // Since coarser neighbors can cover multiple elements of the origin block and
      // because our communication algorithm packs this extra data by hand, we do not wish
      // to duplicate coarser blocks in the neighbor list. Therefore, we only include the
      // coarse block in one offset position
      auto sl_offset = loc.GetSameLevelOffsets(neighp);
      if (sl_offset[0] == ox1 && sl_offset[1] == ox2 && sl_offset[2] == ox3)
        neighbor_locs->push_back(
            NeighborLocation(tneigh.GetParent(), neighp, lcoord_trans));
    }
  }
}

int Tree::Derefine(const LogicalLocation &ref_loc, bool enforce_proper_nesting) {
  PARTHENON_REQUIRE(
      ref_loc.tree() == my_id,
      "Trying to derefine a tree with a LogicalLocation on a different tree.");

  // ref_loc is the block to be added and its daughters are the blocks to be removed
  std::vector<LogicalLocation> daughters = ref_loc.GetDaughters(ndim);

  // Check that we can actually de-refine
  for (LogicalLocation &d : daughters) {
    // Check that the daughters actually exist as leaf nodes
    if (!leaves.count(d)) return 0;

    // Check that removing these blocks doesn't break proper nesting, that just means that
    // any of the daughters same level neighbors can't be in the internal node list (which
    // would imply that the daughter abuts a finer block) Note: these loops check more
    // than is necessary, but as written are simpler than the minimal set
    if (enforce_proper_nesting) {
      const std::vector<int> active{-1, 0, 1};
      const std::vector<int> inactive{0};
      for (int k : (ndim > 2) ? active : inactive) {
        for (int j : (ndim > 1) ? active : inactive) {
          for (int i : (ndim > 0) ? active : inactive) {
            LogicalLocation neigh = d.GetSameLevelNeighbor(i, j, k);
            // Need to check that this derefinement doesn't break proper nesting with
            // a neighboring tree or this tree
            int n_idx = neigh.NeighborTreeIndex();
            for (auto &[neighbor_tree, lcoord_trans] : neighbors[n_idx]) {
              if (neighbor_tree->internal_nodes.count(
                      lcoord_trans.Transform(neigh, neighbor_tree->GetId())))
                return 0;
            }
          }
        }
      }
    }
  }

  // Derefinement is ok
  std::int64_t dgid = std::numeric_limits<std::int64_t>::max();
  for (auto &d : daughters) {
    auto node = leaves.extract(d);
    dgid = std::min(dgid, node.mapped().first);
  }
  internal_nodes.erase(ref_loc);
  leaves.insert(LocMapEntry(ref_loc, dgid, -1));
  return daughters.size() - 1;
}

std::vector<LogicalLocation> Tree::GetSortedMeshBlockList() const {
  std::vector<LogicalLocation> mb_list;
  mb_list.reserve(leaves.size());
  for (auto &[loc, gid] : leaves)
    mb_list.push_back(loc);
  std::sort(mb_list.begin(), mb_list.end(),
            [](const auto &a, const auto &b) { return a < b; });
  return mb_list;
}

std::vector<LogicalLocation> Tree::GetSortedInternalNodeList() const {
  std::vector<LogicalLocation> mb_list;
  mb_list.reserve(internal_nodes.size());
  for (auto &[loc, gid] : internal_nodes)
    mb_list.push_back(loc);
  std::sort(mb_list.begin(), mb_list.end(),
            [](const auto &a, const auto &b) { return a < b; });
  return mb_list;
}

RegionSize Tree::GetBlockDomain(const LogicalLocation &loc) const {
  PARTHENON_REQUIRE(loc.IsInTree(), "Probably there is a mistake...");
  RegionSize out = domain;
  for (auto dir : {X1DIR, X2DIR, X3DIR}) {
    if (!domain.symmetry(dir)) {
      if (loc.level() >= 0) {
        out.xmin(dir) = domain.SymmetrizedLogicalToActualPosition(
            loc.LLCoord(dir, BlockLocation::Left), dir);
        out.xmax(dir) = domain.SymmetrizedLogicalToActualPosition(
            loc.LLCoord(dir, BlockLocation::Right), dir);
      } else {
        // Negative logical levels correspond to reduced block sizes covering the entire
        // domain.
        auto reduction_fac = 1LL << (-loc.level());
        PARTHENON_REQUIRE(domain.nx(dir) % reduction_fac == 0,
                          "Trying to go to too large of a negative level.");
        out.nx(dir) = domain.nx(dir) / reduction_fac;
      }
    }
    // If this is a translational symmetry direction, set the cell to cover the entire
    // tree in that direction.
  }
  return out;
}

std::array<BoundaryFlag, BOUNDARY_NFACES>
Tree::GetBlockBCs(const LogicalLocation &loc) const {
  PARTHENON_REQUIRE(loc.IsInTree(), "Probably there is a mistake...");
  std::array<BoundaryFlag, BOUNDARY_NFACES> block_bcs = boundary_conditions;
  const int nblock = 1 << std::max(loc.level(), 0);
  if (loc.lx1() != 0) block_bcs[BoundaryFace::inner_x1] = BoundaryFlag::block;
  if (loc.lx1() != nblock - 1) block_bcs[BoundaryFace::outer_x1] = BoundaryFlag::block;
  if (loc.lx2() != 0) block_bcs[BoundaryFace::inner_x2] = BoundaryFlag::block;
  if (loc.lx2() != nblock - 1) block_bcs[BoundaryFace::outer_x2] = BoundaryFlag::block;
  if (loc.lx3() != 0) block_bcs[BoundaryFace::inner_x3] = BoundaryFlag::block;
  if (loc.lx3() != nblock - 1) block_bcs[BoundaryFace::outer_x3] = BoundaryFlag::block;
  return block_bcs;
}

void Tree::AddNeighborTree(CellCentOffsets offset, std::shared_ptr<Tree> neighbor_tree,
                           LogicalCoordinateTransformation lcoord_trans,
                           const bool periodic) {
  int location_idx = offset.GetIdx();
  neighbors[location_idx].insert({neighbor_tree.get(), lcoord_trans});
  BoundaryFace fidx = offset.Face();

  if (fidx >= 0)
    boundary_conditions[fidx] = periodic ? BoundaryFlag::periodic : BoundaryFlag::block;
}

void Tree::InsertGid(const LogicalLocation &loc, std::int64_t gid) {
  if (leaves.count(loc)) {
    leaves[loc].second = leaves[loc].first;
    leaves[loc].first = gid;
  } else if (internal_nodes.count(loc)) {
    internal_nodes[loc].second = internal_nodes[loc].first;
    internal_nodes[loc].first = gid;
  } else {
    PARTHENON_FAIL("Tried to assign gid to non-existent block.");
  }
}

std::int64_t Tree::GetGid(const LogicalLocation &loc) const {
  if (leaves.count(loc)) {
    return leaves.at(loc).first;
  } else if (internal_nodes.count(loc)) {
    return internal_nodes.at(loc).first;
  }
  return -1;
}

// Get the gid of the leaf block with the same Morton number
// as loc
std::int64_t Tree::GetLeafGid(const LogicalLocation &loc) const {
  if (leaves.count(loc)) {
    return leaves.at(loc).first;
  } else if (internal_nodes.count(loc)) {
    return GetLeafGid(loc.GetDaughter(0, 0, 0));
  }
  return -1;
}

std::int64_t Tree::GetOldGid(const LogicalLocation &loc) const {
  if (leaves.count(loc)) {
    return leaves.at(loc).second;
  } else if (internal_nodes.count(loc)) {
    return internal_nodes.at(loc).second;
  }
  return -1;
}

void Tree::EnrollBndryFncts(
    ApplicationInput *app_in,
    std::array<std::vector<BValFunc>, BOUNDARY_NFACES> UserBoundaryFunctions_in,
    std::array<std::vector<SBValFunc>, BOUNDARY_NFACES> UserSwarmBoundaryFunctions_in) {
  UserBoundaryFunctions = UserBoundaryFunctions_in;
  UserSwarmBoundaryFunctions = UserSwarmBoundaryFunctions_in;
  static const BValFunc outflow[6] = {
      BoundaryFunction::OutflowInnerX1, BoundaryFunction::OutflowOuterX1,
      BoundaryFunction::OutflowInnerX2, BoundaryFunction::OutflowOuterX2,
      BoundaryFunction::OutflowInnerX3, BoundaryFunction::OutflowOuterX3};
  static const BValFunc reflect[6] = {
      BoundaryFunction::ReflectInnerX1, BoundaryFunction::ReflectOuterX1,
      BoundaryFunction::ReflectInnerX2, BoundaryFunction::ReflectOuterX2,
      BoundaryFunction::ReflectInnerX3, BoundaryFunction::ReflectOuterX3};
  static const SBValFunc soutflow[6] = {
      BoundaryFunction::SwarmOutflowInnerX1, BoundaryFunction::SwarmOutflowOuterX1,
      BoundaryFunction::SwarmOutflowInnerX2, BoundaryFunction::SwarmOutflowOuterX2,
      BoundaryFunction::SwarmOutflowInnerX3, BoundaryFunction::SwarmOutflowOuterX3};
  static const SBValFunc speriodic[6] = {
      BoundaryFunction::SwarmPeriodicInnerX1, BoundaryFunction::SwarmPeriodicOuterX1,
      BoundaryFunction::SwarmPeriodicInnerX2, BoundaryFunction::SwarmPeriodicOuterX2,
      BoundaryFunction::SwarmPeriodicInnerX3, BoundaryFunction::SwarmPeriodicOuterX3};

  for (int f = 0; f < BOUNDARY_NFACES; f++) {
    switch (boundary_conditions[f]) {
    case BoundaryFlag::reflect:
      MeshBndryFnctn[f] = reflect[f];
      break;
    case BoundaryFlag::outflow:
      MeshBndryFnctn[f] = outflow[f];
      SwarmBndryFnctn[f] = soutflow[f];
      break;
    case BoundaryFlag::user:
      if (app_in->boundary_conditions[f] != nullptr) {
        MeshBndryFnctn[f] = app_in->boundary_conditions[f];
      } else {
        std::stringstream msg;
        msg << "A user boundary condition for face " << f
            << " was requested. but no condition was enrolled." << std::endl;
        PARTHENON_THROW(msg);
      }
      break;
    default: // periodic/block BCs handled elsewhere.
      break;
    }

    switch (boundary_conditions[f]) {
    case BoundaryFlag::outflow:
      SwarmBndryFnctn[f] = soutflow[f];
      break;
    case BoundaryFlag::periodic:
      SwarmBndryFnctn[f] = speriodic[f];
      break;
    case BoundaryFlag::reflect:
      // Default "reflect" boundaries not available for swarms; catch later on if swarms
      // are present
      break;
    case BoundaryFlag::user:
      if (app_in->swarm_boundary_conditions[f] != nullptr) {
        // This is checked to be non-null later in Swarm::AllocateBoundaries, in case user
        // boundaries are requested but no swarms are used.
        SwarmBndryFnctn[f] = app_in->swarm_boundary_conditions[f];
      }
      break;
    default: // Default BCs handled elsewhere
      break;
    }
  }
}

} // namespace forest
} // namespace parthenon
