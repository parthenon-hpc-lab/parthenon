#include <vector>

#include <catch2/catch.hpp>

#include "mesh/forest/forest.hpp"
#include "mesh/remesh/placement.hpp"
#include "mesh/remesh/topology.hpp"

using parthenon::AmrTag;
using parthenon::LogicalLocation;
using parthenon::forest::Forest;
using parthenon::forest::Tree;
using parthenon::remesh::Relation;
using parthenon::remesh::TaggedLeaf;

namespace {

Forest MakeResolvedForest() {
  Forest forest;
  forest.root_level = 0;
  forest.AddTree(Tree::create(0, 2, 0));
  forest.GetMeshBlockListAndResolveGids();
  return forest;
}

} // namespace

TEST_CASE("Remesh forest clones are isolated", "[forest][remesh]") {
  auto active = MakeResolvedForest();
  auto candidate = active.CloneForRemesh();

  REQUIRE(active.GetTreePtr(0).get() != candidate.GetTreePtr(0).get());
  candidate.Refine(LogicalLocation(0, 0, 0, 0, 0));
  REQUIRE(active.CountLeafMeshBlock() == 1);
  REQUIRE(candidate.CountLeafMeshBlock() == 4);
}

TEST_CASE("Topology plans preserve direct provenance", "[remesh]") {
  auto active = MakeResolvedForest();
  const LogicalLocation root(0, 0, 0, 0, 0);
  const std::vector<TaggedLeaf> refine_tags{{{root}, AmrTag::refine}};
  auto refine_plan = parthenon::remesh::PlanTopology(active, refine_tags, {0}, 2, 7);

  REQUIRE(refine_plan.changed);
  REQUIRE(refine_plan.added_blocks == 3);
  REQUIRE(refine_plan.removed_blocks == 0);
  REQUIRE(refine_plan.leaves.size() == 4);
  REQUIRE(refine_plan.successors.size() == 1);
  REQUIRE(refine_plan.successors[0].size() == 4);
  REQUIRE(active.CountLeafMeshBlock() == 1);
  for (const auto &leaf : refine_plan.leaves) {
    REQUIRE(leaf.relation == Relation::refined_child);
    REQUIRE(leaf.sources.size() == 1);
    REQUIRE(leaf.sources[0].source.epoch == 7);
    REQUIRE(leaf.sources[0].source.gid == 0);
  }

  auto refined = std::move(*refine_plan.candidate);
  std::vector<TaggedLeaf> derefine_tags;
  for (const auto &leaf : refine_plan.leaves) {
    derefine_tags.push_back({leaf.key, AmrTag::derefine});
  }
  auto derefine_plan =
      parthenon::remesh::PlanTopology(refined, derefine_tags, {0, 0, 0, 0}, 2);
  REQUIRE(derefine_plan.changed);
  REQUIRE(derefine_plan.added_blocks == 0);
  REQUIRE(derefine_plan.removed_blocks == 3);
  REQUIRE(derefine_plan.leaves.size() == 1);
  REQUIRE(derefine_plan.leaves[0].relation == Relation::coarsened_parent);
  REQUIRE(derefine_plan.leaves[0].sources.size() == 4);
}

TEST_CASE("Incomplete sibling permission does not derefine", "[remesh]") {
  auto active = MakeResolvedForest();
  active.Refine(LogicalLocation(0, 0, 0, 0, 0));
  const auto leaves = active.GetMeshBlockListAndResolveGids();
  const std::vector<TaggedLeaf> tags{{{leaves.front()}, AmrTag::derefine}};
  auto plan = parthenon::remesh::PlanTopology(active, tags, {0, 0, 0, 0}, 2);
  REQUIRE_FALSE(plan.changed);
  REQUIRE_FALSE(plan.candidate.has_value());
  REQUIRE(active.CountLeafMeshBlock() == 4);
}

TEST_CASE("Compatibility placement returns contiguous ownership", "[remesh]") {
  const auto ownership =
      parthenon::remesh::PlaceBlocksContiguously({1.0, 2.0, 1.0, 4.0}, 2);
  REQUIRE(ownership.owner == std::vector<int>{0, 0, 0, 1});
  REQUIRE(ownership.local_id == std::vector<int>{0, 1, 2, 0});
  REQUIRE(ownership.rank_starts == std::vector<int>{0, 3});
  REQUIRE(ownership.rank_counts == std::vector<int>{3, 1});
  REQUIRE(ownership.rank_costs == std::vector<double>{4.0, 4.0});
}
