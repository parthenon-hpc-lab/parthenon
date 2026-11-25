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

TEST_CASE("Compatibility placement permits idle leading ranks", "[remesh]") {
  const auto ownership = parthenon::remesh::PlaceBlocksContiguously({2.0}, 3);
  REQUIRE(ownership.owner == std::vector<int>{2});
  REQUIRE(ownership.rank_starts == std::vector<int>{0, 0, 0});
  REQUIRE(ownership.rank_counts == std::vector<int>{0, 0, 1});
}

TEST_CASE("Remesh cost policies use descriptor provenance", "[remesh][cost]") {
  using namespace parthenon::remesh;
  TopologyPlan topology;
  topology.changed = true;
  topology.successors.resize(4);
  topology.leaves = {{{}, 0, {}, Relation::unchanged, {{{0, 0}, {}, 0, {}}}},
                     {{}, 1, {}, Relation::refined_child, {{{0, 1}, {}, 0, {}}}},
                     {{},
                      2,
                      {},
                      Relation::coarsened_parent,
                      {{{0, 2}, {}, 0, {}}, {{0, 3}, {}, 0, {}}}}};
  InventoryPlan inventory;
  inventory.blocks.resize(3);
  inventory.blocks[1].footprint.device_owned = 3 * sizeof(parthenon::Real);
  inventory.blocks[1].swarms.push_back({0, 2, 2});

  REQUIRE(PlanRemeshCosts(topology, inventory, {2.0, 4.0, 6.0, 10.0},
                          CostPolicy::inherit) == std::vector<double>{2.0, 4.0, 8.0});
  const auto framework =
      PlanRemeshCosts(topology, inventory, {2.0, 4.0, 6.0, 10.0}, CostPolicy::parthenon);
  REQUIRE(framework[0] == 1.0);
  REQUIRE(framework[1] == 5.0);

  const auto application = PlanRemeshCosts(
      topology, inventory, {2.0, 4.0, 6.0, 10.0}, CostPolicy::application,
      [](const CostContext &context) {
        return CostEstimate{static_cast<double>(context.old_costs.size()), {}};
      });
  REQUIRE(application == std::vector<double>{1.0, 1.0, 2.0});
  REQUIRE_THROWS(
      PlanRemeshCosts(topology, inventory, {2.0, 4.0, 6.0, 10.0}, CostPolicy::application,
                      [](const CostContext &) { return CostEstimate{0.0, {}}; }));
}

TEST_CASE("Memory constrained placement is contiguous or rejects", "[remesh][budget]") {
  using namespace parthenon::remesh;
  std::vector<MemoryFootprint> footprints(3);
  for (auto &footprint : footprints) {
    footprint.device_owned = 6;
    footprint.quality = EstimateQuality::conservative_bound;
  }
  MemoryBudget budget;
  budget.device_bytes = 10;
  const auto ownership = PlaceBlocksContiguously({1.0, 3.0, 2.0}, footprints, 3, budget);
  REQUIRE(ownership.owner == std::vector<int>{0, 1, 2});
  REQUIRE(ownership.rank_device_bytes == std::vector<std::uint64_t>{6, 6, 6});
  REQUIRE_THROWS(PlaceBlocksContiguously({1.0, 3.0, 2.0}, footprints, 2, budget));

  footprints[0].quality = EstimateQuality::estimate;
  budget.require_hard_guarantee = true;
  REQUIRE_THROWS(PlaceBlocksContiguously({1.0, 3.0, 2.0}, footprints, 3, budget));
}

TEST_CASE("Remesh peak budget includes live sources", "[remesh][budget]") {
  using namespace parthenon::remesh;
  TopologyPlan topology;
  topology.changed = true;
  topology.successors.resize(1);
  topology.leaves = {{{}, 0, {}, Relation::unchanged, {{{0, 0}, {}, 0, {}}}}};
  InventoryPlan inventory;
  inventory.blocks.resize(1);
  inventory.blocks[0].footprint.device_owned = 6;
  inventory.blocks[0].footprint.quality = EstimateQuality::conservative_bound;
  MemoryBudget budget;
  budget.device_bytes = 15;
  REQUIRE_THROWS(PlanRemeshPlacement(topology, inventory, {1.0}, {5}, 1,
                                     CostPolicy::inherit, budget));
}
