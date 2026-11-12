#include "mesh/remesh/topology.hpp"

#include <algorithm>
#include <map>
#include <set>
#include <utility>

#include "utils/error_checking.hpp"

namespace parthenon::remesh {
namespace {

std::vector<LogicalLocation> GetLeaves(const forest::Forest &forest) {
  std::vector<LogicalLocation> leaves;
  for (const auto &tree : forest.GetTrees()) {
    auto tree_leaves = tree->GetSortedMeshBlockList();
    leaves.insert(leaves.end(), tree_leaves.begin(), tree_leaves.end());
  }
  std::sort(leaves.begin(), leaves.end(), [&](const auto &left, const auto &right) {
    return forest.GetGid(left) < forest.GetGid(right);
  });
  return leaves;
}

SourceContribution MakeSource(const forest::Forest &active,
                              const std::vector<int> &old_owners,
                              const LogicalLocation &source,
                              const LogicalLocation &destination,
                              std::uint64_t old_epoch) {
  const auto gid = active.GetGid(source);
  PARTHENON_REQUIRE(gid >= 0 && static_cast<std::size_t>(gid) < old_owners.size(),
                    "Remesh source GID has no owner");
  SourceContribution contribution{{old_epoch, gid}, {source}, old_owners[gid], {}};
  if (destination.level() > source.level()) {
    contribution.child.value = {static_cast<int>(destination.lx1() & 1),
                                static_cast<int>(destination.lx2() & 1),
                                static_cast<int>(destination.lx3() & 1)};
  }
  return contribution;
}

} // namespace

TopologyPlan PlanTopology(const forest::Forest &active,
                          const std::vector<TaggedLeaf> &tags,
                          const std::vector<int> &old_owners, int ndim,
                          std::uint64_t old_epoch) {
  PARTHENON_REQUIRE(ndim >= 1 && ndim <= 3, "Invalid remesh dimensionality");
  const auto old_leaves = GetLeaves(active);
  PARTHENON_REQUIRE(old_owners.size() == old_leaves.size(),
                    "Remesh owner list does not match the active topology");

  std::vector<LogicalLocation> refine;
  std::map<LogicalLocation, std::set<LogicalLocation>> derefine_by_parent;
  for (const auto &tagged : tags) {
    PARTHENON_REQUIRE(active.count(tagged.key.loc) == 1,
                      "Refinement tag refers to a non-leaf block");
    if (tagged.tag == AmrTag::refine) {
      refine.push_back(tagged.key.loc);
    } else if (tagged.tag == AmrTag::derefine) {
      derefine_by_parent[tagged.key.loc.GetParent()].insert(tagged.key.loc);
    }
  }
  if (refine.empty() && derefine_by_parent.empty()) return {};

  std::sort(refine.begin(), refine.end());
  refine.erase(std::unique(refine.begin(), refine.end()), refine.end());
  std::vector<LogicalLocation> derefine;
  for (const auto &[parent, tagged_children] : derefine_by_parent) {
    const auto daughters = parent.GetDaughters(ndim);
    if (tagged_children.size() != daughters.size()) continue;
    const bool complete =
        std::all_of(daughters.begin(), daughters.end(), [&](const auto &daughter) {
          return active.count(daughter) == 1 && tagged_children.count(daughter) == 1;
        });
    if (complete) derefine.push_back(parent);
  }
  std::sort(derefine.begin(), derefine.end(), [](const auto &left, const auto &right) {
    if (left.level() != right.level()) return left.level() > right.level();
    return left < right;
  });
  if (refine.empty() && derefine.empty()) return {};

  auto candidate = active.CloneForRemesh();
  for (const auto &loc : refine)
    candidate.Refine(loc);
  for (const auto &loc : derefine)
    candidate.Derefine(loc);
  auto new_leaves = candidate.GetMeshBlockListAndResolveGids();
  if (old_leaves == new_leaves) return {};

  TopologyPlan plan;
  plan.changed = true;
  plan.successors.resize(old_leaves.size());
  std::set<std::int64_t> refined_sources;
  std::set<std::int64_t> coarsened_sources;
  plan.leaves.reserve(new_leaves.size());
  for (const auto &destination : new_leaves) {
    PlannedLeaf leaf{{destination},
                     candidate.GetGid(destination),
                     candidate.GetBlockDomain(destination),
                     Relation::unchanged,
                     {}};
    if (active.count(destination) == 1) {
      leaf.sources.push_back(
          MakeSource(active, old_owners, destination, destination, old_epoch));
    } else {
      const auto parent = destination.GetParent();
      if (active.count(parent) == 1) {
        PARTHENON_REQUIRE(destination.level() == parent.level() + 1,
                          "Remesh supports one refinement generation per event");
        leaf.relation = Relation::refined_child;
        leaf.sources.push_back(
            MakeSource(active, old_owners, parent, destination, old_epoch));
        refined_sources.insert(leaf.sources.front().source.gid);
      } else {
        const auto daughters = destination.GetDaughters(ndim);
        PARTHENON_REQUIRE(std::all_of(daughters.begin(), daughters.end(),
                                      [&](const auto &daughter) {
                                        return active.count(daughter) == 1;
                                      }),
                          "New remesh leaf has no complete direct provenance");
        leaf.relation = Relation::coarsened_parent;
        for (const auto &daughter : daughters) {
          leaf.sources.push_back(
              MakeSource(active, old_owners, daughter, destination, old_epoch));
          coarsened_sources.insert(leaf.sources.back().source.gid);
        }
      }
    }
    for (const auto &source : leaf.sources) {
      plan.successors[source.source.gid].push_back(leaf.new_gid);
    }
    plan.leaves.push_back(std::move(leaf));
  }

  for (const auto &successors : plan.successors) {
    PARTHENON_REQUIRE(!successors.empty(), "Old remesh leaf has no successor");
  }
  const int daughter_count = 1 << ndim;
  plan.added_blocks = static_cast<int>(refined_sources.size()) * (daughter_count - 1);
  plan.removed_blocks =
      static_cast<int>(coarsened_sources.size() / daughter_count) * (daughter_count - 1);
  plan.candidate.emplace(std::move(candidate));
  return plan;
}

TopologyPlan PlanMigrationTopology(const forest::Forest &active,
                                   const std::vector<int> &old_owners,
                                   const std::uint64_t old_epoch) {
  const auto leaves = GetLeaves(active);
  PARTHENON_REQUIRE(old_owners.size() == leaves.size(),
                    "Remesh owner list does not match the active topology");

  TopologyPlan plan;
  plan.changed = true;
  plan.successors.resize(leaves.size());
  plan.leaves.reserve(leaves.size());
  for (const auto &loc : leaves) {
    const auto gid = active.GetGid(loc);
    plan.leaves.push_back({{loc},
                           gid,
                           active.GetBlockDomain(loc),
                           Relation::unchanged,
                           {MakeSource(active, old_owners, loc, loc, old_epoch)}});
    plan.successors[gid] = {gid};
  }
  return plan;
}

} // namespace parthenon::remesh
