#ifndef MESH_REMESH_REMESH_PLAN_HPP_
#define MESH_REMESH_REMESH_PLAN_HPP_

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "basic_types.hpp"
#include "mesh/domain.hpp"
#include "mesh/forest/forest.hpp"
#include "mesh/forest/logical_location.hpp"

namespace parthenon::remesh {

struct BlockKey {
  LogicalLocation loc;
};

struct EpochBlockId {
  std::uint64_t epoch;
  std::int64_t gid;
};

enum class Relation { unchanged, refined_child, coarsened_parent };

struct ChildOffsets {
  std::array<int, 3> value{0, 0, 0};
};

struct SourceContribution {
  EpochBlockId source;
  BlockKey source_key;
  int source_rank;
  ChildOffsets child;
};

struct PlannedLeaf {
  BlockKey key;
  std::int64_t new_gid;
  RegionSize geometry;
  Relation relation;
  std::vector<SourceContribution> sources;
};

struct TopologyPlan {
  std::optional<forest::Forest> candidate;
  std::vector<PlannedLeaf> leaves;
  std::vector<std::vector<std::int64_t>> successors;
  int added_blocks{0};
  int removed_blocks{0};
  bool changed{false};
};

enum class EstimateQuality { exact_array_request, conservative_bound, estimate };

struct MemoryFootprint {
  std::uint64_t host_owned{0};
  std::uint64_t device_owned{0};
  std::uint64_t execution_stage_reserve{0};
  std::uint64_t finalization_reserve{0};
  EstimateQuality quality{EstimateQuality::estimate};
};

using FieldSchemaId = std::uint32_t;
using SwarmSchemaId = std::uint32_t;

struct CounterPolicy {
  int allocation_count{0};
  int deallocation_count{0};
  bool inherited{false};
};

struct PlannedField {
  FieldSchemaId field;
  bool allocated_after_transfer{false};
  bool state_survives_remesh{false};
  bool reserved_for_finalization{false};
  std::uint64_t data_bytes{0};
  std::uint64_t coarse_bytes{0};
  CounterPolicy counters;
};

struct PlannedSwarm {
  SwarmSchemaId swarm;
  std::uint64_t active_count{0};
  std::uint64_t capacity{0};
};

struct BlockInventory {
  BlockKey key;
  int derefinement_count{0};
  std::vector<PlannedField> fields;
  std::vector<PlannedSwarm> swarms;
  MemoryFootprint footprint;
};

struct ParticleRoute {
  int source_index;
  std::int64_t destination_gid;
};

struct LocalSwarmRoutes {
  std::string swarm_name;
  std::int64_t source_gid;
  std::vector<ParticleRoute> particles;
};

struct OwnershipMap {
  std::vector<int> owner;
  std::vector<int> local_id;
  std::vector<std::vector<std::int64_t>> rank_leaves;
  std::vector<int> rank_starts;
  std::vector<int> rank_counts;
  std::vector<double> rank_costs;
};

struct TaggedLeaf {
  BlockKey key;
  AmrTag tag;
};

} // namespace parthenon::remesh

#endif // MESH_REMESH_REMESH_PLAN_HPP_
