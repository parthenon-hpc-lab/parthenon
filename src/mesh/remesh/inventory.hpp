#ifndef MESH_REMESH_INVENTORY_HPP_
#define MESH_REMESH_INVENTORY_HPP_

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "interface/metadata.hpp"
#include "interface/state_descriptor.hpp"
#include "interface/var_id.hpp"
#include "mesh/remesh/remesh_plan.hpp"

namespace parthenon {

class Mesh;

namespace remesh {

enum class ParticleValueType { real, integer, uint64 };

struct StorageLayout {
  std::array<int, MAX_VARIABLE_DIMENSION> data_dims{};
  std::array<int, MAX_VARIABLE_DIMENSION> coarse_dims{};
  std::uint64_t data_bytes{0};
  std::uint64_t coarse_bytes{0};
  std::uint64_t interior_count{0};
  std::array<std::uint64_t, 8> restricted_counts{};
};

struct FieldSchema {
  VarID id;
  Metadata metadata;
  StorageLayout layout;
  bool state_survives_remesh;
  std::string controller;
};

struct ParticleFieldSchema {
  std::string name;
  ParticleValueType type;
  std::uint64_t components;
};

struct SwarmSchema {
  std::string name;
  Metadata metadata;
  std::vector<ParticleFieldSchema> fields;
  std::uint64_t record_bytes{0};
};

struct InventoryPlan {
  std::vector<FieldSchema> field_schema;
  std::vector<SwarmSchema> swarm_schema;
  std::vector<BlockInventory> blocks;
  std::vector<LocalSwarmRoutes> local_swarm_routes;
};

StorageLayout GetStorageLayout(const Metadata &metadata,
                               const std::array<IndexShape, 3> &index_shapes,
                               bool multilevel);

std::uint64_t SelectSwarmCapacity(std::uint64_t active_count,
                                  std::uint64_t initial_reservation);

InventoryPlan PlanInventory(const Mesh &mesh, const TopologyPlan &topology);

} // namespace remesh
} // namespace parthenon

#endif // MESH_REMESH_INVENTORY_HPP_
