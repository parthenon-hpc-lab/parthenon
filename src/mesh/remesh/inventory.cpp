#include "mesh/remesh/inventory.hpp"

#include <algorithm>
#include <limits>
#include <unordered_map>
#include <utility>

#include "globals.hpp"
#include "mesh/index_shape.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "mesh/meshblock.hpp"
#include "mesh/swarm_amr_remesh.hpp"
#include "pack/default_names.hpp"
#include "parthenon_mpi.hpp"
#include "utils/error_checking.hpp"

namespace parthenon::remesh {

namespace {

std::uint64_t CheckedProduct(const std::array<int, MAX_VARIABLE_DIMENSION> &dims,
                             const std::uint64_t element_size) {
  std::uint64_t result = element_size;
  for (const int dim : dims) {
    PARTHENON_REQUIRE_THROWS(dim >= 0, "Negative remesh storage dimension");
    if (dim == 0) return 0;
    PARTHENON_REQUIRE_THROWS(result <= std::numeric_limits<std::uint64_t>::max() / dim,
                             "Remesh storage size exceeds uint64_t");
    result *= dim;
  }
  return result;
}

std::uint64_t CheckedMultiply(const std::uint64_t left, const std::uint64_t right) {
  PARTHENON_REQUIRE_THROWS(right == 0 ||
                               left <= std::numeric_limits<std::uint64_t>::max() / right,
                           "Remesh inventory size exceeds uint64_t");
  return left * right;
}

std::uint64_t CheckedAdd(const std::uint64_t left, const std::uint64_t right) {
  PARTHENON_REQUIRE_THROWS(left <= std::numeric_limits<std::uint64_t>::max() - right,
                           "Remesh inventory size exceeds uint64_t");
  return left + right;
}

std::uint64_t InteriorCount(const Metadata &metadata,
                            const std::array<IndexShape, 3> &index_shapes,
                            const std::array<int, MAX_VARIABLE_DIMENSION> &dims) {
  if (!metadata.IsMeshTied()) return CheckedProduct(dims, 1);
  const auto &bounds = metadata.IsSet(Metadata::Fine) ? index_shapes[1] : index_shapes[0];
  const auto elements = GetTopologicalElements(GetTopologicalType(metadata));
  const auto tensor_count = CheckedMultiply(
      CheckedMultiply(static_cast<std::uint64_t>(dims[3]), dims[4]), dims[5]);
  std::uint64_t count = 0;
  for (const auto element : elements) {
    const auto memory_element =
        metadata.IsSet(Metadata::CellMemAligned) ? TE::CC : element;
    const auto ib = bounds.GetBoundsI(IndexDomain::interior, memory_element);
    const auto jb = bounds.GetBoundsJ(IndexDomain::interior, memory_element);
    const auto kb = bounds.GetBoundsK(IndexDomain::interior, memory_element);
    auto element_count = CheckedMultiply(ib.e - ib.s + 1, jb.e - jb.s + 1);
    element_count = CheckedMultiply(element_count, kb.e - kb.s + 1);
    count = CheckedAdd(count, CheckedMultiply(element_count, tensor_count));
  }
  return count;
}

std::uint64_t RestrictedCount(const Metadata &metadata,
                              const std::array<IndexShape, 3> &index_shapes,
                              const std::array<int, MAX_VARIABLE_DIMENSION> &dims,
                              const int ox1, const int ox2, const int ox3) {
  if (!metadata.IsMeshTied()) return CheckedProduct(dims, 1);
  const auto &bounds = metadata.IsSet(Metadata::Fine) ? index_shapes[0] : index_shapes[2];
  const bool active_x2 = bounds.GetBoundsJ(IndexDomain::interior, TE::CC).e >
                         bounds.GetBoundsJ(IndexDomain::interior, TE::CC).s;
  const bool active_x3 = bounds.GetBoundsK(IndexDomain::interior, TE::CC).e >
                         bounds.GetBoundsK(IndexDomain::interior, TE::CC).s;
  const auto elements = GetTopologicalElements(GetTopologicalType(metadata));
  const auto tensor_count = CheckedMultiply(
      CheckedMultiply(static_cast<std::uint64_t>(dims[3]), dims[4]), dims[5]);
  std::uint64_t count = 0;
  for (const auto element : elements) {
    auto ib = bounds.GetBoundsI(IndexDomain::interior, element);
    auto jb = bounds.GetBoundsJ(IndexDomain::interior, element);
    auto kb = bounds.GetBoundsK(IndexDomain::interior, element);
    if (ox1 == 0) ib.e -= TopologicalOffsetI(element);
    if (ox2 == 0 && active_x2) jb.e -= TopologicalOffsetJ(element);
    if (ox3 == 0 && active_x3) kb.e -= TopologicalOffsetK(element);
    auto element_count = CheckedMultiply(ib.e - ib.s + 1, jb.e - jb.s + 1);
    element_count = CheckedMultiply(element_count, kb.e - kb.s + 1);
    count = CheckedAdd(count, CheckedMultiply(element_count, tensor_count));
  }
  return count;
}

int CheckedMpiCount(const std::size_t count) {
  PARTHENON_REQUIRE_THROWS(count <= std::numeric_limits<int>::max(),
                           "Remesh inventory MPI count exceeds INT_MAX");
  return static_cast<int>(count);
}

std::uint64_t ComponentCount(const Metadata &metadata) {
  std::uint64_t count = 1;
  for (const int dim : metadata.Shape()) {
    PARTHENON_REQUIRE_THROWS(dim >= 0, "Negative particle field dimension");
    PARTHENON_REQUIRE_THROWS(dim == 0 ||
                                 count <= std::numeric_limits<std::uint64_t>::max() / dim,
                             "Particle component count exceeds uint64_t");
    count *= dim;
  }
  return count;
}

ParticleValueType GetParticleValueType(const Metadata &metadata) {
  if (metadata.IsSet(Metadata::Integer)) return ParticleValueType::integer;
  if (metadata.IsSet(Metadata::UInt64)) return ParticleValueType::uint64;
  PARTHENON_REQUIRE_THROWS(metadata.IsSet(Metadata::Real),
                           "Unsupported particle field type in remesh schema");
  return ParticleValueType::real;
}

std::uint64_t ParticleValueBytes(const ParticleValueType type) {
  if (type == ParticleValueType::integer) return sizeof(int);
  if (type == ParticleValueType::uint64) return sizeof(std::uint64_t);
  return sizeof(Real);
}

std::vector<FieldSchema> BuildFieldSchema(StateDescriptor &packages,
                                          const std::array<IndexShape, 3> &index_shapes,
                                          const bool multilevel) {
  std::vector<std::pair<VarID, Metadata>> fields;
  fields.reserve(packages.AllFields().size());
  for (const auto &[id, metadata] : packages.AllFields())
    fields.emplace_back(id, metadata);
  std::sort(fields.begin(), fields.end(), [](const auto &left, const auto &right) {
    return std::tie(left.first.base_name, left.first.sparse_id) <
           std::tie(right.first.base_name, right.first.sparse_id);
  });

  std::vector<FieldSchema> schema;
  schema.reserve(fields.size());
  for (const auto &[id, metadata] : fields) {
    const bool survives = metadata.IsSet(Metadata::Independent) ||
                          metadata.IsSet(Metadata::FillGhost) ||
                          metadata.IsSet(Metadata::ForceRemeshComm);
    const auto controller =
        packages.ControlVariablesSet() && metadata.IsSet(Metadata::Sparse)
            ? packages.GetFieldController(id.label())
            : id.label();
    schema.push_back({id, metadata, GetStorageLayout(metadata, index_shapes, multilevel),
                      survives, controller});
  }
  return schema;
}

std::vector<SwarmSchema> BuildSwarmSchema(StateDescriptor &packages) {
  std::vector<std::string> names;
  names.reserve(packages.AllSwarms().size());
  for (const auto &[name, metadata] : packages.AllSwarms())
    names.push_back(name);
  std::sort(names.begin(), names.end());

  std::vector<SwarmSchema> schema;
  schema.reserve(names.size());
  for (const auto &name : names) {
    const auto &swarm_metadata = packages.AllSwarms().at(name);
    std::vector<std::pair<std::string, Metadata>> values;
    if (!swarm_metadata.IsSet(Metadata::NoPersistentParticleIds)) {
      values.emplace_back(swarm_position::id::name(), Metadata({Metadata::UInt64}));
    }
    values.emplace_back(swarm_position::x::name(), Metadata({Metadata::Real}));
    values.emplace_back(swarm_position::y::name(), Metadata({Metadata::Real}));
    values.emplace_back(swarm_position::z::name(), Metadata({Metadata::Real}));
    for (const auto &[value_name, metadata] : packages.AllSwarmValues(name)) {
      values.emplace_back(value_name, metadata);
    }
    std::sort(values.begin(), values.end(), [](const auto &left, const auto &right) {
      const auto left_type = GetParticleValueType(left.second);
      const auto right_type = GetParticleValueType(right.second);
      return std::tie(left_type, left.first) < std::tie(right_type, right.first);
    });

    SwarmSchema swarm{name, swarm_metadata, {}, 0};
    for (const auto &[value_name, metadata] : values) {
      const auto type = GetParticleValueType(metadata);
      const auto components = ComponentCount(metadata);
      swarm.fields.push_back({value_name, type, components});
      swarm.record_bytes = CheckedAdd(
          swarm.record_bytes, CheckedMultiply(components, ParticleValueBytes(type)));
    }
    schema.push_back(std::move(swarm));
  }
  return schema;
}

} // namespace

StorageLayout GetStorageLayout(const Metadata &metadata,
                               const std::array<IndexShape, 3> &index_shapes,
                               const bool multilevel) {
  StorageLayout layout;
  layout.data_dims = metadata.IsMeshTied() ? metadata.GetArrayDims(index_shapes, false)
                                           : metadata.GetArrayDims(nullptr, false);
  layout.coarse_dims = metadata.IsMeshTied() ? metadata.GetArrayDims(index_shapes, true)
                                             : metadata.GetArrayDims(nullptr, true);
  layout.data_bytes = CheckedProduct(layout.data_dims, sizeof(Real));
  layout.interior_count = InteriorCount(metadata, index_shapes, layout.data_dims);
  for (int ox3 = 0; ox3 <= 1; ++ox3) {
    for (int ox2 = 0; ox2 <= 1; ++ox2) {
      for (int ox1 = 0; ox1 <= 1; ++ox1) {
        layout.restricted_counts[ox1 + 2 * ox2 + 4 * ox3] =
            RestrictedCount(metadata, index_shapes, layout.coarse_dims, ox1, ox2, ox3);
      }
    }
  }
  const bool requires_coarse =
      metadata.IsSet(Metadata::FillGhost) || metadata.IsSet(Metadata::Independent) ||
      metadata.IsSet(Metadata::ForceRemeshComm) || metadata.IsSet(Metadata::Flux) ||
      metadata.IsSet(Metadata::GMGRestrict) || metadata.IsSet(Metadata::GMGProlongate);
  layout.coarse_bytes = multilevel && requires_coarse
                            ? CheckedProduct(layout.coarse_dims, sizeof(Real))
                            : 0;
  return layout;
}

std::uint64_t SelectSwarmCapacity(const std::uint64_t active_count,
                                  const std::uint64_t initial_reservation) {
  std::uint64_t capacity = std::max<std::uint64_t>(1, initial_reservation);
  while (capacity < active_count) {
    PARTHENON_REQUIRE_THROWS(capacity <= std::numeric_limits<std::uint64_t>::max() / 2,
                             "Swarm capacity exceeds uint64_t");
    capacity *= 2;
  }
  return capacity;
}

InventoryPlan PlanInventory(const Mesh &mesh, const TopologyPlan &topology) {
  PARTHENON_REQUIRE_THROWS(topology.changed, "Inventory requires a changed topology");
  auto packages = mesh.resolved_packages;
  const auto index_shapes = GetIndexShapes(mesh.GetDefaultBlockSize(), mesh.multilevel);
  InventoryPlan plan;
  plan.field_schema = BuildFieldSchema(*packages, index_shapes, mesh.multilevel);
  plan.swarm_schema = BuildSwarmSchema(*packages);

  const std::size_t old_count = topology.successors.size();
  const std::size_t field_count = plan.field_schema.size();
  std::vector<std::size_t> state_fields;
  for (std::size_t field = 0; field < field_count; ++field) {
    if (plan.field_schema[field].metadata.IsSet(Metadata::Sparse)) {
      state_fields.push_back(field);
    }
  }
  std::vector<std::size_t> variable_indices(field_count);
  if (!mesh.block_list.empty()) {
    const auto &variables = mesh.block_list.front()->meshblock_data.Get()->GetVariableVector();
    for (const auto field : state_fields) {
      const auto &label = plan.field_schema[field].id.label();
      const auto variable = std::find_if(
          variables.begin(), variables.end(), [&](const auto &candidate) {
            return candidate->label() == label;
          });
      PARTHENON_REQUIRE(variable != variables.end(),
                        "Remesh inventory field is absent from source block");
      variable_indices[field] = std::distance(variables.begin(), variable);
    }
  }
  std::vector<int> allocated(old_count * field_count, 0);
  std::vector<int> alloc_count(old_count * field_count, 0);
  std::vector<int> dealloc_count(old_count * field_count, 0);
  std::vector<int> derefinement_count(old_count, 0);
  for (std::size_t gid = 0; gid < old_count; ++gid) {
    for (std::size_t field = 0; field < field_count; ++field) {
      if (!plan.field_schema[field].metadata.IsSet(Metadata::Sparse)) {
        allocated[gid * field_count + field] = 1;
        alloc_count[gid * field_count + field] = 1;
      }
    }
  }
  for (const auto &block : mesh.block_list) {
    derefinement_count[block->gid] = block->pmr->DerefinementCount();
    const auto data = block->meshblock_data.Get();
    const auto &variables = data->GetVariableVector();
    for (const auto field : state_fields) {
      const auto &variable = variables[variable_indices[field]];
      const auto offset = block->gid * field_count + field;
      allocated[offset] = variable->IsAllocated();
      alloc_count[offset] = variable->GetAllocationStatus();
      dealloc_count[offset] = variable->dealloc_count;
    }
  }
#ifdef MPI_PARALLEL
  const auto record_size = CheckedMpiCount(1 + 3 * state_fields.size());
  std::vector<int> local_state(mesh.block_list.size() * record_size);
  for (std::size_t local_id = 0; local_id < mesh.block_list.size(); ++local_id) {
    const auto gid = mesh.block_list[local_id]->gid;
    const auto local_offset = local_id * record_size;
    local_state[local_offset] = derefinement_count[gid];
    for (std::size_t state = 0; state < state_fields.size(); ++state) {
      const auto field = state_fields[state];
      const auto source_offset = gid * field_count + field;
      const auto target_offset = local_offset + 1 + 3 * state;
      local_state[target_offset] = allocated[source_offset];
      local_state[target_offset + 1] = alloc_count[source_offset];
      local_state[target_offset + 2] = dealloc_count[source_offset];
    }
  }

  std::vector<int> global_state(old_count * record_size);
  std::vector<int> receive_counts(Globals::nranks);
  std::vector<int> displacements(Globals::nranks);
  const auto local_block_count = CheckedMpiCount(mesh.block_list.size());
  std::vector<int> rank_block_counts(Globals::nranks);
  PARTHENON_MPI_CHECK(MPI_Allgather(&local_block_count, 1, MPI_INT,
                                    rank_block_counts.data(), 1, MPI_INT,
                                    MPI_COMM_WORLD));
  std::size_t displacement = 0;
  for (int rank = 0; rank < Globals::nranks; ++rank) {
    displacements[rank] = CheckedMpiCount(displacement);
    receive_counts[rank] = CheckedMpiCount(
        static_cast<std::size_t>(rank_block_counts[rank]) * record_size);
    displacement += receive_counts[rank];
  }
  PARTHENON_REQUIRE(displacement == global_state.size(),
                    "Remesh inventory block counts disagree with topology");
  const auto local_gid_begin = displacements[Globals::my_rank] / record_size;
  for (std::size_t local_id = 0; local_id < mesh.block_list.size(); ++local_id) {
    PARTHENON_REQUIRE(mesh.block_list[local_id]->gid == local_gid_begin + local_id,
                      "Remesh inventory requires contiguous rank ownership");
  }
  PARTHENON_MPI_CHECK(MPI_Allgatherv(
      local_state.data(), CheckedMpiCount(local_state.size()), MPI_INT,
      global_state.data(), receive_counts.data(), displacements.data(), MPI_INT,
      MPI_COMM_WORLD));

  for (std::size_t gid = 0; gid < old_count; ++gid) {
    const auto global_offset = gid * record_size;
    derefinement_count[gid] = global_state[global_offset];
    for (std::size_t state = 0; state < state_fields.size(); ++state) {
      const auto field = state_fields[state];
      const auto source_offset = global_offset + 1 + 3 * state;
      const auto target_offset = gid * field_count + field;
      allocated[target_offset] = global_state[source_offset];
      alloc_count[target_offset] = global_state[source_offset + 1];
      dealloc_count[target_offset] = global_state[source_offset + 2];
    }
  }
#endif

  std::unordered_map<std::string, std::vector<std::size_t>> fields_by_controller;
  for (std::size_t field = 0; field < field_count; ++field) {
    fields_by_controller[plan.field_schema[field].controller].push_back(field);
  }

  plan.blocks.reserve(topology.leaves.size());
  for (const auto &leaf : topology.leaves) {
    BlockInventory block{leaf.key};
    if (leaf.relation == Relation::unchanged && leaf.sources.size() == 1) {
      block.derefinement_count = derefinement_count[leaf.sources.front().source.gid];
    }
    block.fields.reserve(field_count);
    for (std::size_t field = 0; field < field_count; ++field) {
      const auto &schema = plan.field_schema[field];
      bool exists = !schema.metadata.IsSet(Metadata::Sparse) ||
                    schema.metadata.IsSet(Metadata::ForceAllocOnNewBlocks);
      for (const auto &source : leaf.sources) {
        exists = exists || allocated[source.source.gid * field_count + field];
      }
      block.fields.push_back({static_cast<FieldSchemaId>(field),
                              exists,
                              schema.state_survives_remesh,
                              false,
                              exists ? schema.layout.data_bytes : 0,
                              exists ? schema.layout.coarse_bytes : 0,
                              {},
                              {}});
      block.fields.back().sources.reserve(leaf.sources.size());
      for (const auto &source : leaf.sources) {
        const auto offset = source.source.gid * field_count + field;
        block.fields.back().sources.push_back(
            {source.source.gid,
             static_cast<bool>(allocated[offset]),
             {alloc_count[offset], dealloc_count[offset], true}});
      }
      if (leaf.relation == Relation::unchanged && leaf.sources.size() == 1) {
        const auto offset = leaf.sources.front().source.gid * field_count + field;
        block.fields.back().counters = {alloc_count[offset], dealloc_count[offset], true};
      }
    }
    for (const auto &[controller, fields] : fields_by_controller) {
      const bool group_allocated =
          std::any_of(fields.begin(), fields.end(), [&](const auto field) {
            return block.fields[field].allocated_after_transfer;
          });
      if (group_allocated) {
        for (const auto field : fields)
          block.fields[field].allocated_after_transfer = true;
      }
    }
    for (auto &field : block.fields) {
      if (field.allocated_after_transfer && field.data_bytes == 0) {
        const auto &layout = plan.field_schema[field.field].layout;
        field.data_bytes = layout.data_bytes;
        field.coarse_bytes = layout.coarse_bytes;
      }
      if (leaf.relation == Relation::unchanged && leaf.sources.size() == 1) {
        const auto offset = leaf.sources.front().source.gid * field_count + field.field;
        field.counters.inherited =
            static_cast<bool>(allocated[offset]) == field.allocated_after_transfer;
      }
      block.footprint.device_owned = CheckedAdd(
          block.footprint.device_owned, CheckedAdd(field.data_bytes, field.coarse_bytes));
    }
    plan.blocks.push_back(std::move(block));
  }

  const std::size_t swarm_count = plan.swarm_schema.size();
  std::vector<std::uint64_t> target_counts(topology.leaves.size() * swarm_count, 0);
  for (const auto &block : mesh.block_list) {
    auto swarms = block->meshblock_data.Get()->GetSwarmData();
    for (std::size_t swarm_id = 0; swarm_id < swarm_count; ++swarm_id) {
      const auto &name = plan.swarm_schema[swarm_id].name;
      if (!swarms->Contains(name)) continue;
      auto swarm = swarms->Get(name);
      auto routes = ClassifySwarmParticlesForRemesh(swarm, &mesh, topology, block->gid);
      for (const auto &route : routes) {
        target_counts[route.destination_gid * swarm_count + swarm_id]++;
      }
      plan.local_swarm_routes.push_back({name, block->gid, std::move(routes)});
    }
  }
#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, target_counts.data(),
                                    CheckedMpiCount(target_counts.size()), MPI_UINT64_T,
                                    MPI_SUM, MPI_COMM_WORLD));
#endif
  for (std::size_t target = 0; target < plan.blocks.size(); ++target) {
    for (std::size_t swarm_id = 0; swarm_id < swarm_count; ++swarm_id) {
      const auto count = target_counts[target * swarm_count + swarm_id];
      const auto initial =
          plan.swarm_schema[swarm_id].metadata.InitialSwarmPoolReservation();
      plan.blocks[target].swarms.push_back({static_cast<SwarmSchemaId>(swarm_id), count,
                                            SelectSwarmCapacity(count, initial)});
      plan.blocks[target].footprint.device_owned =
          CheckedAdd(plan.blocks[target].footprint.device_owned,
                     CheckedMultiply(SelectSwarmCapacity(count, initial),
                                     plan.swarm_schema[swarm_id].record_bytes));
    }
    plan.blocks[target].footprint.quality = EstimateQuality::estimate;
  }
  return plan;
}

} // namespace parthenon::remesh
