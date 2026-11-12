#include "mesh/remesh/transfer.hpp"

#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "bvals/comms/bnd_info.hpp"
#include "globals.hpp"
#include "interface/variable.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "mesh/meshblock.hpp"
#include "parthenon_arrays.hpp"
#include "utils/error_checking.hpp"
#include "utils/mpi_types.hpp"
#include "utils/peer_transport.hpp"

namespace parthenon::remesh {

namespace {

std::uint64_t CheckedAdd(const std::uint64_t left, const std::uint64_t right) {
  PARTHENON_REQUIRE_THROWS(left <= std::numeric_limits<std::uint64_t>::max() - right,
                           "Remesh transfer payload exceeds uint64_t");
  return left + right;
}

std::size_t CheckedBufferSize(const std::uint64_t count) {
  PARTHENON_REQUIRE_THROWS(
      count <= static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()),
      "Remesh transfer buffer exceeds size_t");
  return static_cast<std::size_t>(count);
}

const SourceContribution &FindSource(const PlannedLeaf &leaf,
                                     const std::int64_t source_gid) {
  for (const auto &source : leaf.sources) {
    if (source.source.gid == source_gid) return source;
  }
  PARTHENON_THROW("Remesh inventory source is absent from topology provenance");
}

std::shared_ptr<MeshBlock> FindLocalBlock(const BlockList_t &blocks, const int first_gid,
                                          const std::int64_t gid) {
  PARTHENON_REQUIRE(gid >= first_gid && gid < first_gid + blocks.size(),
                    "Remesh transfer references a nonlocal block");
  return blocks[gid - first_gid];
}

std::shared_ptr<Variable<Real>> GetField(const std::shared_ptr<MeshBlock> &block,
                                         const InventoryPlan &inventory,
                                         const FieldSchemaId field) {
  PARTHENON_REQUIRE(field < inventory.field_schema.size(),
                    "Remesh transfer field schema ID is invalid");
  return block->meshblock_data.Get()->GetVarPtr(inventory.field_schema[field].id.label());
}

template <typename Array>
auto Flatten(Array &array) {
  using View = Kokkos::View<Real *, LayoutWrapper, DevMemSpace,
                            Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  return View(array.data(), array.size());
}

void ExtractCoarseChild(const LogicalLocation &fine_loc,
                        const std::shared_ptr<Variable<Real>> &source,
                        const std::shared_ptr<Variable<Real>> &target,
                        MeshBlock *block) {
  auto fine = source->data;
  auto coarse = target->coarse_s;
  const int nt = fine.GetDim(6) - 1;
  const int nu = fine.GetDim(5) - 1;
  const int nv = fine.GetDim(4) - 1;
  auto &cellbounds = target->IsSet(Metadata::Fine) ? block->f_cellbounds
                                                   : block->cellbounds;
  auto &coarse_bounds =
      target->IsSet(Metadata::Fine) ? block->cellbounds : block->c_cellbounds;
  for (const auto element : target->GetTopologicalElements()) {
    const auto entire_element =
        target->IsSet(Metadata::CellMemAligned) ? TE::CC : element;
    const auto ib = coarse_bounds.GetBoundsI(IndexDomain::entire, entire_element);
    const auto jb = coarse_bounds.GetBoundsJ(IndexDomain::entire, entire_element);
    const auto kb = coarse_bounds.GetBoundsK(IndexDomain::entire, entire_element);
    const auto ib_int = cellbounds.GetBoundsI(IndexDomain::interior, element);
    const auto jb_int = cellbounds.GetBoundsJ(IndexDomain::interior, element);
    const auto kb_int = cellbounds.GetBoundsK(IndexDomain::interior, element);
    const auto [ox1, ox2, ox3] = fine_loc.GetLocationInParent();
    const int ks = ox3 == 0 ? 0 : (kb_int.e - kb_int.s + 1) / 2;
    const int js = ox2 == 0 ? 0 : (jb_int.e - jb_int.s + 1) / 2;
    const int is = ox1 == 0 ? 0 : (ib_int.e - ib_int.s + 1) / 2;
    const int element_index = static_cast<int>(element) % 3;
    parthenon::par_for(
        PARTHENON_AUTO_LABEL, 0, nt, 0, nu, 0, nv, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int t, const int u, const int v, const int k, const int j,
                      const int i) {
          coarse(element_index, t, u, v, k, j, i) =
              fine(element_index, t, u, v, k + ks, j + js, i + is);
        });
  }
}

void InjectFineChild(const LogicalLocation &fine_loc,
                     const std::shared_ptr<Variable<Real>> &source,
                     const std::shared_ptr<Variable<Real>> &target, MeshBlock *block) {
  const int ndim = block->pmy_mesh->ndim;
  auto coarse = source->coarse_s;
  auto fine = target->data;
  const int nt = fine.GetDim(6) - 1;
  const int nu = fine.GetDim(5) - 1;
  const int nv = fine.GetDim(4) - 1;
  auto &coarse_bounds =
      target->IsSet(Metadata::Fine) ? block->cellbounds : block->c_cellbounds;
  for (const auto element : target->GetTopologicalElements()) {
    auto ib = coarse_bounds.GetBoundsI(IndexDomain::interior, element);
    auto jb = coarse_bounds.GetBoundsJ(IndexDomain::interior, element);
    auto kb = coarse_bounds.GetBoundsK(IndexDomain::interior, element);
    const auto [ox1, ox2, ox3] = fine_loc.GetLocationInParent();
    if (ox3 == 0 && ndim > 2) kb.e -= TopologicalOffsetK(element);
    if (ox2 == 0 && ndim > 1) jb.e -= TopologicalOffsetJ(element);
    if (ox1 == 0) ib.e -= TopologicalOffsetI(element);
    const int ks =
        (ox3 == 0 || ndim < 3) ? 0 : kb.e - kb.s + 1 - TopologicalOffsetK(element);
    const int js =
        (ox2 == 0 || ndim < 2) ? 0 : jb.e - jb.s + 1 - TopologicalOffsetJ(element);
    const int is = ox1 == 0 ? 0 : ib.e - ib.s + 1 - TopologicalOffsetI(element);
    const int element_index = static_cast<int>(element) % 3;
    parthenon::par_for(
        PARTHENON_AUTO_LABEL, 0, nt, 0, nu, 0, nv, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int t, const int u, const int v, const int k, const int j,
                      const int i) {
          fine(element_index, t, u, v, k + ks, j + js, i + is) =
              coarse(element_index, t, u, v, k, j, i);
        });
  }
  Kokkos::fence();
}

template <typename SourceView>
void ApplyAllocatedRecord(const FieldTransferRecord &record,
                          const InventoryPlan &inventory, const SourceView &source,
                          const BlockList_t &new_blocks, const int new_first_gid) {
  auto block = FindLocalBlock(new_blocks, new_first_gid, record.destination_gid);
  auto variable = GetField(block, inventory, record.field);
  if (!variable->IsAllocated()) block->AllocateSparse(variable->label());
  if (record.operation == RemeshOperation::copy ||
      record.operation == RemeshOperation::coarse_to_fine) {
    PARTHENON_REQUIRE(variable->data.size() == record.payload_count,
                      "Remesh data payload size disagrees with target field");
    auto target = Flatten(variable->data);
    Kokkos::deep_copy(target, source);
  } else {
    PARTHENON_REQUIRE(variable->coarse_s.size() == record.payload_count,
                      "Remesh coarse payload size disagrees with target field");
    auto target = Flatten(variable->coarse_s);
    Kokkos::deep_copy(target, source);
  }

  if (record.operation == RemeshOperation::copy) {
    if (record.source_counters.inherited) {
      variable->dealloc_count = record.source_counters.deallocation_count;
      variable->RestoreAllocationStatusForRemesh(record.source_counters.allocation_count);
    }
    block->pmr->DerefinementCount() =
        inventory.blocks[record.destination_gid].derefinement_count;
  } else if (record.operation == RemeshOperation::coarse_to_fine) {
    ExtractCoarseChild(block->loc, variable, variable, block.get());
  } else {
    const auto &source = record.child.value;
    const auto fine_loc = block->loc.GetDaughter(source[0], source[1], source[2]);
    InjectFineChild(fine_loc, variable, variable, block.get());
  }
}

void ApplyAbsentRecord(const FieldTransferRecord &record, const InventoryPlan &inventory,
                       const BlockList_t &new_blocks, const int new_first_gid) {
  if (record.operation != RemeshOperation::copy) return;
  auto block = FindLocalBlock(new_blocks, new_first_gid, record.destination_gid);
  auto variable = GetField(block, inventory, record.field);
  if (record.source_counters.inherited)
    variable->dealloc_count = record.source_counters.deallocation_count;
  block->pmr->DerefinementCount() =
      inventory.blocks[record.destination_gid].derefinement_count;
}

template <typename Buffer>
void ApplyBatch(const PeerTransferBatch &batch, const Buffer &buffer,
                const InventoryPlan &inventory, const BlockList_t &new_blocks,
                const int new_first_gid) {
  for (const auto &record : batch.records) {
    if (!record.source_allocated) {
      ApplyAbsentRecord(record, inventory, new_blocks, new_first_gid);
      continue;
    }
    auto source = Kokkos::subview(
        buffer, std::make_pair(static_cast<std::size_t>(record.payload_offset),
                               static_cast<std::size_t>(record.payload_offset +
                                                        record.payload_count)));
    ApplyAllocatedRecord(record, inventory, source, new_blocks, new_first_gid);
  }
}

} // namespace

FieldTransferPlan PlanFieldTransfers(const TopologyPlan &topology,
                                     const InventoryPlan &inventory,
                                     const std::vector<int> &old_ranks,
                                     const OwnershipMap &ownership, const bool coalesce) {
  PARTHENON_REQUIRE_THROWS(topology.changed,
                           "Transfer planning requires changed topology");
  PARTHENON_REQUIRE_THROWS(topology.leaves.size() == inventory.blocks.size(),
                           "Transfer topology and inventory disagree");
  PARTHENON_REQUIRE_THROWS(topology.leaves.size() == ownership.owner.size(),
                           "Transfer topology and ownership disagree");
  PARTHENON_REQUIRE_THROWS(topology.successors.size() == old_ranks.size(),
                           "Transfer source ranks disagree with topology");

  std::map<std::pair<int, int>, PeerTransferBatch> batches;
  for (const auto &leaf : topology.leaves) {
    const auto destination_gid = leaf.new_gid;
    const auto destination_rank = ownership.owner[destination_gid];
    const auto &block = inventory.blocks[destination_gid];
    PARTHENON_REQUIRE_THROWS(block.fields.size() == inventory.field_schema.size(),
                             "Transfer inventory has incomplete field schema");
    for (const auto &field : block.fields) {
      if (!field.state_survives_remesh) continue;
      const auto &schema = inventory.field_schema[field.field];
      for (const auto &source_state : field.sources) {
        const auto source_gid = source_state.source_gid;
        PARTHENON_REQUIRE_THROWS(source_gid >= 0 && source_gid < old_ranks.size(),
                                 "Transfer source GID is invalid");
        const auto source_rank = old_ranks[source_gid];
        if (leaf.relation == Relation::unchanged && source_rank == destination_rank)
          continue;

        const auto operation = leaf.relation == Relation::unchanged
                                   ? RemeshOperation::copy
                                   : (leaf.relation == Relation::refined_child
                                          ? RemeshOperation::coarse_to_fine
                                          : RemeshOperation::fine_to_coarse);
        const auto bytes = operation == RemeshOperation::fine_to_coarse
                               ? schema.layout.coarse_bytes
                               : schema.layout.data_bytes;
        PARTHENON_REQUIRE_THROWS(bytes % sizeof(Real) == 0,
                                 "Field transfer payload is not Real-aligned");
        const std::uint64_t count = source_state.allocated ? bytes / sizeof(Real) : 0;
        auto &batch = batches[{source_rank, destination_rank}];
        batch.source_rank = source_rank;
        batch.destination_rank = destination_rank;
        const auto &source = FindSource(leaf, source_gid);
        const auto counters =
            operation == RemeshOperation::copy ? field.counters : source_state.counters;
        batch.records.push_back({source_gid, destination_gid, source_rank,
                                 destination_rank, field.field, operation, source.child,
                                 source_state.allocated, counters, batch.payload_count,
                                 count});
        batch.payload_count = CheckedAdd(batch.payload_count, count);
      }
    }
  }

  FieldTransferPlan plan;
  if (coalesce) {
    plan.batches.reserve(batches.size());
    for (auto &[key, batch] : batches)
      plan.batches.push_back(std::move(batch));
  } else {
    for (auto &[key, batch] : batches) {
      for (auto &record : batch.records) {
        record.payload_offset = 0;
        plan.batches.push_back({batch.source_rank,
                                batch.destination_rank,
                                {std::move(record)},
                                record.payload_count});
      }
    }
  }
  return plan;
}

void ApplyPlannedFieldAllocations(const InventoryPlan &inventory,
                                  const BlockList_t &blocks, const int first_gid) {
  for (std::size_t local_id = 0; local_id < blocks.size(); ++local_id) {
    const auto gid = first_gid + local_id;
    PARTHENON_REQUIRE(gid >= 0 && gid < inventory.blocks.size(),
                      "Remesh target block is absent from inventory");
  }
}

void ExecuteFieldTransfers(const FieldTransferPlan &plan, const InventoryPlan &inventory,
                           const BlockList_t &old_blocks, const int old_first_gid,
                           const BlockList_t &new_blocks, const int new_first_gid,
                           Mesh *mesh) {
  PARTHENON_REQUIRE(mesh != nullptr, "Remesh field transfer requires a mesh");
  std::vector<buf_pool_t<Real>::owner_t> buffers(plan.batches.size());
#ifdef MPI_PARALLEL
  MPI_Comm communicator;
  PARTHENON_MPI_CHECK(MPI_Comm_dup(MPI_COMM_WORLD, &communicator));
  std::vector<mpi_request_t> receive_requests;
  std::vector<mpi_request_t> send_requests;
  constexpr int payload_tag = 0;
#endif

  for (std::size_t batch_id = 0; batch_id < plan.batches.size(); ++batch_id) {
    const auto &batch = plan.batches[batch_id];
    if (batch.source_rank == batch.destination_rank ||
        batch.destination_rank != Globals::my_rank || batch.payload_count == 0)
      continue;
    const auto size = CheckedBufferSize(batch.payload_count);
    if (!mesh->pool_map.Contains(size)) mesh->pool_map.AddPool(size, 1);
    buffers[batch_id] = mesh->pool_map.GetOwningBuffer(size);
#ifdef MPI_PARALLEL
    peer_transport::PostReceives(buffers[batch_id].data(), batch.payload_count,
                                 batch.source_rank, payload_tag, communicator,
                                 receive_requests);
#endif
  }

  for (std::size_t batch_id = 0; batch_id < plan.batches.size(); ++batch_id) {
    const auto &batch = plan.batches[batch_id];
    if (batch.source_rank != Globals::my_rank ||
        batch.source_rank == batch.destination_rank || batch.payload_count == 0)
      continue;
    const auto size = CheckedBufferSize(batch.payload_count);
    if (!mesh->pool_map.Contains(size)) mesh->pool_map.AddPool(size, 1);
    buffers[batch_id] = mesh->pool_map.GetOwningBuffer(size);
    for (const auto &record : batch.records) {
      if (!record.source_allocated) continue;
      auto block = FindLocalBlock(old_blocks, old_first_gid, record.source_gid);
      auto variable = GetField(block, inventory, record.field);
      PARTHENON_REQUIRE(variable->IsAllocated(),
                        "Remesh manifest disagrees with source allocation");
      auto source = record.operation == RemeshOperation::fine_to_coarse
                        ? Flatten(variable->coarse_s)
                        : Flatten(variable->data);
      PARTHENON_REQUIRE(source.size() == record.payload_count,
                        "Remesh manifest disagrees with source payload size");
      auto target =
          Kokkos::subview(buffers[batch_id],
                          std::make_pair(static_cast<std::size_t>(record.payload_offset),
                                         static_cast<std::size_t>(record.payload_offset +
                                                                  record.payload_count)));
      Kokkos::deep_copy(target, source);
    }
  }
  Kokkos::fence();

#ifdef MPI_PARALLEL
  for (std::size_t batch_id = 0; batch_id < plan.batches.size(); ++batch_id) {
    const auto &batch = plan.batches[batch_id];
    if (batch.source_rank != Globals::my_rank ||
        batch.source_rank == batch.destination_rank || batch.payload_count == 0)
      continue;
    peer_transport::PostSends(buffers[batch_id].data(), batch.payload_count,
                              batch.destination_rank, payload_tag, communicator,
                              send_requests);
  }
  WaitAll(receive_requests);
#endif

  for (std::size_t batch_id = 0; batch_id < plan.batches.size(); ++batch_id) {
    const auto &batch = plan.batches[batch_id];
    if (batch.destination_rank != Globals::my_rank ||
        batch.source_rank == batch.destination_rank)
      continue;
    ApplyBatch(batch, buffers[batch_id], inventory, new_blocks, new_first_gid);
  }

  for (const auto &batch : plan.batches) {
    if (batch.source_rank != Globals::my_rank ||
        batch.destination_rank != Globals::my_rank)
      continue;
    for (const auto &record : batch.records) {
      if (!record.source_allocated) {
        ApplyAbsentRecord(record, inventory, new_blocks, new_first_gid);
        continue;
      }
      auto block = FindLocalBlock(old_blocks, old_first_gid, record.source_gid);
      auto variable = GetField(block, inventory, record.field);
      if (record.operation == RemeshOperation::coarse_to_fine) {
        auto target_block =
            FindLocalBlock(new_blocks, new_first_gid, record.destination_gid);
        auto target = GetField(target_block, inventory, record.field);
        if (!target->IsAllocated()) target_block->AllocateSparse(target->label());
        ExtractCoarseChild(target_block->loc, variable, target, target_block.get());
      } else if (record.operation == RemeshOperation::fine_to_coarse) {
        auto target_block =
            FindLocalBlock(new_blocks, new_first_gid, record.destination_gid);
        auto target = GetField(target_block, inventory, record.field);
        if (!target->IsAllocated()) target_block->AllocateSparse(target->label());
        const auto &child = record.child.value;
        const auto fine_loc =
            target_block->loc.GetDaughter(child[0], child[1], child[2]);
        InjectFineChild(fine_loc, variable, target, target_block.get());
      } else {
        auto source = Flatten(variable->data);
        ApplyAllocatedRecord(record, inventory, source, new_blocks, new_first_gid);
      }
    }
  }

#ifdef MPI_PARALLEL
  WaitAll(send_requests);
  PARTHENON_MPI_CHECK(MPI_Comm_free(&communicator));
#endif
}

} // namespace parthenon::remesh
