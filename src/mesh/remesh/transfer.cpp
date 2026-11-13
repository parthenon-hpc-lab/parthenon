#include "mesh/remesh/transfer.hpp"

#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
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

std::uint64_t CheckedMultiply(const std::uint64_t left, const std::uint64_t right) {
  PARTHENON_REQUIRE_THROWS(right == 0 ||
                               left <= std::numeric_limits<std::uint64_t>::max() / right,
                           "Remesh transfer payload exceeds uint64_t");
  return left * right;
}

std::size_t CheckedBufferSize(const std::uint64_t count) {
  PARTHENON_REQUIRE_THROWS(
      count <= static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()),
      "Remesh transfer buffer exceeds size_t");
  return static_cast<std::size_t>(count);
}

void AppendBatchWaves(const PeerTransferBatch &batch,
                      const std::uint64_t max_payload_count,
                      std::vector<PeerTransferBatch> &waves) {
  std::vector<std::vector<FieldTransferRecord>> groups;
  std::map<std::uint64_t, std::size_t> group_by_offset;
  for (const auto &record : batch.records) {
    if (!record.source_allocated) {
      groups.push_back({record});
      continue;
    }
    const auto [entry, inserted] =
        group_by_offset.emplace(record.payload_offset, groups.size());
    if (inserted) groups.emplace_back();
    groups[entry->second].push_back(record);
  }

  PeerTransferBatch wave{batch.source_rank, batch.destination_rank};
  for (auto &group : groups) {
    const auto count = group.front().source_allocated ? group.front().payload_count : 0;
    PARTHENON_REQUIRE_THROWS(count <= max_payload_count,
                             "A remesh record exceeds remesh_max_payload_count");
    if (count > 0 && wave.payload_count > 0 &&
        CheckedAdd(wave.payload_count, count) > max_payload_count) {
      waves.push_back(std::move(wave));
      wave = {batch.source_rank, batch.destination_rank};
    }
    for (auto &record : group) {
      record.payload_offset = wave.payload_count;
      wave.records.push_back(std::move(record));
    }
    wave.payload_count = CheckedAdd(wave.payload_count, count);
  }
  if (!wave.records.empty()) waves.push_back(std::move(wave));
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

template <typename Buffer>
void TransferInterior(const std::shared_ptr<Variable<Real>> &variable, MeshBlock *block,
                      Buffer buffer, const bool pack) {
  auto data = variable->data;
  const int nt = data.GetDim(6);
  const int nu = data.GetDim(5);
  const int nv = data.GetDim(4);
  auto &bounds =
      variable->IsSet(Metadata::Fine) ? block->f_cellbounds : block->cellbounds;
  std::uint64_t offset = 0;
  for (const auto element : variable->GetTopologicalElements()) {
    const auto memory_element =
        variable->IsSet(Metadata::CellMemAligned) ? TE::CC : element;
    const auto ib = bounds.GetBoundsI(IndexDomain::interior, memory_element);
    const auto jb = bounds.GetBoundsJ(IndexDomain::interior, memory_element);
    const auto kb = bounds.GetBoundsK(IndexDomain::interior, memory_element);
    const int ni = ib.e - ib.s + 1;
    const int nj = jb.e - jb.s + 1;
    const int nk = kb.e - kb.s + 1;
    const auto element_offset = offset;
    const int element_index = static_cast<int>(element) % 3;
    parthenon::par_for(
        PARTHENON_AUTO_LABEL, 0, nt - 1, 0, nu - 1, 0, nv - 1, kb.s, kb.e, jb.s, jb.e,
        ib.s, ib.e,
        KOKKOS_LAMBDA(const int t, const int u, const int v, const int k, const int j,
                      const int i) {
          const std::uint64_t index =
              element_offset +
              (((((static_cast<std::uint64_t>(t) * nu + u) * nv + v) * nk + (k - kb.s)) *
                    nj +
                (j - jb.s)) *
                   ni +
               (i - ib.s));
          if (pack)
            buffer(index) = data(element_index, t, u, v, k, j, i);
          else
            data(element_index, t, u, v, k, j, i) = buffer(index);
        });
    auto count = CheckedMultiply(nt, nu);
    count = CheckedMultiply(count, nv);
    count = CheckedMultiply(count, nk);
    count = CheckedMultiply(count, nj);
    count = CheckedMultiply(count, ni);
    offset = CheckedAdd(offset, count);
  }
  PARTHENON_REQUIRE(offset == buffer.size(),
                    "Remesh interior payload size disagrees with field layout");
}

template <typename Buffer>
void TransferRestrictedBuffer(const LogicalLocation &fine_loc,
                              const std::shared_ptr<Variable<Real>> &variable,
                              MeshBlock *block, Buffer buffer, const bool pack) {
  auto coarse = variable->coarse_s;
  auto fine = variable->data;
  const int nt = fine.GetDim(6);
  const int nu = fine.GetDim(5);
  const int nv = fine.GetDim(4);
  auto &coarse_bounds =
      variable->IsSet(Metadata::Fine) ? block->cellbounds : block->c_cellbounds;
  const int ndim = block->pmy_mesh->ndim;
  const auto [ox1, ox2, ox3] = fine_loc.GetLocationInParent();
  std::uint64_t offset = 0;
  for (const auto element : variable->GetTopologicalElements()) {
    auto ib = coarse_bounds.GetBoundsI(IndexDomain::interior, element);
    auto jb = coarse_bounds.GetBoundsJ(IndexDomain::interior, element);
    auto kb = coarse_bounds.GetBoundsK(IndexDomain::interior, element);
    if (ox1 == 0) ib.e -= TopologicalOffsetI(element);
    if (ox2 == 0 && ndim > 1) jb.e -= TopologicalOffsetJ(element);
    if (ox3 == 0 && ndim > 2) kb.e -= TopologicalOffsetK(element);
    const int ni = ib.e - ib.s + 1;
    const int nj = jb.e - jb.s + 1;
    const int nk = kb.e - kb.s + 1;
    const int is = ox1 == 0 ? 0 : ni - TopologicalOffsetI(element);
    const int js = (ox2 == 0 || ndim < 2) ? 0 : nj - TopologicalOffsetJ(element);
    const int ks = (ox3 == 0 || ndim < 3) ? 0 : nk - TopologicalOffsetK(element);
    const auto element_offset = offset;
    const int element_index = static_cast<int>(element) % 3;
    parthenon::par_for(
        PARTHENON_AUTO_LABEL, 0, nt - 1, 0, nu - 1, 0, nv - 1, kb.s, kb.e, jb.s, jb.e,
        ib.s, ib.e,
        KOKKOS_LAMBDA(const int t, const int u, const int v, const int k, const int j,
                      const int i) {
          const std::uint64_t index =
              element_offset +
              (((((static_cast<std::uint64_t>(t) * nu + u) * nv + v) * nk + (k - kb.s)) *
                    nj +
                (j - jb.s)) *
                   ni +
               (i - ib.s));
          if (pack)
            buffer(index) = coarse(element_index, t, u, v, k, j, i);
          else
            fine(element_index, t, u, v, k + ks, j + js, i + is) = buffer(index);
        });
    auto count = CheckedMultiply(nt, nu);
    count = CheckedMultiply(count, nv);
    count = CheckedMultiply(count, nk);
    count = CheckedMultiply(count, nj);
    count = CheckedMultiply(count, ni);
    offset = CheckedAdd(offset, count);
  }
  PARTHENON_REQUIRE(offset == buffer.size(),
                    "Remesh restricted payload size disagrees with field layout");
}

void TransferRestrictedLocal(const LogicalLocation &fine_loc,
                             const std::shared_ptr<Variable<Real>> &source,
                             MeshBlock *source_block,
                             const std::shared_ptr<Variable<Real>> &target,
                             MeshBlock *target_block) {
  auto coarse = source->coarse_s;
  auto fine = target->data;
  const int nt = fine.GetDim(6);
  const int nu = fine.GetDim(5);
  const int nv = fine.GetDim(4);
  auto &coarse_bounds = source->IsSet(Metadata::Fine) ? source_block->cellbounds
                                                      : source_block->c_cellbounds;
  const int ndim = target_block->pmy_mesh->ndim;
  const auto [ox1, ox2, ox3] = fine_loc.GetLocationInParent();
  for (const auto element : source->GetTopologicalElements()) {
    auto ib = coarse_bounds.GetBoundsI(IndexDomain::interior, element);
    auto jb = coarse_bounds.GetBoundsJ(IndexDomain::interior, element);
    auto kb = coarse_bounds.GetBoundsK(IndexDomain::interior, element);
    if (ox1 == 0) ib.e -= TopologicalOffsetI(element);
    if (ox2 == 0 && ndim > 1) jb.e -= TopologicalOffsetJ(element);
    if (ox3 == 0 && ndim > 2) kb.e -= TopologicalOffsetK(element);
    const int ni = ib.e - ib.s + 1;
    const int nj = jb.e - jb.s + 1;
    const int nk = kb.e - kb.s + 1;
    const int is = ox1 == 0 ? 0 : ni - TopologicalOffsetI(element);
    const int js = (ox2 == 0 || ndim < 2) ? 0 : nj - TopologicalOffsetJ(element);
    const int ks = (ox3 == 0 || ndim < 3) ? 0 : nk - TopologicalOffsetK(element);
    const int element_index = static_cast<int>(element) % 3;
    parthenon::par_for(
        PARTHENON_AUTO_LABEL, 0, nt - 1, 0, nu - 1, 0, nv - 1, kb.s, kb.e, jb.s, jb.e,
        ib.s, ib.e,
        KOKKOS_LAMBDA(const int t, const int u, const int v, const int k, const int j,
                      const int i) {
          fine(element_index, t, u, v, k + ks, j + js, i + is) =
              coarse(element_index, t, u, v, k, j, i);
        });
  }
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
  if (record.operation == RemeshOperation::copy &&
      record.payload_mode == FieldPayloadMode::interior) {
    TransferInterior(variable, block.get(), source, false);
  } else if (record.operation == RemeshOperation::fine_to_coarse &&
             record.payload_mode == FieldPayloadMode::restricted) {
    const auto &child = record.child.value;
    const auto fine_loc = block->loc.GetDaughter(child[0], child[1], child[2]);
    TransferRestrictedBuffer(fine_loc, variable, block.get(), source, false);
  } else if (record.operation == RemeshOperation::copy ||
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
                                     const OwnershipMap &ownership, const bool coalesce,
                                     const std::uint64_t max_payload_count) {
  PARTHENON_REQUIRE_THROWS(topology.changed,
                           "Transfer planning requires changed topology");
  PARTHENON_REQUIRE_THROWS(topology.leaves.size() == inventory.blocks.size(),
                           "Transfer topology and inventory disagree");
  PARTHENON_REQUIRE_THROWS(topology.leaves.size() == ownership.owner.size(),
                           "Transfer topology and ownership disagree");
  PARTHENON_REQUIRE_THROWS(topology.successors.size() == old_ranks.size(),
                           "Transfer source ranks disagree with topology");
  PARTHENON_REQUIRE_THROWS(max_payload_count > 0,
                           "Remesh payload wave limit must be positive");

  std::map<std::pair<int, int>, PeerTransferBatch> batches;
  std::map<std::tuple<std::int64_t, int, FieldSchemaId>,
           std::pair<std::uint64_t, std::uint64_t>>
      coarse_to_fine_payloads;
  FieldTransferPlan plan;
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
        const auto &source = FindSource(leaf, source_gid);
        const auto payload_mode =
            operation == RemeshOperation::copy && schema.metadata.IsMeshTied()
                ? FieldPayloadMode::interior
            : operation == RemeshOperation::fine_to_coarse && schema.metadata.IsMeshTied()
                ? FieldPayloadMode::restricted
                : FieldPayloadMode::full;
        const auto full_count = source_state.allocated ? bytes / sizeof(Real) : 0;
        const auto &child = source.child.value;
        const int child_index = child[0] + 2 * child[1] + 4 * child[2];
        const std::uint64_t count = !source_state.allocated ? 0
                                    : payload_mode == FieldPayloadMode::interior
                                        ? schema.layout.interior_count
                                    : payload_mode == FieldPayloadMode::restricted
                                        ? schema.layout.restricted_counts[child_index]
                                        : full_count;
        plan.baseline_payload_count = CheckedAdd(plan.baseline_payload_count, full_count);
        plan.logical_payload_count = CheckedAdd(plan.logical_payload_count, count);
        auto &batch = batches[{source_rank, destination_rank}];
        batch.source_rank = source_rank;
        batch.destination_rank = destination_rank;
        const auto counters =
            operation == RemeshOperation::copy ? field.counters : source_state.counters;
        auto payload_offset = batch.payload_count;
        bool add_payload = source_state.allocated;
        if (coalesce && operation == RemeshOperation::coarse_to_fine && add_payload) {
          const auto key = std::make_tuple(source_gid, destination_rank, field.field);
          const auto previous = coarse_to_fine_payloads.find(key);
          if (previous == coarse_to_fine_payloads.end()) {
            coarse_to_fine_payloads.emplace(key, std::make_pair(payload_offset, count));
          } else {
            PARTHENON_REQUIRE_THROWS(previous->second.second == count,
                                     "Deduplicated C2F payload sizes disagree");
            payload_offset = previous->second.first;
            add_payload = false;
          }
        }
        batch.records.push_back({source_gid, destination_gid, source_rank,
                                 destination_rank, field.field, operation, source.child,
                                 source_state.allocated, counters, payload_offset,
                                 count});
        batch.records.back().payload_mode = payload_mode;
        if (add_payload) batch.payload_count = CheckedAdd(batch.payload_count, count);
      }
    }
  }

  if (coalesce) {
    for (const auto &[key, batch] : batches)
      AppendBatchWaves(batch, max_payload_count, plan.batches);
  } else {
    for (auto &[key, batch] : batches) {
      for (auto &record : batch.records) {
        PARTHENON_REQUIRE_THROWS(record.payload_count <= max_payload_count,
                                 "A remesh record exceeds remesh_max_payload_count");
        record.payload_offset = 0;
        plan.batches.push_back({batch.source_rank,
                                batch.destination_rank,
                                {std::move(record)},
                                record.payload_count});
      }
    }
  }
  for (const auto &batch : plan.batches) {
    plan.payload_count = CheckedAdd(plan.payload_count, batch.payload_count);
    plan.peak_batch_count = std::max(plan.peak_batch_count, batch.payload_count);
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
#ifdef MPI_PARALLEL
  MPI_Comm communicator;
  PARTHENON_MPI_CHECK(MPI_Comm_dup(MPI_COMM_WORLD, &communicator));
  constexpr int payload_tag = 0;
#endif

  for (const auto &batch : plan.batches) {
    if (batch.source_rank == batch.destination_rank) {
      if (batch.source_rank != Globals::my_rank) continue;
      for (const auto &record : batch.records) {
        if (!record.source_allocated) {
          ApplyAbsentRecord(record, inventory, new_blocks, new_first_gid);
          continue;
        }
        auto block = FindLocalBlock(old_blocks, old_first_gid, record.source_gid);
        auto variable = GetField(block, inventory, record.field);
        if (record.payload_mode == FieldPayloadMode::restricted) {
          auto target_block =
              FindLocalBlock(new_blocks, new_first_gid, record.destination_gid);
          auto target = GetField(target_block, inventory, record.field);
          if (!target->IsAllocated()) target_block->AllocateSparse(target->label());
          const auto &child = record.child.value;
          const auto fine_loc =
              target_block->loc.GetDaughter(child[0], child[1], child[2]);
          TransferRestrictedLocal(fine_loc, variable, block.get(), target,
                                  target_block.get());
        } else if (record.operation == RemeshOperation::coarse_to_fine) {
          auto target_block =
              FindLocalBlock(new_blocks, new_first_gid, record.destination_gid);
          auto target = GetField(target_block, inventory, record.field);
          if (!target->IsAllocated()) target_block->AllocateSparse(target->label());
          ExtractCoarseChild(target_block->loc, variable, target, target_block.get());
        } else if (record.operation == RemeshOperation::fine_to_coarse) {
          auto source = Flatten(variable->coarse_s);
          ApplyAllocatedRecord(record, inventory, source, new_blocks, new_first_gid);
        } else {
          auto source = Flatten(variable->data);
          ApplyAllocatedRecord(record, inventory, source, new_blocks, new_first_gid);
        }
      }
      continue;
    }

#ifdef MPI_PARALLEL
    buf_pool_t<Real>::owner_t buffer;
    if (batch.payload_count > 0 && (batch.source_rank == Globals::my_rank ||
                                    batch.destination_rank == Globals::my_rank)) {
      const auto size = CheckedBufferSize(batch.payload_count);
      if (!mesh->pool_map.Contains(size)) mesh->pool_map.AddPool(size, 1);
      buffer = mesh->pool_map.GetOwningBuffer(size);
    }

    std::vector<mpi_request_t> receive_requests;
    if (batch.destination_rank == Globals::my_rank && batch.payload_count > 0) {
      peer_transport::PostReceives(buffer.data(), batch.payload_count, batch.source_rank,
                                   payload_tag, communicator, receive_requests);
    }

    if (batch.source_rank == Globals::my_rank && batch.payload_count > 0) {
      std::set<std::uint64_t> packed_offsets;
      for (const auto &record : batch.records) {
        if (!record.source_allocated ||
            !packed_offsets.insert(record.payload_offset).second)
          continue;
        auto block = FindLocalBlock(old_blocks, old_first_gid, record.source_gid);
        auto variable = GetField(block, inventory, record.field);
        PARTHENON_REQUIRE(variable->IsAllocated(),
                          "Remesh manifest disagrees with source allocation");
        auto target = Kokkos::subview(
            buffer, std::make_pair(static_cast<std::size_t>(record.payload_offset),
                                   static_cast<std::size_t>(record.payload_offset +
                                                            record.payload_count)));
        if (record.payload_mode == FieldPayloadMode::interior) {
          TransferInterior(variable, block.get(), target, true);
          continue;
        }
        if (record.payload_mode == FieldPayloadMode::restricted) {
          TransferRestrictedBuffer(block->loc, variable, block.get(), target, true);
          continue;
        }
        auto source = record.operation == RemeshOperation::fine_to_coarse
                          ? Flatten(variable->coarse_s)
                          : Flatten(variable->data);
        PARTHENON_REQUIRE(source.size() == record.payload_count,
                          "Remesh manifest disagrees with source payload size");
        Kokkos::deep_copy(target, source);
      }
      Kokkos::fence();
    }

    std::vector<mpi_request_t> send_requests;
    if (batch.source_rank == Globals::my_rank && batch.payload_count > 0) {
      peer_transport::PostSends(buffer.data(), batch.payload_count,
                                batch.destination_rank, payload_tag, communicator,
                                send_requests);
    }
    WaitAll(receive_requests);
    if (batch.destination_rank == Globals::my_rank) {
      ApplyBatch(batch, buffer, inventory, new_blocks, new_first_gid);
      Kokkos::fence();
    }
    WaitAll(send_requests);
#else
    PARTHENON_FAIL("Non-MPI remesh plan contains a remote field transfer");
#endif
  }

#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(MPI_Comm_free(&communicator));
#endif
}

} // namespace parthenon::remesh
