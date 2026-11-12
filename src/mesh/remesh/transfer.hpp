#ifndef MESH_REMESH_TRANSFER_HPP_
#define MESH_REMESH_TRANSFER_HPP_

#include <cstdint>
#include <limits>
#include <vector>

#include "mesh/remesh/inventory.hpp"
#include "mesh/remesh/remesh_plan.hpp"

namespace parthenon {

class Mesh;
class MeshBlock;
using BlockList_t = std::vector<std::shared_ptr<MeshBlock>>;

namespace remesh {

FieldTransferPlan PlanFieldTransfers(
    const TopologyPlan &topology, const InventoryPlan &inventory,
    const std::vector<int> &old_ranks, const OwnershipMap &ownership,
    bool coalesce = true,
    std::uint64_t max_payload_count = std::numeric_limits<std::uint64_t>::max());

void ApplyPlannedFieldAllocations(const InventoryPlan &inventory,
                                  const BlockList_t &blocks, int first_gid);

void ExecuteFieldTransfers(const FieldTransferPlan &plan, const InventoryPlan &inventory,
                           const BlockList_t &old_blocks, int old_first_gid,
                           const BlockList_t &new_blocks, int new_first_gid, Mesh *mesh);

} // namespace remesh

} // namespace parthenon

#endif // MESH_REMESH_TRANSFER_HPP_
