#ifndef MESH_REMESH_PLACEMENT_HPP_
#define MESH_REMESH_PLACEMENT_HPP_

#include <cstdint>
#include <vector>

#include "mesh/remesh/inventory.hpp"
#include "mesh/remesh/remesh_plan.hpp"

namespace parthenon::remesh {

OwnershipMap PlaceBlocksContiguously(const std::vector<double> &costs, int nranks);
OwnershipMap PlaceBlocksContiguously(const std::vector<double> &costs,
                                     const std::vector<MemoryFootprint> &footprints,
                                     int nranks, const MemoryBudget &budget);
void ValidateContiguousOwnership(const OwnershipMap &ownership);

std::vector<double> PlanRemeshCosts(const TopologyPlan &topology,
                                    const InventoryPlan &inventory,
                                    const std::vector<double> &old_costs,
                                    CostPolicy policy,
                                    const CostCallback &application_cost = nullptr);

PlacementPlan PlanRemeshPlacement(const TopologyPlan &topology,
                                  const InventoryPlan &inventory,
                                  const std::vector<double> &old_costs,
                                  const std::vector<std::uint64_t> &source_device_bytes,
                                  int nranks, CostPolicy policy,
                                  const MemoryBudget &budget = {},
                                  const CostCallback &application_cost = nullptr);

} // namespace parthenon::remesh

#endif // MESH_REMESH_PLACEMENT_HPP_
