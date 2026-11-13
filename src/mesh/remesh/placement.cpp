#include "mesh/remesh/placement.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

#include "utils/error_checking.hpp"

namespace parthenon::remesh {

namespace {

std::uint64_t CheckedAdd(const std::uint64_t left, const std::uint64_t right) {
  PARTHENON_REQUIRE_THROWS(left <= std::numeric_limits<std::uint64_t>::max() - right,
                           "Remesh memory accounting exceeds uint64_t");
  return left + right;
}

void ValidateCosts(const std::vector<double> &costs) {
  PARTHENON_REQUIRE_THROWS(!costs.empty(),
                           "Remesh placement requires at least one block");
  PARTHENON_REQUIRE_THROWS(
      std::all_of(costs.begin(), costs.end(),
                  [](const double cost) { return std::isfinite(cost) && cost > 0.0; }),
      "Remesh placement costs must be finite and positive");
}

OwnershipMap BuildOwnership(const std::vector<int> &owners,
                            const std::vector<double> &costs,
                            const std::vector<MemoryFootprint> *footprints,
                            const int nranks) {
  OwnershipMap ownership;
  ownership.owner = owners;
  ownership.local_id.resize(costs.size());
  ownership.rank_leaves.resize(nranks);
  ownership.rank_starts.assign(nranks, 0);
  ownership.rank_counts.assign(nranks, 0);
  ownership.rank_costs.assign(nranks, 0.0);
  ownership.rank_host_bytes.assign(nranks, 0);
  ownership.rank_device_bytes.assign(nranks, 0);

  for (std::int64_t gid = 0; gid < owners.size(); ++gid) {
    const int owner = owners[gid];
    PARTHENON_REQUIRE_THROWS(owner >= 0 && owner < nranks, "Invalid remesh owner rank");
    ownership.local_id[gid] = ownership.rank_leaves[owner].size();
    ownership.rank_leaves[owner].push_back(gid);
    ++ownership.rank_counts[owner];
    ownership.rank_costs[owner] += costs[gid];
    if (footprints != nullptr) {
      ownership.rank_host_bytes[owner] =
          CheckedAdd(ownership.rank_host_bytes[owner], (*footprints)[gid].host_owned);
      ownership.rank_device_bytes[owner] =
          CheckedAdd(ownership.rank_device_bytes[owner], (*footprints)[gid].device_owned);
    }
  }
  for (int rank = 1; rank < nranks; ++rank) {
    ownership.rank_starts[rank] =
        ownership.rank_starts[rank - 1] + ownership.rank_counts[rank - 1];
  }
  ValidateContiguousOwnership(ownership);
  return ownership;
}

bool Fits(const std::uint64_t host, const std::uint64_t device,
          const MemoryBudget &budget) {
  return (!budget.host_bytes || host <= *budget.host_bytes) &&
         (!budget.device_bytes || device <= *budget.device_bytes);
}

std::uint64_t PlacementDeviceBytes(const MemoryFootprint &footprint) {
  auto bytes = CheckedAdd(footprint.device_owned, footprint.execution_stage_reserve);
  return CheckedAdd(bytes, footprint.finalization_reserve);
}

struct Chunk {
  std::size_t begin;
  std::size_t end;
  double cost;
  std::uint64_t host;
  std::uint64_t device;
};

Chunk MakeChunk(const std::size_t begin, const std::size_t end,
                const std::vector<double> &costs,
                const std::vector<MemoryFootprint> &footprints) {
  Chunk chunk{begin, end, 0.0, 0, 0};
  for (std::size_t block = begin; block < end; ++block) {
    chunk.cost += costs[block];
    chunk.host = CheckedAdd(chunk.host, footprints[block].host_owned);
    chunk.device = CheckedAdd(chunk.device, PlacementDeviceBytes(footprints[block]));
  }
  return chunk;
}

std::vector<Chunk> BuildFeasibleChunks(const std::vector<double> &costs,
                                       const std::vector<MemoryFootprint> &footprints,
                                       const int nranks, const MemoryBudget &budget) {
  std::vector<Chunk> reverse_chunks;
  std::size_t end = costs.size();
  while (end > 0) {
    std::size_t begin = end;
    std::uint64_t host = 0;
    std::uint64_t device = 0;
    while (begin > 0) {
      const auto next_host = CheckedAdd(host, footprints[begin - 1].host_owned);
      const auto next_device =
          CheckedAdd(device, PlacementDeviceBytes(footprints[begin - 1]));
      if (!Fits(next_host, next_device, budget)) break;
      --begin;
      host = next_host;
      device = next_device;
    }
    PARTHENON_REQUIRE_THROWS(begin != end,
                             "A remesh block exceeds the configured memory budget");
    reverse_chunks.push_back(MakeChunk(begin, end, costs, footprints));
    end = begin;
  }
  PARTHENON_REQUIRE_THROWS(reverse_chunks.size() <= nranks,
                           "No contiguous remesh placement satisfies the memory budget");

  std::vector<Chunk> chunks(reverse_chunks.rbegin(), reverse_chunks.rend());
  while (chunks.size() < std::min(costs.size(), static_cast<std::size_t>(nranks))) {
    auto split = chunks.end();
    for (auto it = chunks.begin(); it != chunks.end(); ++it) {
      if (it->end - it->begin > 1 && (split == chunks.end() || it->cost > split->cost)) {
        split = it;
      }
    }
    if (split == chunks.end()) break;

    std::size_t cut = split->begin + 1;
    double left_cost = costs[split->begin];
    double best_difference = std::abs(split->cost - 2.0 * left_cost);
    for (std::size_t candidate = split->begin + 2; candidate < split->end; ++candidate) {
      left_cost += costs[candidate - 1];
      const double difference = std::abs(split->cost - 2.0 * left_cost);
      if (difference < best_difference) {
        best_difference = difference;
        cut = candidate;
      }
    }
    const auto left = MakeChunk(split->begin, cut, costs, footprints);
    const auto right = MakeChunk(cut, split->end, costs, footprints);
    split = chunks.erase(split);
    split = chunks.insert(split, right);
    chunks.insert(split, left);
  }
  return chunks;
}

EstimateQuality CombineQuality(const EstimateQuality left, const EstimateQuality right) {
  return static_cast<EstimateQuality>(
      std::max(static_cast<int>(left), static_cast<int>(right)));
}

void AddBlockMemory(RankMemoryFootprint &rank, const MemoryFootprint &block,
                    const bool device) {
  const auto resident = device ? block.device_owned : block.host_owned;
  rank.immediate_resident = CheckedAdd(rank.immediate_resident, resident);
  if (device) {
    rank.next_advance_reserve =
        CheckedAdd(rank.next_advance_reserve, block.execution_stage_reserve);
    rank.finalization_reserve =
        CheckedAdd(rank.finalization_reserve, block.finalization_reserve);
  }
  rank.transfer_reserve = std::max(rank.transfer_reserve, resident);
  rank.quality = CombineQuality(rank.quality, block.quality);
}

void FinishPeak(RankMemoryFootprint &rank) {
  rank.peak = CheckedAdd(rank.source_resident, rank.immediate_resident);
  rank.peak = CheckedAdd(rank.peak, rank.next_advance_reserve);
  rank.peak = CheckedAdd(rank.peak, rank.finalization_reserve);
  rank.peak = CheckedAdd(rank.peak, rank.transfer_reserve);
  rank.peak = CheckedAdd(rank.peak, rank.runtime_overhead);
}

} // namespace

OwnershipMap PlaceBlocksContiguously(const std::vector<double> &costs, int nranks) {
  PARTHENON_REQUIRE_THROWS(nranks > 0, "Remesh placement requires at least one rank");
  ValidateCosts(costs);
  const double total_cost = std::accumulate(costs.begin(), costs.end(), 0.0);

  std::vector<int> owners(costs.size());
  int rank = nranks - 1;
  double target_cost = total_cost / nranks;
  double rank_cost = 0.0;
  double remaining_cost = total_cost;
  for (int block = static_cast<int>(costs.size()) - 1; block >= 0; --block) {
    PARTHENON_REQUIRE_THROWS(target_cost > 0.0,
                             "Remesh placement would leave a rank without a block");
    rank_cost += costs[block];
    owners[block] = rank;
    if (rank_cost >= target_cost && rank > 0) {
      --rank;
      remaining_cost -= rank_cost;
      rank_cost = 0.0;
      target_cost = remaining_cost / (rank + 1);
    }
  }
  return BuildOwnership(owners, costs, nullptr, nranks);
}

OwnershipMap PlaceBlocksContiguously(const std::vector<double> &costs,
                                     const std::vector<MemoryFootprint> &footprints,
                                     const int nranks, const MemoryBudget &budget) {
  PARTHENON_REQUIRE_THROWS(nranks > 0, "Remesh placement requires at least one rank");
  ValidateCosts(costs);
  PARTHENON_REQUIRE_THROWS(costs.size() == footprints.size(),
                           "Remesh costs and memory footprints disagree");
  if (budget.require_hard_guarantee) {
    PARTHENON_REQUIRE_THROWS(
        std::none_of(footprints.begin(), footprints.end(),
                     [](const auto &footprint) {
                       return footprint.quality == EstimateQuality::estimate;
                     }),
        "A hard remesh memory guarantee requires bounded storage declarations");
  }
  if (!budget.host_bytes && !budget.device_bytes) {
    auto ownership = PlaceBlocksContiguously(costs, nranks);
    for (std::size_t gid = 0; gid < footprints.size(); ++gid) {
      const auto rank = ownership.owner[gid];
      ownership.rank_host_bytes[rank] =
          CheckedAdd(ownership.rank_host_bytes[rank], footprints[gid].host_owned);
      ownership.rank_device_bytes[rank] =
          CheckedAdd(ownership.rank_device_bytes[rank], footprints[gid].device_owned);
    }
    return ownership;
  }

  const auto chunks = BuildFeasibleChunks(costs, footprints, nranks, budget);
  std::vector<int> owners(costs.size());
  const int first_rank = nranks - chunks.size();
  for (int chunk = 0; chunk < chunks.size(); ++chunk) {
    for (std::size_t block = chunks[chunk].begin; block < chunks[chunk].end; ++block)
      owners[block] = first_rank + chunk;
  }
  return BuildOwnership(owners, costs, &footprints, nranks);
}

void ValidateContiguousOwnership(const OwnershipMap &ownership) {
  PARTHENON_REQUIRE_THROWS(ownership.rank_starts.size() == ownership.rank_counts.size(),
                           "Ownership rank arrays disagree");
  for (int rank = 0; rank < ownership.rank_leaves.size(); ++rank) {
    PARTHENON_REQUIRE_THROWS(ownership.rank_leaves[rank].size() ==
                                 ownership.rank_counts[rank],
                             "Ownership rank count disagrees with its leaf list");
    for (int local_id = 0; local_id < ownership.rank_leaves[rank].size(); ++local_id) {
      const auto gid = ownership.rank_leaves[rank][local_id];
      PARTHENON_REQUIRE_THROWS(
          gid == ownership.rank_starts[rank] + local_id,
          "Initial remesh placement must be contiguous in GID order");
      PARTHENON_REQUIRE_THROWS(ownership.owner[gid] == rank &&
                                   ownership.local_id[gid] == local_id,
                               "Ownership reverse lookup is inconsistent");
    }
  }
}

std::vector<double> PlanRemeshCosts(const TopologyPlan &topology,
                                    const InventoryPlan &inventory,
                                    const std::vector<double> &old_costs,
                                    const CostPolicy policy,
                                    const CostCallback &application_cost) {
  PARTHENON_REQUIRE_THROWS(topology.changed, "Cost planning requires a changed topology");
  PARTHENON_REQUIRE_THROWS(topology.leaves.size() == inventory.blocks.size(),
                           "Remesh topology and inventory disagree");
  PARTHENON_REQUIRE_THROWS(old_costs.size() == topology.successors.size(),
                           "Remesh source cost count disagrees with topology");
  ValidateCosts(old_costs);
  PARTHENON_REQUIRE_THROWS(policy != CostPolicy::application || application_cost,
                           "Application remesh cost policy requires a callback");

  std::vector<double> costs(topology.leaves.size());
  for (const auto &leaf : topology.leaves) {
    PARTHENON_REQUIRE_THROWS(leaf.new_gid >= 0 && leaf.new_gid < inventory.blocks.size(),
                             "Invalid remesh destination GID");
    double cost = 0.0;
    if (policy == CostPolicy::inherit) {
      double weighted_cost = 0.0;
      double total_weight = 0.0;
      for (const auto &source : leaf.sources) {
        PARTHENON_REQUIRE_THROWS(source.source.gid >= 0 &&
                                     source.source.gid < old_costs.size(),
                                 "Invalid remesh source GID");
        weighted_cost += old_costs[source.source.gid];
        total_weight += 1.0;
      }
      PARTHENON_REQUIRE_THROWS(total_weight > 0.0, "Remesh leaf has no source cost");
      cost = weighted_cost / total_weight;
    } else if (policy == CostPolicy::parthenon) {
      const auto &block = inventory.blocks[leaf.new_gid];
      std::uint64_t particles = 0;
      for (const auto &swarm : block.swarms)
        particles = CheckedAdd(particles, swarm.active_count);
      cost = std::max(1.0, static_cast<double>(block.footprint.device_owned) /
                                   static_cast<double>(sizeof(Real)) +
                               static_cast<double>(particles));
    } else {
      std::vector<CostObservation> observations;
      observations.reserve(leaf.sources.size());
      for (const auto &source : leaf.sources) {
        PARTHENON_REQUIRE_THROWS(source.source.gid >= 0 &&
                                     source.source.gid < old_costs.size(),
                                 "Invalid remesh source GID");
        observations.push_back({source.source, old_costs[source.source.gid], 1.0});
      }
      cost = application_cost({leaf, inventory.blocks[leaf.new_gid], observations}).value;
    }
    PARTHENON_REQUIRE_THROWS(
        std::isfinite(cost) && cost > 0.0,
        "Remesh cost policy returned a non-finite or non-positive cost");
    costs[leaf.new_gid] = cost;
  }
  return costs;
}

PlacementPlan PlanRemeshPlacement(const TopologyPlan &topology,
                                  const InventoryPlan &inventory,
                                  const std::vector<double> &old_costs,
                                  const std::vector<std::uint64_t> &source_device_bytes,
                                  const int nranks, const CostPolicy policy,
                                  const MemoryBudget &budget,
                                  const CostCallback &application_cost) {
  PARTHENON_REQUIRE_THROWS(source_device_bytes.size() == nranks,
                           "Remesh source memory count disagrees with rank count");
  PlacementPlan plan;
  plan.costs = PlanRemeshCosts(topology, inventory, old_costs, policy, application_cost);
  std::vector<MemoryFootprint> footprints;
  footprints.reserve(inventory.blocks.size());
  for (const auto &block : inventory.blocks)
    footprints.push_back(block.footprint);
  plan.ownership = PlaceBlocksContiguously(plan.costs, footprints, nranks, budget);
  plan.host_memory.resize(nranks);
  plan.device_memory.resize(nranks);
  for (int rank = 0; rank < nranks; ++rank) {
    plan.device_memory[rank].source_resident = source_device_bytes[rank];
    for (const auto gid : plan.ownership.rank_leaves[rank]) {
      AddBlockMemory(plan.host_memory[rank], footprints[gid], false);
      AddBlockMemory(plan.device_memory[rank], footprints[gid], true);
    }
    FinishPeak(plan.host_memory[rank]);
    FinishPeak(plan.device_memory[rank]);
    PARTHENON_REQUIRE_THROWS(!budget.host_bytes ||
                                 plan.host_memory[rank].peak <= *budget.host_bytes,
                             "Remesh host-memory estimate exceeds the configured budget");
    PARTHENON_REQUIRE_THROWS(
        !budget.device_bytes || plan.device_memory[rank].peak <= *budget.device_bytes,
        "Remesh device-memory estimate exceeds the configured budget");
  }
  plan.hard_memory_guarantee =
      budget.require_hard_guarantee &&
      std::none_of(footprints.begin(), footprints.end(), [](const auto &footprint) {
        return footprint.quality == EstimateQuality::estimate;
      });
  return plan;
}

} // namespace parthenon::remesh
