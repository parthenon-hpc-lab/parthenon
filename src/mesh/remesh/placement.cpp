#include "mesh/remesh/placement.hpp"

#include <algorithm>
#include <numeric>

#include "utils/error_checking.hpp"

namespace parthenon::remesh {

OwnershipMap PlaceBlocksContiguously(const std::vector<double> &costs, int nranks) {
  PARTHENON_REQUIRE(nranks > 0, "Remesh placement requires at least one rank");
  PARTHENON_REQUIRE(!costs.empty(), "Remesh placement requires at least one block");
  PARTHENON_REQUIRE(
      std::all_of(costs.begin(), costs.end(), [](double cost) { return cost >= 0.0; }),
      "Remesh placement costs must be nonnegative");
  const double total_cost = std::accumulate(costs.begin(), costs.end(), 0.0);
  PARTHENON_REQUIRE(total_cost > 0.0, "Remesh placement requires positive total cost");

  OwnershipMap ownership;
  ownership.owner.resize(costs.size());
  ownership.local_id.resize(costs.size());
  ownership.rank_leaves.resize(nranks);
  ownership.rank_starts.assign(nranks, 0);
  ownership.rank_counts.assign(nranks, 0);
  ownership.rank_costs.assign(nranks, 0.0);

  int rank = nranks - 1;
  double target_cost = total_cost / nranks;
  double rank_cost = 0.0;
  double remaining_cost = total_cost;
  for (int block = static_cast<int>(costs.size()) - 1; block >= 0; --block) {
    PARTHENON_REQUIRE(target_cost > 0.0,
                      "Remesh placement would leave a rank without a block");
    rank_cost += costs[block];
    ownership.owner[block] = rank;
    if (rank_cost >= target_cost && rank > 0) {
      --rank;
      remaining_cost -= rank_cost;
      rank_cost = 0.0;
      target_cost = remaining_cost / (rank + 1);
    }
  }

  for (std::int64_t gid = 0; gid < ownership.owner.size(); ++gid) {
    const int owner = ownership.owner[gid];
    ownership.local_id[gid] = ownership.rank_leaves[owner].size();
    ownership.rank_leaves[owner].push_back(gid);
    ++ownership.rank_counts[owner];
    ownership.rank_costs[owner] += costs[gid];
  }
  for (int current_rank = 1; current_rank < nranks; ++current_rank) {
    ownership.rank_starts[current_rank] =
        ownership.rank_starts[current_rank - 1] + ownership.rank_counts[current_rank - 1];
  }
  ValidateContiguousOwnership(ownership);
  return ownership;
}

void ValidateContiguousOwnership(const OwnershipMap &ownership) {
  PARTHENON_REQUIRE(ownership.rank_starts.size() == ownership.rank_counts.size(),
                    "Ownership rank arrays disagree");
  for (int rank = 0; rank < ownership.rank_leaves.size(); ++rank) {
    PARTHENON_REQUIRE(ownership.rank_leaves[rank].size() == ownership.rank_counts[rank],
                      "Ownership rank count disagrees with its leaf list");
    for (int local_id = 0; local_id < ownership.rank_leaves[rank].size(); ++local_id) {
      const auto gid = ownership.rank_leaves[rank][local_id];
      PARTHENON_REQUIRE(gid == ownership.rank_starts[rank] + local_id,
                        "Initial remesh placement must be contiguous in GID order");
      PARTHENON_REQUIRE(ownership.owner[gid] == rank &&
                            ownership.local_id[gid] == local_id,
                        "Ownership reverse lookup is inconsistent");
    }
  }
}

} // namespace parthenon::remesh
