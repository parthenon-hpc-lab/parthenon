//========================================================================================
// (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001 for Los
// Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
// for the U.S. Department of Energy/National Nuclear Security Administration. All rights
// in the program are reserved by Triad National Security, LLC, and the U.S. Department
// of Energy/National Nuclear Security Administration. The Government is granted for
// itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do so.
//========================================================================================

// This file was made in part with generative AI.

#include <algorithm>
#include <utility>
#include <vector>

#include "mesh/mesh_layout.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

MeshLayout::MeshLayout(std::vector<LogicalLocation> locs, std::vector<int> ranks,
                       std::vector<double> costs, int ndim)
    : locs_(std::move(locs)), ranks_(std::move(ranks)), costs_(std::move(costs)),
      ndim_(ndim) {
  PARTHENON_REQUIRE(ranks_.size() == locs_.size() && costs_.size() == locs_.size(),
                    "A mesh layout needs a rank and a cost for every block.");
  gid_of_.reserve(locs_.size());
  for (int gid = 0; gid < locs_.size(); ++gid)
    gid_of_[locs_[gid]] = gid;
}

std::vector<int> MeshLayout::GetOverlappingGids(const LogicalLocation &loc) const {
  const int same = Gid(loc);
  if (same >= 0) return std::vector<int>{same};
  const int parent = Gid(loc.GetParent());
  if (parent >= 0) return std::vector<int>{parent};
  std::vector<int> gids;
  for (const auto &d : loc.GetDaughters(ndim_)) {
    const int gid = Gid(d);
    PARTHENON_REQUIRE(gid >= 0,
                      "Remeshing only supports one level of refinement per block.");
    gids.push_back(gid);
  }
  std::sort(gids.begin(), gids.end());
  return gids;
}

} // namespace parthenon
