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

#include <utility>
#include <vector>

#include "mesh/mesh_layout.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

MeshLayout::MeshLayout(std::vector<LogicalLocation> locs, std::vector<int> ranks,
                       std::vector<double> costs)
    : locs_(std::move(locs)), ranks_(std::move(ranks)), costs_(std::move(costs)) {
  PARTHENON_REQUIRE(ranks_.size() == locs_.size() && costs_.size() == locs_.size(),
                    "A mesh layout needs a rank and a cost for every block.");
}

MeshLayoutOverlap::MeshLayoutOverlap(const std::vector<LogicalLocation> &old_locs,
                                     const std::vector<LogicalLocation> &new_locs)
    : old_of_new_(new_locs.size()), new_of_old_(old_locs.size()),
      change_(new_locs.size()) {
  const int nold = old_locs.size();
  const int nnew = new_locs.size();
  int on = 0, nn = 0;
  while (on < nold && nn < nnew) {
    const auto &oloc = old_locs[on];
    const auto &nloc = new_locs[nn];
    if (nloc.level() == oloc.level()) {
      PARTHENON_REQUIRE(nloc == oloc, "Old and new meshes are not in the same order.");
      old_of_new_[nn] = {on, 1};
      new_of_old_[on] = {nn, 1};
      change_[nn] = Change::same;
      ++on, ++nn;
    } else if (nloc.level() == oloc.level() + 1) {
      // Old block on was refined into the new blocks starting at nn
      const int first = nn;
      for (; nn < nnew && new_locs[nn].level() == nloc.level() &&
             new_locs[nn].GetParent() == oloc;
           ++nn) {
        old_of_new_[nn] = {on, 1};
        change_[nn] = Change::refined;
      }
      PARTHENON_REQUIRE(nn > first, "Old and new meshes are not in the same order.");
      new_of_old_[on++] = {first, nn - first};
    } else if (nloc.level() + 1 == oloc.level()) {
      // The old blocks starting at on were derefined into new block nn
      const int first = on;
      for (; on < nold && old_locs[on].level() == oloc.level() &&
             old_locs[on].GetParent() == nloc;
           ++on)
        new_of_old_[on] = {nn, 1};
      PARTHENON_REQUIRE(on > first, "Old and new meshes are not in the same order.");
      old_of_new_[nn] = {first, on - first};
      change_[nn++] = Change::derefined;
    } else {
      PARTHENON_FAIL("Remeshing only supports one level of refinement per block.");
    }
  }
  PARTHENON_REQUIRE(on == nold && nn == nnew,
                    "Old and new meshes do not cover the same domain.");
}

} // namespace parthenon
