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

#ifndef MESH_MESH_LAYOUT_HPP_
#define MESH_MESH_LAYOUT_HPP_

// This file was made in part with generative AI.

#include <unordered_map>
#include <vector>

#include "mesh/forest/logical_location.hpp"

namespace parthenon {

// The leaf blocks of a mesh and where they live: their logical locations in gid order,
// with the rank and cost of each block, plus a lookup from logical location to gid.
// During a remesh the old and new meshes each have a layout, and the blocks of one that
// overlap a block of the other follow from location lookups.
class MeshLayout {
 public:
  MeshLayout() = default;
  MeshLayout(std::vector<LogicalLocation> locs, std::vector<int> ranks,
             std::vector<double> costs, int ndim);

  int NumBlocks() const { return locs_.size(); }
  const LogicalLocation &Loc(int gid) const { return locs_[gid]; }
  int Rank(int gid) const { return ranks_[gid]; }
  double Cost(int gid) const { return costs_[gid]; }

  const std::vector<LogicalLocation> &Locs() const { return locs_; }
  const std::vector<int> &Ranks() const { return ranks_; }
  const std::vector<double> &Costs() const { return costs_; }

  // The gid of the block at loc, or -1 if loc is not a block of this layout
  int Gid(const LogicalLocation &loc) const {
    auto it = gid_of_.find(loc);
    return it == gid_of_.end() ? -1 : it->second;
  }

  // Gids of the blocks of this layout that overlap loc, a block of another layout of the
  // same mesh at most one level apart: the block at loc, its parent, or all of its
  // daughters (in increasing gid order)
  std::vector<int> GetOverlappingGids(const LogicalLocation &loc) const;

 private:
  std::vector<LogicalLocation> locs_;
  std::vector<int> ranks_;
  std::vector<double> costs_;
  std::unordered_map<LogicalLocation, int> gid_of_;
  int ndim_ = 3;
};

} // namespace parthenon

#endif // MESH_MESH_LAYOUT_HPP_
