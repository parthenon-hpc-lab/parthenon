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

#include <cstdint>
#include <vector>

#include "mesh/forest/logical_location.hpp"

namespace parthenon {

// The leaf blocks of a mesh and where they live: their logical locations in gid order,
// with the rank and cost of each block. During a remesh the old and new meshes each have
// a layout, and a MeshLayoutOverlap relates the blocks of the two.
class MeshLayout {
 public:
  MeshLayout() = default;
  MeshLayout(std::vector<LogicalLocation> locs, std::vector<int> ranks,
             std::vector<double> costs);

  int NumBlocks() const { return locs_.size(); }
  const LogicalLocation &Loc(int gid) const { return locs_[gid]; }
  int Rank(int gid) const { return ranks_[gid]; }
  double Cost(int gid) const { return costs_[gid]; }

  const std::vector<LogicalLocation> &Locs() const { return locs_; }
  const std::vector<int> &Ranks() const { return ranks_; }
  const std::vector<double> &Costs() const { return costs_; }

 private:
  std::vector<LogicalLocation> locs_;
  std::vector<int> ranks_;
  std::vector<double> costs_;
};

// The gids first to first + count - 1
struct GidRange {
  int first = 0;
  int count = 0;
  int end() const { return first + count; }
};

// How the blocks of the old and new meshes of a remesh overlap: each new block is the
// same as an old block, a daughter of one, or the parent of several. Both meshes list
// their blocks in forest order, which puts the daughters of a block contiguously in its
// place, so the overlap follows from one pass over the two lists.
class MeshLayoutOverlap {
 public:
  MeshLayoutOverlap() = default;
  MeshLayoutOverlap(const std::vector<LogicalLocation> &old_locs,
                    const std::vector<LogicalLocation> &new_locs);

  // The old blocks overlapping new block nn (the block at the same location, its parent,
  // or all of its daughters) and the new blocks overlapping old block on
  GidRange OldGids(int nn) const { return old_of_new_[nn]; }
  GidRange NewGids(int on) const { return new_of_old_[on]; }

  // The old block at the location of new block nn, or -1, and vice versa
  int SameOldGid(int nn) const {
    return change_[nn] == Change::same ? old_of_new_[nn].first : -1;
  }
  int SameNewGid(int on) const {
    const int nn = new_of_old_[on].first;
    return new_of_old_[on].count == 1 && change_[nn] == Change::same ? nn : -1;
  }
  // Whether new block nn is a daughter of an old block
  bool IsRefined(int nn) const { return change_[nn] == Change::refined; }

 private:
  enum class Change : std::int8_t { same, refined, derefined };
  std::vector<GidRange> old_of_new_, new_of_old_;
  std::vector<Change> change_;
};

} // namespace parthenon

#endif // MESH_MESH_LAYOUT_HPP_
