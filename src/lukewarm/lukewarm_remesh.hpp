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
#ifndef LUKEWARM_LUKEWARM_REMESH_HPP_
#define LUKEWARM_LUKEWARM_REMESH_HPP_

// This file was made in part with generative AI.

#include <memory>
#include <vector>

#include "mesh/forest/logical_location.hpp"

namespace parthenon {

class LukewarmContainer;
class Mesh;

// A remesh discards every MeshData, and with them the lukewarms they own. Particles come
// through it in two steps:
//   1. DetachLukewarms(), before the old MeshData go, settles every particle in its
//      block and takes the lukewarms off the old MeshData.
//   2. RemeshLukewarms(), once the new MeshData have their lukewarms, sends each
//      particle from its old MeshData to the new one owning its new block: its old
//      block, its parent, or the daughter it lies in.
// Particles move from MeshData to MeshData as in every other exchange, packed on device
// and sent straight from device memory. Both steps are collective over all ranks.

// The lukewarms of the old MeshData of this rank, waiting for the new MeshData
struct DetachedLukewarms {
  std::vector<std::shared_ptr<LukewarmContainer>> containers;
  // The gids of the blocks of each container, in its order
  std::vector<std::vector<int>> gids;
  // The location, rank, and owning MeshData of every old block, by gid
  std::vector<LogicalLocation> locs;
  std::vector<int> ranks;
  std::vector<int> owners;
};

// locs and ranks hold the location and rank of every block, by gid
DetachedLukewarms DetachLukewarms(Mesh *pmesh, const std::vector<LogicalLocation> &locs,
                                  const std::vector<int> &ranks);

// oldtonew holds the new gid of every old block, or of its parent, or of its first
// daughter, and locs and ranks the location and rank of every new block
void RemeshLukewarms(Mesh *pmesh, DetachedLukewarms &&detached,
                     const std::vector<int> &oldtonew,
                     const std::vector<LogicalLocation> &locs,
                     const std::vector<int> &ranks);

} // namespace parthenon

#endif // LUKEWARM_LUKEWARM_REMESH_HPP_
