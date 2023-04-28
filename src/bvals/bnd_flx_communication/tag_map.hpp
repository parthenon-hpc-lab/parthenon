//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020-2022 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2022. Triad National Security, LLC. All rights reserved.
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

#ifndef BVALS_BND_FLX_COMMUNICATION_TAG_MAP_HPP_
#define BVALS_BND_FLX_COMMUNICATION_TAG_MAP_HPP_

#include <map>
#include <memory>
#include <unordered_map>

#include "basic_types.hpp"

namespace parthenon {

// Struct uniquely identifying a particular geometric element (i.e. face, edge, or node)
// of a particular block
struct BlockGeometricElementId {
  int gid;
  int orientation;
};
// Need to define (somewhat arbitrarily) comparison operators for the
// BlockGeometricElement since we want the elements that coincide across to blocks across
// ranks to have a natural ordering so they can be placed in a map (which is ordered)
inline bool operator<(BlockGeometricElementId a, BlockGeometricElementId b) {
  if (a.gid == b.gid) return a.orientation < b.orientation;
  return a.gid < b.gid;
}
inline bool operator>(BlockGeometricElementId a, BlockGeometricElementId b) {
  if (a.gid == b.gid) return a.orientation > b.orientation;
  return a.gid > b.gid;
}
inline bool operator==(BlockGeometricElementId a, BlockGeometricElementId b) {
  return (a.gid == b.gid && a.orientation == b.orientation);
}

// We also need the concept of an unordered pair, since each communication channel
// is two way. When an element corresponding to a two-way channel is entered into our
// map we want it to have the same value no matter which order the two
// BlockGeometricelementIds were passed to the constructor
template <class T>
struct UnorderedPair {
  UnorderedPair(T in1, T in2)
      : first(in1 < in2 ? in1 : in2), second(in1 > in2 ? in1 : in2) {}
  T first, second;
};
// Need to also define an ordering for an UnorderedPair, once again this choice is
// somewhat arbitrary
template <class T>
inline bool operator<(UnorderedPair<T> a, UnorderedPair<T> b) {
  if (a.first == b.first) return a.second < b.second;
  return a.first < b.first;
}
template <class T>
inline bool operator>(UnorderedPair<T> a, UnorderedPair<T> b) {
  if (a.first == b.first) return a.second > b.second;
  return a.first > b.first;
}

class MeshBlock;
template <class T>
class MeshData;
class NeighborBlock;

class TagMap {
  // Unique keys defined by a two-way communication channel
  using rank_pair_t = UnorderedPair<BlockGeometricElementId>;
  // Map between a communication channel key and a unique MPI tag
  using rank_pair_map_t = std::map<rank_pair_t, int>;
  // Map of maps where the key corresponds to the MPI rank of the
  // other process
  using tag_map_t = std::unordered_map<int, rank_pair_map_t>;

  tag_map_t map_;

  // Given the two blocks (one described by the MeshBlock and the other described by the
  // firsts NeighborBlock information) return an ordered pair of BlockGeometricElementIds
  // corresponding to the two blocks geometric elements that coincide. This serves as a
  // unique key defining the two-way communication channel between these elements
  rank_pair_t MakeChannelPair(const std::shared_ptr<MeshBlock> &pmb,
                              const NeighborBlock &nb);

 public:
  void clear() { map_.clear(); }

  // Inserts all of the communication channels known about by MeshData md into the map
  void AddMeshDataToMap(std::shared_ptr<MeshData<Real>> &md);

  // Once all MeshData objects have inserted their known channels into the map, we can
  // iterate through a map for a given rank pair (which is already ordered by key because
  // of the properties of st::map) and assign each key a unique tag. By construction, this
  // tag is consistent across all ranks.
  void ResolveMap();

  // After the map has been resolved, get the tag for a particular MeshBlock NeighborBlock
  // pair
  int GetTag(const std::shared_ptr<MeshBlock> &pmb, const NeighborBlock &nb);
};
} // namespace parthenon

#endif // BVALS_BND_FLX_COMMUNICATION_TAG_MAP_HPP_
