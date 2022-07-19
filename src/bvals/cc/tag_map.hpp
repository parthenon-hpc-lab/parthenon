//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020 The Parthenon collaboration
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

#ifndef BVALS_CC_TAG_MAP_HPP_
#define BVALS_CC_TAG_MAP_HPP_

#include <map> 

namespace parthenon {

struct BlockGeometricElementId {
  int gid;
  int orientation;  
};

inline bool operator<(BlockGeometricElementId a, BlockGeometricElementId b) {
  if (a.gid == b.gid) return a.orientation < b.orientation;  
  return a.gid < b.gid; 
}
inline bool operator>(BlockGeometricElementId a, BlockGeometricElementId b) {
  if (a.gid == b.gid) return a.orientation > b.orientation;  
  return a.gid > b.gid; 
}
inline bool operator==(BlockGeometricElementId a, BlockGeometricElementId b) {
  if (a.gid==b.gid && a.orientation == b.orientation) return true;
  return false;
}

template<class T>
struct UnorderedPair { 
  UnorderedPair(T in1, T in2) 
      : first(in1 < in2 ? in1 : in2), 
        second(in1 > in2 ? in1 : in2) {} 
  T first, second;
};
template<class T> 
inline bool operator<(UnorderedPair<T> a, UnorderedPair<T> b) {
  if (a.first == b.first) return a.second < b.second; 
  return a.first < b.first; 
}
template<class T> 
inline bool operator>(UnorderedPair<T> a, UnorderedPair<T> b) {
  if (a.first == b.first) return a.second > b.second; 
  return a.first > b.first; 
}

class MeshBlock;
template<class T>
class MeshData; 
class NeighborBlock;

class TagMap { 
  using rank_pair_t = UnorderedPair<BlockGeometricElementId>; 
  using rank_pair_map_t = std::map<rank_pair_t, int>; 
  using tag_map_t = std::map<int, rank_pair_map_t>; 
  
  tag_map_t map_; 
  
  rank_pair_t MakeChannelPair(const std::shared_ptr<MeshBlock> &pmb, const NeighborBlock &nb);
  
 public:   
  void clear() {map_.clear();} 

  void AddMeshDataToMap(std::shared_ptr<MeshData<Real>> &md);

  void ResolveMap();

  int GetTag(const std::shared_ptr<MeshBlock>& pmb, const NeighborBlock &nb);
};
} // namespace parthenon

#endif // BVALS_CC_TAG_MAP_HPP_
