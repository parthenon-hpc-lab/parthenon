//========================================================================================
// (C) (or copyright) 2024. Triad National Security, LLC. All rights reserved.
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


#include <algorithm>
#include <array>
#include <iterator>
#include <map>
#include <memory>
#include <set>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "basic_types.hpp"
#include "defs.hpp"
#include "mesh/forest/forest_node.hpp"
#include "mesh/forest/forest_topology.hpp"
#include "mesh/forest/logical_location.hpp"
#include "mesh/forest/tree.hpp"
#include "utils/bit_hacks.hpp"
#include "utils/indexer.hpp"

namespace parthenon {
namespace forest {

template<class T> 
std::vector<std::shared_ptr<Node>> NodeListOverlap(T nodes_1, T nodes_2) {
  std::sort(std::begin(nodes_1), std::end(nodes_1));
  std::sort(std::begin(nodes_2), std::end(nodes_2));
  std::vector<std::shared_ptr<Node>> node_intersection;
  std::set_intersection(std::begin(nodes_1), std::end(nodes_1), 
                        std::begin(nodes_2), std::end(nodes_2), 
                        std::back_inserter(node_intersection));
  return node_intersection;
} 

void Face::SetNeighbors() { 
  std::unordered_set<std::shared_ptr<Face>> neighbors_local;
  for (auto &node : nodes)
    neighbors_local.insert(node->associated_faces.begin(),
                           node->associated_faces.end()); 
  for (std::shared_ptr<Face> neighbor : neighbors_local) { 
    auto node_overlap = NodeListOverlap(nodes, neighbor->nodes);
    if (node_overlap.size() > 0 && node_overlap.size() < 4) {
      std::array<int, 2> offset{0, 0};
      for (int i=0; i < 4; ++i) {
        if (std::find(node_overlap.begin(), node_overlap.end(), nodes[i]) != node_overlap.end()) {
          for (int o = 0; o < 2; ++o) offset[o] += node_to_offset[i][o];
        }
      }
      for (auto &o : offset) o /= node_overlap.size();
      neighbors[offset[0] + 1][offset[1] + 1].push_back(neighbor);
    }
  }
}

void Face::SetEdgeCoordinateTransforms() { 
  for (int ox = -1; ox <= 1; ++ox) {
    for (int oy = -1; oy <= 1; ++oy) {
      if (std::abs(ox) + std::abs(oy) == 1) { 
        for (auto &neighbor : neighbors[ox + 1][oy + 1]) { 
          auto node_overlap = NodeListOverlap(nodes, neighbor->nodes);
          auto {loc, edge} = GetEdge(node_overlap); 
          auto {nloc, nedge} = neighbor->GetEdge(node_overlap); 
          coord_trans[ox1 + 1][oy + 1].push_back(
            LogicalCoordinateTransformationFromSharedEdge2D(loc, nloc, edge.RelativeOrientation(nedge)));
        }
      }
    }
  }
}

void Face::SetNodeCoordinateTransforms() { 
  for (int ox = -1; ox <= 1; ++ox) {
    for (int oy = -1; oy <= 1; ++oy) {
      if (std::abs(ox) + std::abs(oy) == 2) { 
        for (auto &neighbor : neighbors[ox + 1][oy + 1]) { 
          // TODO(LFR): Find an edge neighbor that is shared by both 
        }
      }
    }
  }
}

} // namespace forest
} // namespace parthenon

