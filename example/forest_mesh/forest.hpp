//========================================================================================
// (C) (or copyright) 2023. Triad National Security, LLC. All rights reserved.
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

#include <array>
#include <map>
#include <set>
#include <vector>
#include <memory>
#include <tuple> 
#include <unordered_map>
#include <unordered_set> 

#include "basic_types.hpp"
#include "mesh/logical_location.hpp"

namespace parthenon {
namespace forest { 

  using LogicalLocMap_t = std::map<LogicalLocation, std::pair<int, int>>;
  constexpr int NDIM = 2; 
  template <class T, int SIZE>
  using sptr_vec_t = std::array<std::shared_ptr<T>, SIZE>;
  
  enum class Direction : int {I = 0, J = 1, K = 2};
  enum class EdgeLoc : int {South = 0, West = 1, East = 2, North = 3};

  class Face; 
  class Node { 
   public: 
    Node(int id_in, std::array<Real, NDIM> pos) : id(id_in), x(pos) {} 
    std::uint32_t id;
    std::array<Real, NDIM> x;
    std::unordered_set<std::shared_ptr<Face>> associated_faces;  
  };
  
  class Edge {
   public: 
    Edge() = default;
    Edge(sptr_vec_t<Node, 2> nodes_in) : nodes(nodes_in) {}
  
    sptr_vec_t<Node, 2> nodes;
    Direction dir; 
  
    int RelativeOrientation(const Edge &e2) { 
      if (nodes[0] == e2.nodes[0] && nodes[1] == e2.nodes[1]) {
        return 1; 
      }
      else if (nodes[0] == e2.nodes[1] && nodes[1] == e2.nodes[0]) {
        return -1;    
      } else { 
        return 0;
      }
    }
  };
  
  class Face : public std::enable_shared_from_this<Face> { 
   private:  
    struct Private_t {};
   public: 
    Face() = default;
  
    // Constructor that can only be called internally 
    Face(sptr_vec_t<Node, 4> nodes_in, Private_t) : nodes(nodes_in) {
      edges[EdgeLoc::South] = Edge({nodes[0], nodes[1]});
      edges[EdgeLoc::West] = Edge({nodes[0], nodes[2]});
      edges[EdgeLoc::East] = Edge({nodes[1], nodes[3]});
      edges[EdgeLoc::North] = Edge({nodes[2], nodes[3]});
    }
  
    static std::shared_ptr<Face> create(sptr_vec_t<Node, 4> nodes_in) {
      auto result = std::make_shared<Face>(nodes_in, Private_t());
      // Associate the new face with the nodes
      for (auto & node : result->nodes)
          node->associated_faces.insert(result);
      return result;
    }
  
    std::shared_ptr<Face> getptr() {
      return shared_from_this();
    }
  
    sptr_vec_t<Node, 4> nodes;
    std::unordered_map<EdgeLoc, Edge> edges; 
    LogicalLocMap_t tree;
  };
  
  void ListFaces(const std::shared_ptr<Node>& node) { 
    for (auto & face : node->associated_faces) { 
      printf("{%i, %i, %i, %i}\n", face->nodes[0]->id, face->nodes[1]->id,
          face->nodes[2]->id, face->nodes[3]->id);
    }
  }
  
  using NeighborDesc = std::tuple<std::shared_ptr<Face>, EdgeLoc, int>;
  std::vector<NeighborDesc> FindEdgeNeighbors(const std::shared_ptr<Face> &face_in, EdgeLoc loc) {
    std::vector<NeighborDesc> neighbors;
    auto edge = face_in->edges[loc];
  
    std::unordered_set<std::shared_ptr<Face>> possible_neighbors;
    for (auto &node : edge.nodes)  
      possible_neighbors.insert(node->associated_faces.begin(), node->associated_faces.end());    
    
    // Check each neighbor to see if it shares an edge
    for (auto &neigh : possible_neighbors) {
      if (neigh != face_in) {
        for (auto &[neigh_loc, neigh_edge] : neigh->edges) {
          int orientation = edge.RelativeOrientation(neigh_edge);
          if (orientation)
            neighbors.push_back(std::make_tuple(neigh, neigh_loc, orientation));
        }
      }
    }
    return neighbors;
  } 
} // namespace forest
} // namespace parthenon