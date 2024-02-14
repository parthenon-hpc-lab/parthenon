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

  constexpr int NDIM = 2; 
  template <class T, int SIZE>
  using sptr_vec_t = std::array<std::shared_ptr<T>, SIZE>;
  
  enum class Direction : uint {I = 0, J = 1, K = 2};

  struct EdgeLoc { 
    Direction dir; 
    bool lower;
    
    // In 2D we can ignore connectivity of K-direction faces, 
    int GetFaceIdx2D() const {
      return (1 - 2 * lower) * std::pow(3, (static_cast<uint>(dir) + 1) % 2) + 1 + 3 + 9;
    } 

    const static EdgeLoc South; 
    const static EdgeLoc North; 
    const static EdgeLoc West; 
    const static EdgeLoc East; 
  };
  inline const EdgeLoc EdgeLoc::South{Direction::I, true}; 
  inline const EdgeLoc EdgeLoc::North{Direction::I, false}; 
  inline const EdgeLoc EdgeLoc::West{Direction::J, true}; 
  inline const EdgeLoc EdgeLoc::East{Direction::J, false}; 
  inline bool operator==(const EdgeLoc &lhs, const EdgeLoc &rhs) {
    return (lhs.dir == rhs.dir) && (lhs.lower == rhs.lower);
  }

}
}

template<>
class std::hash<parthenon::forest::EdgeLoc> {
 public:
  std::size_t operator()(const parthenon::forest::EdgeLoc &key) const noexcept {
    return 2 * static_cast<uint>(key.dir) + key.lower;
  }
};

namespace parthenon {
namespace forest { 
  
  struct RelativeOrientation { 
    RelativeOrientation() : dir_connection{0, 1, 2}, dir_flip{false, false, false} {};
  
    static RelativeOrientation FromSharedEdge2D(EdgeLoc origin, EdgeLoc neighbor, int orientation) { 
      if (origin.dir == Direction::K || neighbor.dir == Direction::K) { 
        PARTHENON_FAIL("In 2D we shouldn't have explicit edges in the Z direction.");
      }
  
      RelativeOrientation out;
      out.dir_connection[static_cast<uint>(origin.dir)] = static_cast<uint>(neighbor.dir);
      out.dir_flip[static_cast<uint>(origin.dir)] = orientation == -1;
      out.dir_connection[(static_cast<uint>(origin.dir) + 1) % 2] = (static_cast<uint>(neighbor.dir) + 1) % 2; 
      out.dir_flip[(static_cast<uint>(origin.dir) + 1) % 2] = (neighbor.lower == origin.lower);
      return out;
    }
  
    LogicalLocation Transform(const LogicalLocation &loc_in) const { 
      std::array<std::int64_t, 3> l_out; 
      int nblock = 1LL << loc_in.level();  
      for (int dir = 0; dir < 3; ++dir) {  
        std::int64_t l_in = loc_in.l(dir);
        // First shift the logical location index back into the interior 
        // of a bordering tree assuming they have the same coordinate 
        // orientation
        l_in = (l_in + nblock) % nblock; 
        // Then permute (and possibly flip) the coordinate indices 
        // to move to the logical coordinate system of the new tree
        if (dir_flip[dir]) { 
          l_out[abs(dir_connection[dir])] = nblock - 1 - l_in;
        } else {
          l_out[abs(dir_connection[dir])] = l_in; 
        }
      }    
      return LogicalLocation(loc_in.level(), l_out[0], l_out[1], l_out[2]);
    }
  
    int dir_connection[3]; 
    bool dir_flip[3];
  };
  
  // We don't allow for periodic boundaries, since we can encode periodicity through connectivity in the forest
  class Tree { 
   public: 
    Tree(int ndim, int root_level) : ndim(ndim) { 
      // Add internal and leaf nodes of the initial tree
      for (int l = 0; l <= root_level; ++l) { 
        for (int k = 0; k < (ndim > 2 ? (1LL << l) : 1); ++k) { 
          for (int j = 0; j < (ndim > 1 ? (1LL << l) : 1); ++j) { 
            for (int i = 0; i < (ndim > 0 ? (1LL << l) : 1); ++i) { 
              if (l == root_level) {
                leaves.emplace(l, i, j, k); 
              } else {
                internal_nodes.emplace(l, i, j, k); 
              }
            }
          }
        }
      }
    }
  
    static std::shared_ptr<Tree> create(int ndim, int root_level) {
      return std::make_shared<Tree>(ndim, root_level);
    }
  
    int Refine(LogicalLocation ref_loc) {
      // Check that this is a valid refinement location 
      if (!leaves.count(ref_loc)) return 0; // Can't refine a block that doesn't exist
      
      // Perform the refinement for this block 
      std::vector<LogicalLocation> daughters = ref_loc.GetDaughters(ndim); 
      leaves.erase(ref_loc); 
      internal_nodes.insert(ref_loc);
      leaves.insert(daughters.begin(), daughters.end());
      int nadded = daughters.size();
  
      // Enforce internal proper nesting
      LogicalLocation parent = ref_loc.GetParent();
      int ox1 = ref_loc.lx1() - (parent.lx1() << 1); 
      int ox2 = ref_loc.lx2() - (parent.lx2() << 1); 
      int ox3 = ref_loc.lx3() - (parent.lx3() << 1); 
  
      for (int k = 0; k < (ndim > 2 ? 2 : 1); ++k) {
        for (int j = 0; j < (ndim > 1 ? 2 : 1); ++j) {
          for (int i = 0; i < (ndim > 0 ? 2 : 1); ++i) {
            LogicalLocation neigh = parent.GetSameLevelNeighbor(i + ox1 - 1, j + ox2 - (ndim > 1), k + ox3 - (ndim > 2));
            if (leaves.count(neigh)) {
              nadded += Refine(neigh);
            }
            if (!neigh.IsInTree()) {
              // Need to communicate this refinement action to possible neighboring tree(s) and 
              // trigger refinement there
              int n_idx = neigh.NeighborTreeIndex(); 
              for (auto & [neighbor_tree, orientation] : neighbors[n_idx]) {
                nadded += neighbor_tree->Refine(orientation.Transform(neigh));
              }
            }
          }
        }
      }
      return nadded;
    }
  
    int Derefine(LogicalLocation ref_loc) { 
      // ref_loc is the block to be added and its daughters are the blocks to be removed 
      std::vector<LogicalLocation> daughters = ref_loc.GetDaughters(ndim);
  
      // Check that we can actually de-refine 
      for (LogicalLocation &d : daughters) { 
        // Check that the daughters actually exist as leaf nodes 
        if (!leaves.count(d)) return 0; 
        
        // Check that removing these blocks doesn't break proper nesting, that just means that any of the daughters 
        // same level neighbors can't be in the internal node list (which would imply that the daughter abuts a finer block) 
        // Note: these loops check more than is necessary, but as written are simpler than the minimal set
        const std::vector<int> active{-1, 0, 1};
        const std::vector<int> inactive{0};
        for (int k : (ndim > 2) ? active : inactive) {
          for (int j : (ndim > 1) ? active : inactive) {
            for (int i : (ndim > 0) ? active : inactive) {
              LogicalLocation neigh = d.GetSameLevelNeighbor(i, j, k);
              if (internal_nodes.count(neigh)) return 0; 
              if (!neigh.IsInTree()) { 
                // Need to check that this derefinement doesn't break proper nesting with
                // a neighboring tree
              }
            }
          }
        }
      }
  
      // Derefinement is ok
      for (auto &d : daughters)
          leaves.erase(d);
      internal_nodes.erase(ref_loc); 
      leaves.insert(ref_loc);
      return daughters.size();
    }
    
    void Print(std::string fname) const {
      FILE * pFile;
      pFile = fopen(fname.c_str(), "w");
      for (const auto &l : leaves) 
        fprintf(pFile, "%i, %i, %i\n", l.level(), l.lx1(), l.lx2());
      fclose(pFile);
    }
    
    void AddNeighbor(int location_idx, std::shared_ptr<Tree> neighbor_tree, RelativeOrientation orient) { 
      neighbors[location_idx].push_back(std::make_pair(neighbor_tree, orient));   
    }
  
   private:
    int ndim;  
    std::unordered_set<LogicalLocation> leaves; 
    std::unordered_set<LogicalLocation> internal_nodes; 
    std::array<std::vector<std::pair<std::shared_ptr<Tree>, RelativeOrientation>>, 27> neighbors;
  };

  class Face; 
  class Node { 
   public: 
    Node(int id_in, std::array<Real, NDIM> pos) : id(id_in), x(pos) {}
    static std::shared_ptr<Node> create(int id, std::array<Real, NDIM> pos) { 
      return std::make_shared<Node>(id, pos);
    } 
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
    Face() : tree(Tree::create(NDIM, 0)) {};
  
    // Constructor that can only be called internally 
    Face(sptr_vec_t<Node, 4> nodes_in, Private_t) : nodes(nodes_in), tree(Tree::create(NDIM, 0)) {
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
    std::shared_ptr<Tree> tree;
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