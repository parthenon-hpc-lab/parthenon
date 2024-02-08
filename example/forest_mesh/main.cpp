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

#include "basic_types.hpp"
#include "forest.hpp"
#include "mesh/logical_location.hpp"
#include "parthenon_manager.hpp"

using parthenon::ParthenonManager;
using parthenon::ParthenonStatus;
using parthenon::LogicalLocation;
using parthenon::Real;
using namespace parthenon::forest;

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
            // Need to communicate this refinement action to possible neighboring tree and 
            // trigger refinement there
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
  
  void Print() const { 
    for (const auto &l : leaves) 
      printf("%i, %i, %i\n", l.level(), l.lx1(), l.lx2());
  }

 private:
  int ndim;  
  std::unordered_set<LogicalLocation> leaves; 
  std::unordered_set<LogicalLocation> internal_nodes; 
  // Add pointers to neighboring trees 
};

int main(int argc, char *argv[]) {


  // Things to do:
  // 1. Get a single tree mesh to work with just a map of LogicalLocations
  //    a. Need to perform refinement operations, deal with periodicity  
  // 2. Figure out how to include ghost leaves 
  // 3. Assign positions to leaves based on parent hexahedron
  // 4. Write out tree mesh

  
  Tree tree(2, 2);
  //tree.Print();
  //printf("\n");
  tree.Refine(LogicalLocation(2, 1, 1, 0));
  tree.Refine(LogicalLocation(3, 3, 3, 0));
  tree.Refine(LogicalLocation(4, 7, 7, 0));
  tree.Print(); 
  //printf("\n");
  //ParthenonManager pman;
  
  //LogicalLocation loc;

  //// Simplest possible setup with two blocks with the same orientation sharing one edge 
  //std::unordered_map<uint64_t, std::shared_ptr<Node>> nodes;
  //nodes[0] = std::make_shared<Node>(0, std::array<Real, NDIM>{0.0, 0.0});
  //nodes[1] = std::make_shared<Node>(1, std::array<Real, NDIM>{1.0, 0.0});
  //nodes[2] = std::make_shared<Node>(2, std::array<Real, NDIM>{1.0, 1.0});
  //nodes[3] = std::make_shared<Node>(3, std::array<Real, NDIM>{0.0, 1.0});
  //nodes[4] = std::make_shared<Node>(4, std::array<Real, NDIM>{2.0, 0.0});
  //nodes[5] = std::make_shared<Node>(5, std::array<Real, NDIM>{2.0, 1.0});
  
  //std::vector<std::shared_ptr<Face>> zones;
  //zones.emplace_back(Face::create(sptr_vec_t<Node, 4>{nodes[3], nodes[0], nodes[2], nodes[1]})); 
  //zones.emplace_back(Face::create(sptr_vec_t<Node, 4>{nodes[1], nodes[4], nodes[2], nodes[5]})); 

  //ListFaces(nodes[0]); 
  //ListFaces(nodes[2]); 

  //auto west_neighbors = FindEdgeNeighbors(zones[1], EdgeLoc::West); 
  //auto north_neighbors = FindEdgeNeighbors(zones[1], EdgeLoc::North); 
  //printf("west neighbor loc = %i orientation = %i\n", std::get<1>(west_neighbors[0]), std::get<2>(west_neighbors[0]));
  //printf("north neighbors: %lu\n", north_neighbors.size());

  // MPI and Kokkos can no longer be used
  return 0;
}
