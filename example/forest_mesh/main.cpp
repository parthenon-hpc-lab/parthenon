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

struct RelativeOrientation { 
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

int main(int argc, char *argv[]) {
  // Things to do:
  // 1. Get a single tree mesh to work with just a map of LogicalLocations
  //    a. Need to perform refinement operations, deal with periodicity  
  // 2. Figure out how to include ghost leaves 
  // 3. Assign positions to leaves based on parent hexahedron
  // 4. Write out tree mesh
  auto tree1 = std::make_shared<Tree>(2, 1);
  auto tree2 = std::make_shared<Tree>(2, 1);
  auto tree3 = std::make_shared<Tree>(2, 1);

  RelativeOrientation tree1to2; 
  tree1to2.dir_connection[0] = 1; 
  tree1to2.dir_connection[1] = 0; 
  tree1to2.dir_connection[2] = 2; 
  tree1to2.dir_flip[0] = true;
  tree1to2.dir_flip[1] = false;
  tree1to2.dir_flip[2] = false;
   
  RelativeOrientation tree2to1;
  tree2to1.dir_connection[0] = 1; 
  tree2to1.dir_connection[1] = 0; 
  tree2to1.dir_connection[2] = 2; 
  tree2to1.dir_flip[0] = false;
  tree2to1.dir_flip[1] = true;
  tree2to1.dir_flip[2] = false;

  RelativeOrientation tree1to3; 
  tree1to3.dir_connection[0] = 0; 
  tree1to3.dir_connection[1] = 1; 
  tree1to3.dir_connection[2] = 2; 
  tree1to3.dir_flip[0] = false;
  tree1to3.dir_flip[1] = false;
  tree1to3.dir_flip[2] = false;
  RelativeOrientation tree3to1 = tree1to3; 

  RelativeOrientation tree2to3; 
  tree2to3.dir_connection[0] = 0; 
  tree2to3.dir_connection[1] = 1; 
  tree2to3.dir_connection[2] = 2; 
  tree2to3.dir_flip[0] = true;
  tree2to3.dir_flip[1] = true;
  tree2to3.dir_flip[2] = false;
  
  RelativeOrientation tree3to2 = tree2to3;

  tree1->AddNeighbor(2 + 3 * 1 + 9 * 1, tree2, tree1to2);
  tree1->AddNeighbor(1 + 3 * 2 + 9 * 1, tree3, tree1to3); 
  
  tree2->AddNeighbor(1 + 3 * 2 + 9 * 1, tree1, tree2to1);
  tree2->AddNeighbor(2 + 3 * 1 + 9 * 1, tree3, tree2to3);

  tree3->AddNeighbor(1 + 3 * 0 + 9 * 1, tree1, tree3to1);
  tree3->AddNeighbor(2 + 3 * 1 + 9 * 1, tree2, tree3to2);

  //tree.Print();
  //printf("\n");
  tree1->Refine(LogicalLocation(1, 1, 1, 0));
  tree1->Refine(LogicalLocation(2, 3, 3, 0));
  //tree1->Refine(LogicalLocation(3, 3, 3, 0));
  //tree1->Refine(LogicalLocation(4, 7, 7, 0));
  tree1->Print("tree1.txt");
  tree2->Print("tree2.txt");
  tree3->Print("tree3.txt");

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
