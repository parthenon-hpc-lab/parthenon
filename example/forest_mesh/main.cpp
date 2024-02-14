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

int main(int argc, char *argv[]) {
  // Simplest possible setup with two blocks with the same orientation sharing one edge 
  std::unordered_map<uint64_t, std::shared_ptr<Node>> nodes;
  nodes[0] = Node::create(0, {0.0, 0.0});
  nodes[1] = Node::create(1, {1.0, 0.0});
  nodes[2] = Node::create(2, {1.0, 1.0});
  nodes[3] = Node::create(3, {0.0, 1.0});
  nodes[4] = Node::create(4, {2.0, 0.0});
  nodes[5] = Node::create(5, {2.0, 1.0});
  
  std::vector<std::shared_ptr<Face>> zones;
  zones.emplace_back(Face::create({nodes[3], nodes[0], nodes[2], nodes[1]})); 
  zones.emplace_back(Face::create({nodes[1], nodes[4], nodes[2], nodes[5]})); 

  for (auto & zone : zones) { 
    for (auto side : {EdgeLoc::North, EdgeLoc::East, EdgeLoc::South, EdgeLoc::West}) {
      auto neighbors = FindEdgeNeighbors(zone, side); 
      for (auto &n : neighbors) { 
        auto orient = RelativeOrientation::FromSharedEdge2D(side, std::get<1>(n), std::get<2>(n));
        zone->tree->AddNeighbor(side.GetFaceIdx2D(), std::get<0>(n)->tree, orient);
      } 
    }
  }
  zones[1]->tree->Refine(LogicalLocation(0, 0, 0, 0));
  zones[1]->tree->Refine(LogicalLocation(1, 0, 0, 0));
  zones[1]->tree->Refine(LogicalLocation(2, 0, 0, 0));
  
  // Write out forest for matplotlib
  FILE *pfile; 
  pfile = fopen("faces.txt", "w");
  int z = 0;
  for (auto &zone : zones) {
    fprintf(pfile, "%i", z);
    for (auto & n : zone->nodes) { 
      fprintf(pfile, ", %e, %e", n->x[0], n->x[1]);
    }
    fprintf(pfile, "\n");

    zone->tree->Print("tree" + std::to_string(z) + ".txt");
    z++;
  }
  fclose(pfile);

  return 0;
}
