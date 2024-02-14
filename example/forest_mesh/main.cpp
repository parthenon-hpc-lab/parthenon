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


struct mesh_t { 
  std::unordered_map<uint64_t, std::shared_ptr<Node>> nodes;
  std::vector<std::shared_ptr<Face>> zones;

  void SetTreeConnections() { 
    for (auto & zone : zones) { 
      for (auto side : {EdgeLoc::North, EdgeLoc::East, EdgeLoc::South, EdgeLoc::West}) {
        auto neighbors = FindEdgeNeighbors(zone, side); 
        for (auto &n : neighbors) { 
          auto orient = RelativeOrientation::FromSharedEdge2D(side, std::get<1>(n), std::get<2>(n));
          zone->tree->AddNeighbor(side.GetFaceIdx2D(), std::get<0>(n)->tree, orient);
        } 
      }
    }
  } 
};

mesh_t two_blocks() { 
  mesh_t mesh; 
  mesh.nodes[0] = Node::create(0, {0.0, 0.0});
  mesh.nodes[1] = Node::create(1, {1.0, 0.0});
  mesh.nodes[2] = Node::create(2, {1.0, 1.0});
  mesh.nodes[3] = Node::create(3, {0.0, 1.0});
  mesh.nodes[4] = Node::create(4, {2.0, 0.0});
  mesh.nodes[5] = Node::create(5, {2.0, 1.0});
  
  auto &n = mesh.nodes; 
  mesh.zones.emplace_back(Face::create({n[3], n[0], n[2], n[1]})); 
  mesh.zones.emplace_back(Face::create({n[1], n[4], n[2], n[5]}));  
  
  mesh.SetTreeConnections();
  // Do some refinements that should propagate into tree 1
  mesh.zones[1]->tree->Refine(LogicalLocation(0, 0, 0, 0));
  mesh.zones[1]->tree->Refine(LogicalLocation(1, 0, 0, 0));
  mesh.zones[1]->tree->Refine(LogicalLocation(2, 0, 0, 0));

  return mesh;
}

mesh_t squared_circle() { 
  mesh_t mesh; 
  // The outer square
  mesh.nodes[0] = Node::create(0, {0.0, 0.0});
  mesh.nodes[1] = Node::create(1, {3.0, 0.0});
  mesh.nodes[2] = Node::create(2, {0.0, 3.0});
  mesh.nodes[3] = Node::create(3, {3.0, 3.0});
  
  // The inner square
  mesh.nodes[4] = Node::create(4, {1.0, 1.0});
  mesh.nodes[5] = Node::create(5, {2.0, 1.0});
  mesh.nodes[6] = Node::create(6, {1.0, 2.0});
  mesh.nodes[7] = Node::create(7, {2.0, 2.0});

  
  auto &n = mesh.nodes; 
  // South block 
  mesh.zones.emplace_back(Face::create({n[0], n[1], n[4], n[5]}));
  
  // West block 
  mesh.zones.emplace_back(Face::create({n[0], n[4], n[2], n[6]}));

  // North block 
  mesh.zones.emplace_back(Face::create({n[6], n[7], n[2], n[3]}));

  // East block 
  mesh.zones.emplace_back(Face::create({n[5], n[1], n[7], n[3]}));

  // Center block 
  mesh.zones.emplace_back(Face::create({n[4], n[5], n[6], n[7]}));

  mesh.SetTreeConnections();
  
  // Do some refinements that should propagate into the south and west trees
  mesh.zones[4]->tree->Refine(LogicalLocation(0, 0, 0, 0));
  mesh.zones[4]->tree->Refine(LogicalLocation(1, 0, 0, 0));
  mesh.zones[4]->tree->Refine(LogicalLocation(2, 0, 0, 0));

  mesh.zones[1]->tree->Refine(LogicalLocation(1, 0, 1, 0));
  mesh.zones[1]->tree->Refine(LogicalLocation(2, 0, 3, 0));

  return mesh;
}

int main(int argc, char *argv[]) {
  auto mesh = squared_circle();
  
  // Write out forest for matplotlib
  FILE *pfile; 
  pfile = fopen("faces.txt", "w");
  int z = 0;
  for (auto &zone : mesh.zones) {
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
