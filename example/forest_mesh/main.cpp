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
#include "defs.hpp"
#include "forest_topology.hpp"
#include "mesh/forest.hpp"
#include "mesh/logical_location.hpp"
#include "parthenon_manager.hpp"

using parthenon::LogicalLocation;
using parthenon::ParthenonManager;
using parthenon::ParthenonStatus;
using parthenon::Real;
using parthenon::RegionSize;
using parthenon::CoordinateDirection::X1DIR;
using parthenon::CoordinateDirection::X2DIR;
using parthenon::CoordinateDirection::X3DIR;
using namespace parthenon::forest;

struct mesh_t {
  std::unordered_map<uint64_t, std::shared_ptr<Node>> nodes;
  std::vector<std::shared_ptr<Face>> zones;

  void SetTreeConnections() {
    for (auto &zone : zones) {
      for (auto side : {EdgeLoc::North, EdgeLoc::East, EdgeLoc::South, EdgeLoc::West}) {
        auto neighbors = FindEdgeNeighbors(zone, side);
        for (auto &n : neighbors) {
          auto orient =
              RelativeOrientationFromSharedEdge2D(side, std::get<1>(n), std::get<2>(n));
          zone->tree->AddNeighborTree(side.GetFaceIdx2D(), std::get<0>(n)->tree, orient);
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
  mesh.zones.emplace_back(Face::create(0, {n[3], n[0], n[2], n[1]}));
  mesh.zones.emplace_back(Face::create(1, {n[1], n[4], n[2], n[5]}));

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
  mesh.zones.emplace_back(Face::create(0, {n[0], n[1], n[4], n[5]}));

  // West block
  mesh.zones.emplace_back(Face::create(1, {n[0], n[4], n[2], n[6]}));

  // North block
  mesh.zones.emplace_back(Face::create(2, {n[6], n[7], n[2], n[3]}));

  // East block
  mesh.zones.emplace_back(Face::create(3, {n[5], n[1], n[7], n[3]}));

  // Center block
  mesh.zones.emplace_back(Face::create(4, {n[4], n[5], n[6], n[7]}));

  mesh.SetTreeConnections();

  // Do some refinements that should propagate into the south and west trees
  mesh.zones[4]->tree->Refine(LogicalLocation(0, 0, 0, 0));
  mesh.zones[4]->tree->Refine(LogicalLocation(1, 0, 0, 0));
  mesh.zones[4]->tree->Refine(LogicalLocation(2, 0, 0, 0));

  mesh.zones[1]->tree->Refine(LogicalLocation(1, 0, 1, 0));
  mesh.zones[1]->tree->Refine(LogicalLocation(2, 0, 3, 0));

  return mesh;
}

void PrintBlockStructure(std::string fname, std::shared_ptr<Tree> tree) {
  FILE *pFile;
  pFile = fopen(fname.c_str(), "w");
  for (const auto &[l, gid] : tree->GetLeaves())
    fprintf(pFile, "%i, %i, %i\n", l.level(), l.lx1(), l.lx2());
  fclose(pFile);
}

int main(int argc, char *argv[]) {
  auto mesh = squared_circle();

  // Write out forest for matplotlib
  FILE *pfile;
  pfile = fopen("faces.txt", "w");
  int z = 0;
  for (auto &zone : mesh.zones) {
    fprintf(pfile, "%i", z);
    for (auto &n : zone->nodes) {
      fprintf(pfile, ", %e, %e", n->x[0], n->x[1]);
    }
    fprintf(pfile, "\n");

    PrintBlockStructure("tree" + std::to_string(z) + ".txt", zone->tree);
    z++;
  }
  fclose(pfile);

  RegionSize mesh_size({0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}, {1.0, 1.0, 1.0}, {64, 128, 1},
                       {false, false, true});
  RegionSize block_size(mesh_size);
  block_size.nx(X1DIR) = 16;
  block_size.nx(X2DIR) = 16;

  std::array<parthenon::BoundaryFlag, parthenon::BOUNDARY_NFACES> bcs{
      parthenon::BoundaryFlag::outflow, parthenon::BoundaryFlag::outflow,
      parthenon::BoundaryFlag::outflow, parthenon::BoundaryFlag::outflow,
      parthenon::BoundaryFlag::outflow, parthenon::BoundaryFlag::outflow};

  auto forest = Forest::AthenaXX(mesh_size, block_size, bcs);

  printf("ntrees: %i\n", forest.trees.size());
  auto block_list = forest.GetMeshBlockListAndResolveGids();
  printf("number of blocks = %i\n", block_list.size());
  pfile = fopen("faces.txt", "w");
  for (uint64_t gid = 0; gid < block_list.size(); ++gid) {
    auto dmn = forest.GetBlockDomain(block_list[gid]);
    fprintf(pfile, "%i, %e, %e, %e, %e, %e, %e, %e, %e\n", gid, dmn.xmin(X1DIR),
            dmn.xmin(X2DIR), dmn.xmax(X1DIR), dmn.xmin(X2DIR), dmn.xmin(X1DIR),
            dmn.xmax(X2DIR), dmn.xmax(X1DIR), dmn.xmax(X2DIR));
  }
  fclose(pfile);

  /*
  for (uint64_t gid = 0; gid < block_list.size(); ++gid) {
    for (int ox1 : {-1, 0, 1}) {
      for (int ox2 : {-1, 0, 1}) {
        auto neigh_vec = forest.FindNeighbor(block_list[gid], ox1, ox2, 0);
        for (auto &neigh : neigh_vec) {
          auto ngid = gid_map[neigh.global_loc.global_loc][neigh.global_loc.origin_loc];
          if (ngid != gid) {
            printf("%i -> %i\n", gid, ngid);
          }
        }
      }
    }
  }
  */

  return 0;
}
