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
#include "mesh/forest/forest.hpp"
#include "mesh/forest/forest_topology.hpp"
#include "mesh/forest/logical_location.hpp"
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

Forest two_blocks() {
  std::unordered_map<uint64_t, std::shared_ptr<Node>> nodes;
  nodes[0] = Node::create(0, {0.0, 0.0});
  nodes[1] = Node::create(1, {1.0, 0.0});
  nodes[2] = Node::create(2, {1.0, 1.0});
  nodes[3] = Node::create(3, {0.0, 1.0});
  nodes[4] = Node::create(4, {2.0, 0.0});
  nodes[5] = Node::create(5, {2.0, 1.0});

  auto &n = nodes;
  ForestDefinition forest_def; 
  auto &faces = forest_def.faces;
  faces.emplace_back(Face::create(0, {n[3], n[0], n[2], n[1]}));
  faces.emplace_back(Face::create(1, {n[1], n[4], n[2], n[5]}));

  auto forest = Forest::Make2D(forest_def);

  // Do some refinements that should propagate into tree 0
  forest.Refine(LogicalLocation(1, 0, 0, 0, 0));
  forest.Refine(LogicalLocation(1, 1, 0, 0, 0));
  forest.Refine(LogicalLocation(1, 2, 0, 0, 0));

  return forest;
}

Forest four_blocks() {
  std::unordered_map<uint64_t, std::shared_ptr<Node>> nodes;
  nodes[0] = Node::create(0, {0.0, 0.0});
  nodes[1] = Node::create(1, {1.0, 0.0});
  nodes[2] = Node::create(2, {1.0, 1.0});
  nodes[3] = Node::create(3, {0.0, 1.0});

  nodes[4] = Node::create(4, {2.0, 0.0});
  nodes[5] = Node::create(5, {2.0, 1.0});

  nodes[6] = Node::create(6, {0.0, 2.0});
  nodes[7] = Node::create(7, {1.0, 2.0});
  nodes[8] = Node::create(8, {2.0, 2.0});

  auto &n = nodes;
  ForestDefinition forest_def; 
  auto &faces = forest_def.faces;
  
  faces.emplace_back(Face::create(0, {n[3], n[0], n[2], n[1]}));
  faces.emplace_back(Face::create(1, {n[1], n[4], n[2], n[5]}));
  faces.emplace_back(Face::create(2, {n[3], n[2], n[6], n[7]}));
  faces.emplace_back(Face::create(3, {n[8], n[7], n[5], n[2]}));

  auto forest = Forest::Make2D(forest_def);

  // Do some refinements that should propagate into tree 0
  forest.Refine(LogicalLocation(1, 0, 0, 0, 0));
  forest.Refine(LogicalLocation(1, 1, 0, 0, 0));
  forest.Refine(LogicalLocation(1, 2, 0, 0, 0));

  forest.Refine(LogicalLocation(0, 1, 0, 1, 0));
  forest.Refine(LogicalLocation(0, 2, 0, 3, 0));

  return forest;
}

Forest n_blocks(int nblocks_min, int nblocks_max) {
  std::unordered_map<uint64_t, std::shared_ptr<Node>> nodes;
  ForestDefinition forest_def; 
  auto &faces = forest_def.faces;
  int nc = 0; 
  int fc = 0;
  Real xoffset = 0.0; 
  for (int nblocks = nblocks_min; nblocks <= nblocks_max; ++nblocks) {
    for (int point = 0; point < 2 * nblocks; ++point) {
      nodes[nc + point] = Node::create(
          nc + point, {std::sin(point * M_PI / nblocks) + xoffset, std::cos(point * M_PI / nblocks)});
    }
    nodes[nc + 2 * nblocks] = Node::create(nc + 2 * nblocks, {0.0 + xoffset, 0.0});
    auto &n = nodes;
    for (int t = 0; t < nblocks; ++t)
      faces.emplace_back(Face::create(
          fc + t, {n[nc + 2 * t + 1],
                   n[nc + 2 * t],
                   n[nc + (2 * t + 2) % (2 * nblocks)],
                   n[nc + 2 * nblocks]}));
    nc += 2 * nblocks + 1;
    fc += nblocks;
    xoffset += 2.2;
  }
  auto forest = Forest::Make2D(forest_def);

  // Do some refinements that should propagate into all trees
  fc = 0;
  for (int nblocks = nblocks_min; nblocks <= nblocks_max; ++nblocks) {
    forest.Refine(LogicalLocation(fc + 2, 0, 0, 0, 0));
    forest.Refine(LogicalLocation(fc + 2, 1, 1, 1, 0));
    forest.Refine(LogicalLocation(fc + 2, 2, 3, 3, 0));
    fc += nblocks;
  }

  return forest;
}

Forest squared_circle() {
  std::unordered_map<uint64_t, std::shared_ptr<Node>> nodes;
  // The outer square
  nodes[0] = Node::create(0, {0.0, 0.0});
  nodes[1] = Node::create(1, {3.0, 0.0});
  nodes[2] = Node::create(2, {0.0, 3.0});
  nodes[3] = Node::create(3, {3.0, 3.0});

  // The inner square
  nodes[4] = Node::create(4, {1.0, 1.0});
  nodes[5] = Node::create(5, {2.0, 1.0});
  nodes[6] = Node::create(6, {1.0, 2.0});
  nodes[7] = Node::create(7, {2.0, 2.0});

  ForestDefinition forest_def; 
  auto &faces = forest_def.faces;
  auto &n = nodes;
  // South block
  faces.emplace_back(Face::create(0, {n[0], n[1], n[4], n[5]}));

  // West block
  faces.emplace_back(Face::create(1, {n[0], n[4], n[2], n[6]}));

  // North block
  faces.emplace_back(Face::create(2, {n[6], n[7], n[2], n[3]}));

  // East block
  faces.emplace_back(Face::create(3, {n[5], n[1], n[7], n[3]}));

  // Center block
  faces.emplace_back(Face::create(4, {n[4], n[5], n[6], n[7]}));
  
  auto forest = Forest::Make2D(forest_def);

  // Do some refinements that should propagate into the south and west trees
  forest.Refine(LogicalLocation(4, 0, 0, 0, 0));
  forest.Refine(LogicalLocation(4, 1, 0, 0, 0));
  forest.Refine(LogicalLocation(4, 2, 0, 0, 0));

  forest.Refine(LogicalLocation(1, 1, 0, 1, 0));
  forest.Refine(LogicalLocation(1, 2, 0, 3, 0));

  return forest;
}

void PrintBlockStructure(std::string fname, std::shared_ptr<Tree> tree) {
  FILE *pFile;
  pFile = fopen(fname.c_str(), "w");
  for (const auto &l : tree->GetSortedMeshBlockList())
    fprintf(pFile, "%i, %li, %li\n", l.level(), l.lx1(), l.lx2());
  fclose(pFile);
}

int main(int argc, char *argv[]) {
  int nblocks_min = 4;
  if (argc > 1) nblocks_min = atoi(argv[1]);
  int nblocks_max = nblocks_min;
  if (argc > 2) nblocks_max = atoi(argv[1]);
  PARTHENON_REQUIRE(nblocks_min > 1, "Need more than one block.");
  auto forest = nblocks_min > 2 ? n_blocks(nblocks_min, nblocks_max) : two_blocks();

  // Write out forest for matplotlib
  FILE *pfile;
  pfile = fopen("faces.txt", "w");
  for (auto &tree : forest.GetTrees()) {
    fprintf(pfile, "%lu", tree->GetId());
    for (auto &n : tree->forest_nodes)
      fprintf(pfile, ", %e, %e", n->x[0], n->x[1]);
    fprintf(pfile, "\n");

    PrintBlockStructure("tree" + std::to_string(tree->GetId()) + ".txt", tree);
  }
  fclose(pfile);

  return 0;
}
