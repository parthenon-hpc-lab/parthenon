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

int main(int argc, char *argv[]) {
  using parthenon::ParthenonManager;
  using parthenon::ParthenonStatus;
  using parthenon::LogicalLocation;
  using parthenon::Real;
  using namespace parthenon::forest;

  ParthenonManager pman;
  
  LogicalLocation loc;

  // Simplest possible setup with two blocks with the same orientation sharing one edge 
  std::unordered_map<uint64_t, std::shared_ptr<Node>> nodes;
  nodes[0] = std::make_shared<Node>(0, std::array<Real, NDIM>{0.0, 0.0});
  nodes[1] = std::make_shared<Node>(1, std::array<Real, NDIM>{1.0, 0.0});
  nodes[2] = std::make_shared<Node>(2, std::array<Real, NDIM>{1.0, 1.0});
  nodes[3] = std::make_shared<Node>(3, std::array<Real, NDIM>{0.0, 1.0});
  nodes[4] = std::make_shared<Node>(4, std::array<Real, NDIM>{2.0, 0.0});
  nodes[5] = std::make_shared<Node>(5, std::array<Real, NDIM>{2.0, 1.0});
  
  std::vector<std::shared_ptr<Face>> zones;
  zones.emplace_back(Face::create(sptr_vec_t<Node, 4>{nodes[3], nodes[0], nodes[2], nodes[1]})); 
  zones.emplace_back(Face::create(sptr_vec_t<Node, 4>{nodes[1], nodes[4], nodes[2], nodes[5]})); 

  ListFaces(nodes[0]); 
  ListFaces(nodes[2]); 

  auto west_neighbors = FindEdgeNeighbors(zones[1], EdgeLoc::West); 
  auto north_neighbors = FindEdgeNeighbors(zones[1], EdgeLoc::North); 
  printf("west neighbor loc = %i orientation = %i\n", std::get<1>(west_neighbors[0]), std::get<2>(west_neighbors[0]));
  printf("north neighbors: %lu\n", north_neighbors.size());

  // MPI and Kokkos can no longer be used
  return 0;
}
