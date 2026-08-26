//========================================================================================
// (C) (or copyright) 2020-2024. Triad National Security, LLC. All rights reserved.
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

// Standard Includes
#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// Parthenon Includes
#include <parthenon/driver.hpp>

// Local Includes
#include "boundary_exchange.hpp"
#include "boundary_exchange_driver.hpp"
#include "mesh/forest/forest_node.hpp"
#include "mesh/forest/forest_topology.hpp"

// Preludes
using namespace parthenon::driver::prelude;

using boundary_exchange::BoundaryExchangeDriver;

Packages_t ProcessPackages(std::unique_ptr<ParameterInput> &pin);

int main(int argc, char *argv[]) {
  ParthenonManager pman;

  pman.app_input->ProcessPackages = ProcessPackages;

  // This is called on each mesh block whenever the mesh changes.
  // pman.app_input->InitMeshBlockUserData = &calculate_pi::SetInOrOutBlock;

  auto manager_status = pman.ParthenonInitEnv(argc, argv);
  if (manager_status == ParthenonStatus::complete) {
    pman.ParthenonFinalize();
    return 0;
  }
  if (manager_status == ParthenonStatus::error) {
    pman.ParthenonFinalize();
    return 1;
  }

  // Create the nodes for the forest, the x-y positions are only used for
  // visualizing the forest configuration and *do not* determine the global
  // coordinates of the trees.
  std::unordered_map<uint64_t, std::shared_ptr<parthenon::forest::Node>> nodes;
  nodes[0] = parthenon::forest::Node::create(0, {0.0, 0.0});
  nodes[1] = parthenon::forest::Node::create(1, {1.0, 0.0});
  nodes[2] = parthenon::forest::Node::create(2, {1.0, 1.0});
  nodes[3] = parthenon::forest::Node::create(3, {0.0, 1.0});
  nodes[4] = parthenon::forest::Node::create(4, {2.0, 0.0});
  nodes[5] = parthenon::forest::Node::create(5, {2.0, 1.0});
  nodes[6] = parthenon::forest::Node::create(6, {0.0, 2.0});
  nodes[7] = parthenon::forest::Node::create(7, {1.0, 2.0});
  nodes[8] = parthenon::forest::Node::create(8, {2.0, 2.0});

  auto &n = nodes;
  parthenon::forest::ForestDefinition forest_def;

  using edge_t = parthenon::forest::Edge;
  using ar3_t = std::array<Real, 3>;
  using ai3_t = std::array<int, 3>;
  forest_def.AddFace(0, {n[1], n[2], n[0], n[3]}, ar3_t{0.0, 0.0, 0.0},
                     ar3_t{1.0, 1.0, 1.0});
  // forest_def.AddFace(0, {n[0], n[1], n[3], n[2]}, ar3_t{0.0, 0.0, 0.0},
  // ar3_t{1.0, 1.0, 1.0});
  forest_def.AddBC(edge_t({n[0], n[1]}));
  forest_def.AddBC(edge_t({n[0], n[3]}));

  forest_def.AddFace(1, {n[1], n[4], n[2], n[5]}, ar3_t{2.0, 0.0, 0.0},
                     ar3_t{3.0, 1.0, 1.0});
  forest_def.AddBC(edge_t({n[1], n[4]}));
  forest_def.AddBC(edge_t({n[4], n[5]}));

  forest_def.AddFace(3, {n[3], n[2], n[6], n[7]}, ar3_t{0.0, 2.0, 0.0},
                     ar3_t{1.0, 3.0, 1.0});
  forest_def.AddBC(edge_t({n[6], n[7]}));
  forest_def.AddBC(edge_t({n[3], n[6]}));

  forest_def.AddFace(4, {n[2], n[5], n[7], n[8]}, ar3_t{2.0, 2.0, 0.0},
                     ar3_t{3.0, 3.0, 1.0});
  forest_def.AddBC(edge_t({n[5], n[8]}));
  forest_def.AddBC(edge_t({n[7], n[8]}));

  forest_def.AddInitialRefinement(parthenon::LogicalLocation(0, 1, 0, 0, 0));
  pman.ParthenonInitPackagesAndMesh(forest_def);

  // This needs to be scoped so that the driver object is destructed before Finalize
  {
    BoundaryExchangeDriver driver(pman.pinput.get(), pman.app_input.get(),
                                  pman.pmesh.get());

    auto driver_status = driver.Execute();
  }
  // call MPI_Finalize if necessary
  pman.ParthenonFinalize();

  return 0;
}

Packages_t ProcessPackages(std::unique_ptr<ParameterInput> &pin) {
  Packages_t packages;
  // only have one package for this app, but will typically have more things added to
  packages.Add(boundary_exchange::Initialize(pin.get()));
  return packages;
}

// this should set up initial conditions of independent variables on the block
// this app only has one variable of derived type, so nothing to do here.
// in this case, just use the weak version
// void MeshBlock::ProblemGenerator(ParameterInput *pin) {
//  // nothing to do here for this app
//}

parthenon::DriverStatus BoundaryExchangeDriver::Execute() {
  // this is where the main work is orchestrated
  // No evolution in this driver.  Just calculates something once.
  // For evolution, look at the EvolutionDriver
  // PreExecute();

  // The tasks compute pi and store it in the param "pi_val"
  ConstructAndExecuteTaskLists<>(this);
  pouts->MakeOutputs(pmesh, pinput);

  return DriverStatus::complete;
}

template <typename T>
TaskCollection BoundaryExchangeDriver::MakeTaskCollection(T &blocks) {
  const int num_partitions = pmesh->DefaultNumPartitions();

  TaskCollection tc;
  TaskRegion &async_region = tc.AddRegion(num_partitions);
  {
    // asynchronous region where area is computed per partition
    for (int i = 0; i < num_partitions; i++) {
      auto &tl = async_region[i];
      auto &md = pmesh->mesh_data.GetOrAdd("base", i);
      TaskID none(0);
      auto fill = tl.AddTask(none, SetBlockValues, md.get());
      auto bound = AddBoundaryExchangeTasks(fill, tl, md, true);
    }
  }

  // Synchronize

  // Check that boundaries have correct values

  return tc;
}
