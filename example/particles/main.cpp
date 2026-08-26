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

#include "parthenon_manager.hpp"

#include "bvals/boundary_conditions.hpp"
#include "bvals/boundary_conditions_generic.hpp"
#include "particles.hpp"

using namespace parthenon::BoundaryFunction;

// Example inner boundary condition (this just reuses existing features) to show how to
// create and enroll a user swarm boundary condition. Note that currently both Swarm and
// field boundary conditions must be provided when "user" is specified.
// Note that BCType::Periodic cannot be enrolled as a user boundary condition.
void SwarmUserInnerX1(std::shared_ptr<Swarm> &swarm) {
  GenericSwarmBC<X1DIR, BCSide::Inner, BCType::Outflow>(swarm);
}

void SwarmUserOuterX1(std::shared_ptr<Swarm> &swarm) {
  GenericSwarmBC<X1DIR, BCSide::Outer, BCType::Outflow>(swarm);
}

int main(int argc, char *argv[]) {
  using parthenon::ParthenonManager;
  using parthenon::ParthenonStatus;
  ParthenonManager pman;

  // call ParthenonInit to initialize MPI and Kokkos, parse the input deck, and set up
  auto manager_status = pman.ParthenonInitEnv(argc, argv);
  if (manager_status == ParthenonStatus::complete) {
    pman.ParthenonFinalize();
    return 0;
  }
  if (manager_status == ParthenonStatus::error) {
    pman.ParthenonFinalize();
    return 1;
  }

  // Redefine parthenon defaults
  pman.app_input->ProcessPackages = particles_example::ProcessPackages;
  pman.app_input->RegisterBoundaryCondition(BoundaryFace::inner_x1,
                                            parthenon::BoundaryFunction::OutflowInnerX1);
  pman.app_input->RegisterBoundaryCondition(BoundaryFace::outer_x1,
                                            parthenon::BoundaryFunction::OutflowOuterX1);
  pman.app_input->RegisterSwarmBoundaryCondition(BoundaryFace::inner_x1,
                                                 SwarmUserInnerX1);
  pman.app_input->RegisterSwarmBoundaryCondition(BoundaryFace::outer_x1,
                                                 SwarmUserOuterX1);
  // Note that this example does not use a ProblemGenerator
  pman.ParthenonInitPackagesAndMesh();

  // This needs to be scoped so that the driver object is destructed before Finalize
  {
    // Initialize the driver
    particles_example::ParticleDriver driver(pman.pinput.get(), pman.app_input.get(),
                                             pman.pmesh.get());

    // This line actually runs the simulation
    auto driver_status = driver.Execute();
  }
  // call MPI_Finalize and Kokkos::finalize if necessary
  pman.ParthenonFinalize();

  // MPI and Kokkos can no longer be used

  return (0);
}
