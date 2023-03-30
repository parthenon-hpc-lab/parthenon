//========================================================================================
// (C) (or copyright) 2020-2022. Triad National Security, LLC. All rights reserved.
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

#include "particles.hpp"

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
  pman.app_input->ProblemGenerator = particles_example::ProblemGenerator;
  if (pman.pinput->GetString("parthenon/mesh", "ix1_bc") == "user") {
    // In this case, we are setting a custom swarm boundary condition while still using
    // a default parthenon boundary condition for cell variables. In general, one can
    // provide both custom cell variable and swarm boundary conditions. However, to use
    // custom boundary conditions for either cell variables or swarms, the parthenon
    // boundary must be set to "user" and both cell variable and swarm boundaries provided
    // as here.
    pman.app_input->boundary_conditions[parthenon::BoundaryFace::inner_x1] =
        parthenon::BoundaryFunction::OutflowInnerX1;
    pman.app_input->swarm_boundary_conditions[parthenon::BoundaryFace::inner_x1] =
        particles_example::SetSwarmIX1UserBC;
  }
  if (pman.pinput->GetString("parthenon/mesh", "ox1_bc") == "user") {
    // Again, we use a default parthenon boundary condition for cell variables but a
    // custom swarm boundary condition.
    pman.app_input->boundary_conditions[parthenon::BoundaryFace::outer_x1] =
        parthenon::BoundaryFunction::OutflowOuterX1;
    pman.app_input->swarm_boundary_conditions[parthenon::BoundaryFace::outer_x1] =
        particles_example::SetSwarmOX1UserBC;
  }
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
