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

#include "bvals/boundary_conditions_generic.hpp"
#include "parthenon_manager.hpp"

#include "poisson_driver.hpp"

using namespace parthenon;
using namespace parthenon::BoundaryFunction;
template <CoordinateDirection DIR, BCSide SIDE>
auto GetBoundaryCondition() {
  return [](std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) -> void {
    using namespace parthenon;
    using namespace parthenon::BoundaryFunction;
    GenericBC<DIR, SIDE, BCType::FixedFace, variable_names::any>(rc, coarse, 0.0);
  };
}

int main(int argc, char *argv[]) {
  using parthenon::ParthenonManager;
  using parthenon::ParthenonStatus;
  ParthenonManager pman;

  // Redefine parthenon defaults
  pman.app_input->ProcessPackages = poisson_example::ProcessPackages;
  pman.app_input->MeshProblemGenerator = poisson_example::ProblemGenerator;

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
  // Now that ParthenonInit has been called and setup succeeded, the code can now
  // make use of MPI and Kokkos

  // Set boundary conditions
  pman.app_input->boundary_conditions[parthenon::BoundaryFace::inner_x1] =
      GetBoundaryCondition<X1DIR, BCSide::Inner>();
  pman.app_input->boundary_conditions[parthenon::BoundaryFace::inner_x2] =
      GetBoundaryCondition<X2DIR, BCSide::Inner>();
  pman.app_input->boundary_conditions[parthenon::BoundaryFace::inner_x3] =
      GetBoundaryCondition<X3DIR, BCSide::Inner>();
  pman.app_input->boundary_conditions[parthenon::BoundaryFace::outer_x1] =
      GetBoundaryCondition<X1DIR, BCSide::Outer>();
  pman.app_input->boundary_conditions[parthenon::BoundaryFace::outer_x2] =
      GetBoundaryCondition<X2DIR, BCSide::Outer>();
  pman.app_input->boundary_conditions[parthenon::BoundaryFace::outer_x3] =
      GetBoundaryCondition<X3DIR, BCSide::Outer>();
  pman.ParthenonInitPackagesAndMesh();

  // This needs to be scoped so that the driver object is destructed before Finalize
  bool success = true;
  {
    // Initialize the driver
    poisson_example::PoissonDriver driver(pman.pinput.get(), pman.app_input.get(),
                                          pman.pmesh.get());

    // This line actually runs the simulation
    auto driver_status = driver.Execute();
    if (driver_status != parthenon::DriverStatus::complete ||
        driver.final_rms_residual > 1.e-10 || driver.final_rms_error > 1.e-12)
      success = false;
  }
  // call MPI_Finalize and Kokkos::finalize if necessary
  pman.ParthenonFinalize();

  // MPI and Kokkos can no longer be used
  return static_cast<int>(!success);
}
