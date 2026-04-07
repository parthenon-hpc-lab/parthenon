//========================================================================================
// (C) (or copyright) 2021-2025. Triad National Security, LLC. All rights reserved.
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

#ifndef EXAMPLE_DIFFUSION_DIFFUSION_DRIVER_HPP_
#define EXAMPLE_DIFFUSION_DIFFUSION_DRIVER_HPP_

#include <memory>
#include <vector>

#include "diffusion_hypre.hpp"
#include <kokkos_abstraction.hpp>
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>
#include <solvers/solver_base.hpp>

namespace diffusion_example {
using namespace parthenon::driver::prelude;
using namespace parthenon;

class DiffusionDriver : public EvolutionDriver {
 public:
  DiffusionDriver(ParameterInput *pin, ApplicationInput *app_in, Mesh *pm)
      : EvolutionDriver(pin, app_in, pm), integrator(pin) {
    u2.val = 1e200;
    //    InitializeOutputs();
  }
  // This next function essentially defines the driver.
  TaskCollection MakeTaskCollection();
  TaskCollection MakeTaskCollectionHypre();
  TaskCollection MakeTaskCollectionNative();
  TaskListStatus Step() override;

  // DriverStatus Execute() override;
  void OutputDownstreamCycleDiagnostics() override {
    auto pkg = pmesh->packages.Get("diffusion_package");
    bool print{true};
#ifdef DIFFUSION_WITH_HYPRE
    if (pkg->Param<bool>("use_hypre")) {
      auto hypre_solver =
          pkg->Param<std::shared_ptr<diffusion_package::HypreSolver>>("hypre_solver");
      std::cout << " v-cycles=" << hypre_solver->niter * 2
                << " rel_resid=" << hypre_solver->rnorm;
      print = false;
    }
#endif
    if (print) {
      auto solver_type = pkg->Param<std::string>("solver");
      auto psolver =
          pkg->Param<std::shared_ptr<parthenon::solvers::SolverBase>>("solver_pointer");
      const auto alpha = pkg->Param<Real>("diagonal_alpha");
      int v_cycles = psolver->GetFinalIterations();
      auto res = psolver->GetFinalResidual();
      if (solver_type == "BiCGSTAB") v_cycles *= 2;
      std::cout << " v-cycles=" << v_cycles
                << " rel_resid=" << res / (alpha * sqrt(u2.val));
    }
  }

  void PostExecute(DriverStatus status) override {
    EvolutionDriver::PostExecute(status);
    if (parthenon::Globals::my_rank == 0) {
      auto pkg = pmesh->packages.Get("diffusion_package");
      if (pkg->Param<bool>("report_timings")) {
        printf("\nTiming data\n-----------\n");
        auto psolver =
            pkg->Param<std::shared_ptr<parthenon::solvers::SolverBase>>("solver_pointer");
        std::cout << "Solver breakdown: \n" << psolver->solver_timings;
        psolver->solver_timings.clear();
      }
    }
  }

 private:
  LowStorageIntegrator integrator;
  parthenon::AllReduce<Real> u2;
};

void ProblemGenerator(Mesh *pm, parthenon::ParameterInput *pin, MeshData<Real> *md);
parthenon::Packages_t ProcessPackages(std::unique_ptr<parthenon::ParameterInput> &pin);

} // namespace diffusion_example

#endif // EXAMPLE_DIFFUSION_DIFFUSION_DRIVER_HPP_
