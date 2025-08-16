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

#include <kokkos_abstraction.hpp>
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

namespace diffusion_example {
using namespace parthenon::driver::prelude;
using namespace parthenon;

class DiffusionDriver : public EvolutionDriver {
 public:
  DiffusionDriver(ParameterInput *pin, ApplicationInput *app_in, Mesh *pm)
      : EvolutionDriver(pin, app_in, pm), integrator(pin) {
    //    InitializeOutputs();
  }
  // This next function essentially defines the driver.
  TaskCollection MakeTaskCollection();
  TaskListStatus Step();

  // DriverStatus Execute() override;

 private:
  LowStorageIntegrator integrator;
};

void ProblemGenerator(Mesh *pm, parthenon::ParameterInput *pin, MeshData<Real> *md);
parthenon::Packages_t ProcessPackages(std::unique_ptr<parthenon::ParameterInput> &pin);

} // namespace diffusion_example

#endif // EXAMPLE_DIFFUSION_DIFFUSION_DRIVER_HPP_
