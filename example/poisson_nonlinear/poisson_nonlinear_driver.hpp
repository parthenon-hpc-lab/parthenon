//========================================================================================
// (C) (or copyright) 2021. Triad National Security, LLC. All rights reserved.
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

#ifndef EXAMPLE_POISSON_CG_POISSON_CG_DRIVER_HPP_
#define EXAMPLE_POISSON_CG_POISSON_CG_DRIVER_HPP_

#include <memory>
#include <vector>

#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

namespace poisson_example {
using namespace parthenon::driver::prelude;

class PoissonDriver : public Driver {
 public:
  PoissonDriver(ParameterInput *pin, ApplicationInput *app_in, Mesh *pm)
      : Driver(pin, app_in, pm) {
    InitializeOutputs();
  }
  // This next function essentially defines the driver.
  TaskCollection MakeTaskCollection(BlockList_t &blocks);

  DriverStatus Execute() override;

 private:
  // we'll demonstrate doing a global all reduce of a scalar
  AllReduce<Real> total_mass;
  // and a reduction onto one rank of a scalar
  Reduce<int> max_rank;
  // and we'll do an all reduce of a vector just for fun
  AllReduce<std::vector<int>> vec_reduce;
};

void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
parthenon::Packages_t ProcessPackages(std::unique_ptr<parthenon::ParameterInput> &pin);

} // namespace poisson_example

#endif // EXAMPLE_POISSON_CG_POISSON_CG_DRIVER_HPP_
