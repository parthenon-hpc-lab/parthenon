//========================================================================================
// (C) (or copyright) 2021-2022. Triad National Security, LLC. All rights reserved.
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

#ifndef EXAMPLE_POISSON_GMG_POISSON_DRIVER_HPP_
#define EXAMPLE_POISSON_GMG_POISSON_DRIVER_HPP_

#include <memory>
#include <vector>

#include <kokkos_abstraction.hpp>
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

  void AddMultiGridTasks(TaskCollection &tc, int level, int max_level);

 private:
  // we'll demonstrate doing a global all reduce of a scalar There
  // must be one (All)Reduce object per var per rank, and they must be
  // in appropriate scope so that they don't get
  // garbage-collected... e.g., the objects persist between and
  // accross task lists. A natural place is here in the driver. But
  // the data they point to might need to live in the params of a
  // package, as we've done here.
  AllReduce<Real> total_mass;
  AllReduce<Real> update_norm;
  // and a reduction onto one rank of a scalar
  Reduce<int> max_rank;
  // and we'll do an all reduce of a vector just for fun
  AllReduce<std::vector<int>> vec_reduce;
  // We reduce a view too, but it's stored as a param.
};

void ProblemGenerator(Mesh *pm, parthenon::ParameterInput *pin, MeshData<Real> *md);
parthenon::Packages_t ProcessPackages(std::unique_ptr<parthenon::ParameterInput> &pin);

} // namespace poisson_example

#endif // EXAMPLE_POISSON_GMG_POISSON_DRIVER_HPP_
