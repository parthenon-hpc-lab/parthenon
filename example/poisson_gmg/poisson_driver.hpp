//========================================================================================
// (C) (or copyright) 2021-2023. Triad National Security, LLC. All rights reserved.
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
  TaskCollection MakeTaskCollectionProRes(BlockList_t &blocks);
  TaskCollection MakeTaskCollectionMG(BlockList_t &blocks);
  TaskCollection MakeTaskCollectionMGCG(BlockList_t &blocks);
  TaskCollection MakeTaskCollectionMGBiCGSTAB(BlockList_t &blocks);

  DriverStatus Execute() override;

  template <class TL_t>
  TaskID AddMultiGridTasksPartitionLevel(TL_t &region, TaskID dependence, int partition,
                                         int level, int min_level, int max_level,
                                         bool final);
  void AddRestrictionProlongationLevel(TaskRegion &region, int level, int min_level,
                                       int max_level);

  Real final_rms_error, final_rms_residual;

 private:
  // Necessary reductions for checking error from exact solution
  AllReduce<Real> err;
};

void ProblemGenerator(Mesh *pm, parthenon::ParameterInput *pin, MeshData<Real> *md);
parthenon::Packages_t ProcessPackages(std::unique_ptr<parthenon::ParameterInput> &pin);

} // namespace poisson_example

#endif // EXAMPLE_POISSON_GMG_POISSON_DRIVER_HPP_
