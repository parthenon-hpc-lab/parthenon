//========================================================================================
// (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
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

#ifndef EXAMPLE_POISSON_POISSON_DRIVER_HPP_
#define EXAMPLE_POISSON_POISSON_DRIVER_HPP_

#include "driver/driver.hpp"

#include "parameter_input.hpp"
#include "parthenon_mpi.hpp"
#include "poisson_package.hpp"

namespace poisson {
using namespace parthenon::driver::prelude

class PoissonDriver : public IterationDriver {
 public:
  PoissonDriver(ParameterInput *pin, Mesh *pm) : Driver(pin, pm) {
    max_residual = pinput->GetReal("parthenon/iterations","residual");
    // TODO(JMM): inputs 
  }
  TaskListStatus Step() {
    return DriverUtils::ConstructAndExecuteBlockTasks<>(this);
  }
  bool KeepGoing() {
    residual = 0;
    MeshBlock *pmb pmesh->pblock;
    while (pmb != nullptr) {
      Container<Real> &rc = pmb->real_containers.Get();
      Real block_residual = GetResidual(rc);
      residual = block_residual > residual? block_residual : residual;
    }

#ifdef MPI_PARALLEL
    MPI_Allreduce(MPI_IN_PLACE, &residual, 1,
                  MPI_PARTHENON_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif
    return residual > max_residual;
  }
  TaskList MakeTaskList(MeshBlock *pmb);
  void OutputCycleDiagnostics();
  Real residual = 0;
  Real max_residual;
};

} // namespace poisson

#endif // EXAMPLE_POISSON_POISSON_DRIVER_HPP_
