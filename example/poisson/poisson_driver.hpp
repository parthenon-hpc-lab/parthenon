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

#include <cmath>
#include <string>

#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

#include "poisson_package.hpp"
#include "tasks/task_list.hpp"
#include "utils/error_checking.hpp"

namespace poisson {
using namespace parthenon::driver::prelude;
using namespace parthenon::package::prelude;
using parthenon::TaskListStatus;

// TODO(JMM): figure out how to add IterationDriver to the prelude
class PoissonDriver : public IterationDriver {
 public:
  enum class ResidualNorm { Infinity, L2 };
  PoissonDriver(ParameterInput *pin, Mesh *pm) : IterationDriver(pin, pm) {
    target_residual = pinput->GetReal("parthenon/iterations", "residual");
    ncheck = pinput->GetOrAddInteger("parthenon/iterations", "ncheck", 1);
    std::string normstring =
        pinput->GetOrAddString("parthenon/iterations", "residual_norm", "L2");
    if (normstring.compare("infinity") == 0) {
      norm = ResidualNorm::Infinity;
    } else if (normstring.compare("L2") == 0) {
      norm = ResidualNorm::L2;
    } else {
      PARTHENON_THROW("residual norm must be inifnity or L2");
    }
  }
  TaskListStatus Step() { return ConstructAndExecuteBlockTasks<>(this); }
  bool KeepGoing() {
    // only check residual every ncheck iterations
    if ((ncycle % ncheck != 0) || (ncycle == 0)) return true;
    residual = 0;
    MeshBlock *pmb = pmesh->pblock;
    while (pmb != nullptr) {
      auto &rc = pmb->real_containers.Get();
      if (norm == ResidualNorm::Infinity) {
        Real block_residual = GetInfResidual(rc);
        residual = block_residual > residual ? block_residual : residual;
      } else {
        residual += GetL2Residual(rc);
      }
      pmb = pmb->next;
    }
#ifdef MPI_PARALLEL
    if (norm == ResidualNorm::Infinity) {
      MPI_Allreduce(MPI_IN_PLACE, &residual, 1, MPI_PARTHENON_REAL, MPI_MAX,
                    MPI_COMM_WORLD);
    } else {
      MPI_Allreduce(MPI_IN_PLACE, &residual, 1, MPI_PARTHENON_REAL, MPI_SUM,
                    MPI_COMM_WORLD);
    }
#endif
    if (norm == ResidualNorm::L2) {
      auto shape = pmesh->mesh_size;
      Real vol = ((shape.x1max - shape.x1min) * (shape.x2max - shape.x2min) *
                  (shape.x3max - shape.x3min));
      residual = std::sqrt(residual) / vol;
    }
    return residual > target_residual;
  }
  TaskList MakeTaskList(MeshBlock *pmb);
  void OutputCycleDiagnostics();
  int ncheck = 1;
  Real residual = 0;
  Real target_residual;
  ResidualNorm norm;
};

} // namespace poisson

#endif // EXAMPLE_POISSON_POISSON_DRIVER_HPP_
