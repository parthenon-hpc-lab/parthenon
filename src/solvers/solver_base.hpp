//========================================================================================
// (C) (or copyright) 2023-2024. Triad National Security, LLC. All rights reserved.
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
#ifndef SOLVERS_SOLVER_BASE_HPP_
#define SOLVERS_SOLVER_BASE_HPP_

#include <algorithm>
#include <cstdio>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "interface/mesh_data.hpp"
#include "interface/meshblock_data.hpp"
#include "tasks/tasks.hpp"

namespace parthenon {

namespace solvers {

class SolverBase {
 public:
  virtual ~SolverBase() {}

  virtual TaskID AddSetupTasks(TaskList &tl, TaskID dependence, int partition,
                               Mesh *pmesh) = 0;
  virtual TaskID AddTasks(TaskList &tl, TaskID dependence, int partition,
                          Mesh *pmesh) = 0;

  Real GetFinalResidual() const { return final_residual; }
  int GetFinalIterations() const { return final_iteration; }

 protected:
  Real final_residual;
  int final_iteration;
};

} // namespace solvers

} // namespace parthenon

#endif // SOLVERS_SOLVER_BASE_HPP_
