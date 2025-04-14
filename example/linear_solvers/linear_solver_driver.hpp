//========================================================================================
// (C) (or copyright) 2021-2024. Triad National Security, LLC. All rights reserved.
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

#ifndef EXAMPLE_LINEAR_SOLVERS_LINEAR_SOLVER_DRIVER_HPP_
#define EXAMPLE_LINEAR_SOLVERS_LINEAR_SOLVER_DRIVER_HPP_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <kokkos_abstraction.hpp>
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

namespace linear_solver_example {
using namespace parthenon::driver::prelude;

class LinearSolverDriver : public Driver {
 public:
  LinearSolverDriver(ParameterInput *pin, ApplicationInput *app_in, Mesh *pm)
      : Driver(pin, app_in, pm) {
    InitializeOutputs();
  }
  // This next function essentially defines the driver.
  TaskCollection MakeTaskCollection(BlockList_t &blocks);

  DriverStatus Execute() override;

  std::map<std::string, Real> final_rms_error, final_rms_residual;

  // Necessary reductions for checking error from exact solution
  AllReduce<Real> err;
 private:

  using initialize_vector_func_t = std::function<parthenon::TaskStatus(parthenon::ParameterInput*, std::shared_ptr<parthenon::MeshData<parthenon::Real>>)>;
  template <class solver_TypeList>
  void AddSolverTaskRegion(parthenon::TaskCollection &tc,
                           std::string pacakge_label,
                           initialize_vector_func_t Initialize,
                           initialize_vector_func_t SetRHS,
                           initialize_vector_func_t SetExact);
};

void ProblemGenerator(Mesh *pm, parthenon::ParameterInput *pin, MeshData<Real> *md);
parthenon::Packages_t ProcessPackages(std::unique_ptr<parthenon::ParameterInput> &pin);

} // namespace linear_solver_example

#endif // EXAMPLE_LINEAR_SOLVERS_LINEAR_SOLVER_DRIVER_HPP_
