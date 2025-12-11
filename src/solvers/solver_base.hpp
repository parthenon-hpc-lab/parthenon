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
namespace parthenon {

namespace solvers {

// Used for checking if a given equations class has a SetBoundary function
template <typename T, typename = void>
struct has_SetBoundary : std::false_type {};

template <typename T>
struct has_SetBoundary<
    T, std::void_t<decltype(std::declval<T>().SetBoundary(
           std::declval<std::shared_ptr<MeshData<Real>> &>(), std::declval<bool>()))>>
    : std::true_type {};

// Solver base class
class SolverBase {
 public:
  SolverBase(const std::string &container_base, const std::string &container_u,
             const std::string &container_rhs)
      : container_base(container_base), container_u(container_u),
        container_rhs(container_rhs) {}

  virtual ~SolverBase() {}

  virtual TaskID AddSetupTasks(TaskList &tl, TaskID dependence, int partition,
                               Mesh *pmesh) = 0;
  virtual TaskID AddTasks(TaskList &tl, TaskID dependence, int partition,
                          Mesh *pmesh) = 0;

  // Provide access to the underlying matrix operator for convenience
  virtual TaskID Ax(TaskList &tl, TaskID dependence,
                    std::shared_ptr<MeshData<Real>> &md_mat,
                    std::shared_ptr<MeshData<Real>> &md_in,
                    std::shared_ptr<MeshData<Real>> &md_out) = 0;

  virtual void SetConstantProlongation(bool const_pro) {}

  Real GetFinalResidual() const { return final_residual; }
  int GetFinalIterations() const { return final_iteration; }

  void SetRHSContainerLabel(const std::string &rhs) { container_rhs = rhs; }
  const std::string &GetBaseContainerLabel() const { return container_base; }
  const std::string &GetRHSContainerLabel() const { return container_rhs; }
  const std::string &GetSolutionContainerLabel() const { return container_u; }

  const std::vector<std::string> &GetFieldLabels() const { return sol_fields; }

  bool initial_guess_is_zero{false};

  static inline TimingAccumulatorDictionary solver_timings;

 protected:
  // Labels of all fields included in the vector
  std::vector<std::string> sol_fields;
  // Name of user defined container that should contain information required to
  // calculate the matrix part of the matrix vector product
  std::string container_base;
  // User defined container in which the solution will reside, only needs to contain
  // sol_fields
  // TODO(LFR): Also allow for an initial guess to come in here
  std::string container_u;
  // User defined container containing the rhs vector, only needs to contain sol_fields
  std::string container_rhs;

  Real final_residual;
  int final_iteration;
};

} // namespace solvers

} // namespace parthenon

#endif // SOLVERS_SOLVER_BASE_HPP_
