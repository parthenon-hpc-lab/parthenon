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
#ifndef SOLVERS_CG_SOLVER_HPP_
#define SOLVERS_CG_SOLVER_HPP_

#include <cstdio>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "interface/mesh_data.hpp"
#include "interface/meshblock_data.hpp"
#include "interface/state_descriptor.hpp"
#include "kokkos_abstraction.hpp"
#include "solvers/mg_solver.hpp"
#include "solvers/solver_base.hpp"
#include "solvers/solver_utils.hpp"
#include "tasks/tasks.hpp"
#include "utils/type_list.hpp"

namespace parthenon {

namespace solvers {

struct CGParams {
  MGParams mg_params;
  int max_iters = 1000;
  std::shared_ptr<Real> residual_tolerance = std::make_shared<Real>(1.e-12);
  bool precondition = true;
  bool print_per_step = false;
  bool relative_residual = false;
  CGParams() = default;
  CGParams(ParameterInput *pin, const std::string &input_block) {
    max_iters = pin->GetOrAddInteger(input_block, "max_iterations", max_iters);
    *residual_tolerance =
        pin->GetOrAddReal(input_block, "residual_tolerance", *residual_tolerance);
    precondition = pin->GetOrAddBoolean(input_block, "precondition", precondition);
    print_per_step = pin->GetOrAddBoolean(input_block, "print_per_step", print_per_step);
    mg_params = MGParams(pin, input_block);
    relative_residual =
        pin->GetOrAddBoolean(input_block, "relative_residual", relative_residual);
  }
};

// The equations class must include a template method
//
//   template <class x_t, class y_t, class TL_t>
//   TaskID Ax(TL_t &tl, TaskID depends_on, std::shared_ptr<MeshData<Real>> &md)
//
// that takes a field associated with x_t and applies
// the matrix A to it and stores the result in y_t.
template <class equations, class preconditioner_t = MGSolver<equations>>
class CGSolver : public SolverBase {
  using FieldTL = typename equations::IndependentVars;

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
  // Internal containers for solver which create deep copies of sol_fields
  std::string container_x, container_r, container_v, container_p;

  static inline std::size_t id{0};

 public:
  CGSolver(const std::string &container_base, const std::string &container_u,
           const std::string &container_rhs, ParameterInput *pin,
           const std::string &input_block, const equations &eq_in = equations())
      : preconditioner(container_base, container_u, container_rhs, pin, input_block,
                       eq_in),
        container_base(container_base), container_u(container_u),
        container_rhs(container_rhs), params_(pin, input_block), iter_counter(0),
        eqs_(eq_in) {
    FieldTL::IterateTypes(
        [this](auto t) { this->sol_fields.push_back(decltype(t)::name()); });
    std::string solver_id = "cg" + std::to_string(id++);
    container_x = solver_id + "_x";
    container_r = solver_id + "_r";
    container_v = solver_id + "_v";
    container_p = solver_id + "_p";
  }

  TaskID AddSetupTasks(TaskList &tl, TaskID dependence, int partition, Mesh *pmesh) {
    return preconditioner.AddSetupTasks(tl, dependence, partition, pmesh);
  }

  TaskID AddTasks(TaskList &tl, TaskID dependence, const int partition, Mesh *pmesh) {
    using namespace utils;
    TaskID none;
    auto partitions = pmesh->GetDefaultBlockPartitions();
    // Should contain all fields necessary for applying the matrix to a give state vector,
    // e.g. diffusion coefficients and diagonal, these will not be modified by the solvers
    auto &md_base = pmesh->mesh_data.Add(container_base, partitions[partition]);
    // Container in which the solution is stored and with which the downstream user can
    // interact. This container only requires the fields in sol_fields
    auto &md_u = pmesh->mesh_data.Add(container_u, partitions[partition], sol_fields);
    // Container of the rhs, only requires fields in sol_fields
    auto &md_rhs = pmesh->mesh_data.Add(container_rhs, partitions[partition], sol_fields);
    // Internal solver containers
    auto &md_x = pmesh->mesh_data.Add(container_x, md_u, sol_fields);
    auto &md_r = pmesh->mesh_data.Add(container_r, md_u, sol_fields);
    // TODO(LFR): The v container can probably be removed and the u container used in its
    // stead
    auto &md_v = pmesh->mesh_data.Add(container_v, md_u, sol_fields);
    auto &md_p = pmesh->mesh_data.Add(container_p, md_u, sol_fields);

    iter_counter = 0;
    bool multilevel = pmesh->multilevel;

    // Initialization: u <- 0, r <- rhs, p <- 0, ru <- 1
    auto zero_u = tl.AddTask(dependence, TF(SetToZero<FieldTL>), md_u);
    auto zero_v = tl.AddTask(dependence, TF(SetToZero<FieldTL>), md_v);
    auto zero_x = tl.AddTask(dependence, TF(SetToZero<FieldTL>), md_x);
    auto zero_p = tl.AddTask(dependence, TF(SetToZero<FieldTL>), md_p);
    auto copy_r = tl.AddTask(dependence, TF(CopyData<FieldTL>), md_rhs, md_r);
    auto get_rhs2 = none;
    if (params_.relative_residual || params_.print_per_step)
      get_rhs2 = DotProduct<FieldTL>(dependence, tl, &rhs2, md_rhs, md_rhs);
    auto initialize = tl.AddTask(
        TaskQualifier::once_per_region | TaskQualifier::local_sync,
        zero_u | zero_v | zero_x | zero_p | copy_r | get_rhs2, "zero factors",
        [](CGSolver *solver) {
          solver->iter_counter = -1;
          solver->ru.val = std::numeric_limits<Real>::max();
          return TaskStatus::complete;
        },
        this);

    if (params_.print_per_step && Globals::my_rank == 0) {
      initialize = tl.AddTask(
          TaskQualifier::once_per_region, initialize, "print to screen",
          [&](CGSolver *solver, std::shared_ptr<Real> res_tol, bool relative_residual,
              Mesh *pm) {
            Real tol = relative_residual
                           ? *res_tol * std::sqrt(solver->rhs2.val / pm->GetTotalCells())
                           : *res_tol;
            printf("# [0] v-cycle\n# [1] rms-residual (tol = %e) \n# [2] rms-error\n",
                   tol);
            printf("0 %e\n", std::sqrt(solver->rhs2.val / pm->GetTotalCells()));
            return TaskStatus::complete;
          },
          this, params_.residual_tolerance, params_.relative_residual, pmesh);
    }

    // BEGIN ITERATIVE TASKS
    auto [itl, solver_id] = tl.AddSublist(initialize, {1, params_.max_iters});

    auto sync = itl.AddTask(TaskQualifier::local_sync, none,
                            []() { return TaskStatus::complete; });
    auto reset = itl.AddTask(
        TaskQualifier::once_per_region, sync, "update values",
        [](CGSolver *solver) {
          solver->ru_old = solver->ru.val;
          solver->iter_counter++;
          return TaskStatus::complete;
        },
        this);

    // 1. u <- M r
    auto precon = reset;
    if (params_.precondition) {
      auto set_rhs = itl.AddTask(precon, TF(CopyData<FieldTL>), md_r, md_rhs);
      auto zero_u = itl.AddTask(precon, TF(SetToZero<FieldTL>), md_u);
      precon =
          preconditioner.AddLinearOperatorTasks(itl, set_rhs | zero_u, partition, pmesh);
    } else {
      precon = itl.AddTask(precon, TF(CopyData<FieldTL>), md_r, md_u);
    }

    // 2. beta <- r dot u / r dot u {old}
    auto get_ru = DotProduct<FieldTL>(precon, itl, &ru, md_r, md_u);

    // 3. p <- u + beta p
    auto correct_p = itl.AddTask(
        get_ru, "p <- u + beta p",
        [](CGSolver *solver, std::shared_ptr<MeshData<Real>> &md_u,
           std::shared_ptr<MeshData<Real>> &md_p) {
          Real beta = solver->iter_counter > 0 ? solver->ru.val / solver->ru_old : 0.0;
          return AddFieldsAndStore<FieldTL>(md_u, md_p, md_p, 1.0, beta);
        },
        this, md_u, md_p);

    // 4. v <- A p
    auto comm =
        AddBoundaryExchangeTasks<BoundaryType::any>(correct_p, itl, md_p, multilevel);
    auto get_v = eqs_.Ax(itl, comm, md_base, md_p, md_v);

    // 5. alpha <- r dot u / p dot v (calculate denominator)
    auto get_pAp = DotProduct<FieldTL>(get_v, itl, &pAp, md_p, md_v);

    // 6. x <- x + alpha p
    auto correct_x = itl.AddTask(
        get_pAp, "x <- x + alpha p",
        [](CGSolver *solver, std::shared_ptr<MeshData<Real>> &md_x,
           std::shared_ptr<MeshData<Real>> &md_p) {
          Real alpha = solver->ru.val / solver->pAp.val;
          return AddFieldsAndStore<FieldTL>(md_x, md_p, md_x, 1.0, alpha);
        },
        this, md_x, md_p);

    // 6. r <- r - alpha A p
    auto correct_r = itl.AddTask(
        get_pAp, "r <- r - alpha A p",
        [](CGSolver *solver, std::shared_ptr<MeshData<Real>> &md_r,
           std::shared_ptr<MeshData<Real>> &md_v) {
          Real alpha = solver->ru.val / solver->pAp.val;
          return AddFieldsAndStore<FieldTL>(md_r, md_v, md_r, 1.0, -alpha);
        },
        this, md_r, md_v);

    // 7. Check and print out residual
    auto get_res = DotProduct<FieldTL>(correct_r, itl, &residual, md_r, md_r);

    auto print = itl.AddTask(
        TaskQualifier::once_per_region, get_res,
        [&](CGSolver *solver, Mesh *pmesh) {
          Real rms_res = std::sqrt(solver->residual.val / pmesh->GetTotalCells());
          if (Globals::my_rank == 0 && solver->params_.print_per_step)
            printf("\t%i %e\n", solver->iter_counter, rms_res);
          return TaskStatus::complete;
        },
        this, pmesh);

    auto check = itl.AddTask(
        TaskQualifier::completion, get_res | correct_x, "completion",
        [](CGSolver *solver, Mesh *pmesh, int max_iter, std::shared_ptr<Real> res_tol,
           bool relative_residual) {
          Real rms_res = std::sqrt(solver->residual.val / pmesh->GetTotalCells());
          solver->final_residual = rms_res;
          solver->final_iteration = solver->iter_counter;
          Real tol = relative_residual
                         ? *res_tol * std::sqrt(solver->rhs2.val / pmesh->GetTotalCells())
                         : *res_tol;
          if (rms_res < tol || solver->iter_counter >= max_iter) {
            solver->final_residual = rms_res;
            solver->final_iteration = solver->iter_counter;
            return TaskStatus::complete;
          }
          return TaskStatus::iterate;
        },
        this, pmesh, params_.max_iters, params_.residual_tolerance,
        params_.relative_residual);

    return tl.AddTask(solver_id, TF(CopyData<FieldTL>), md_x, md_u);
  }

  Real GetSquaredResidualSum() const { return residual.val; }
  int GetCurrentIterations() const { return iter_counter; }

  CGParams &GetParams() { return params_; }

 protected:
  preconditioner_t preconditioner;
  CGParams params_;
  int iter_counter;
  AllReduce<Real> ru, pAp, residual, rhs2;
  Real ru_old;
  equations eqs_;
};

} // namespace solvers
} // namespace parthenon

#endif // SOLVERS_CG_SOLVER_HPP_
