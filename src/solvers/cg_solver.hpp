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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "interface/mesh_data.hpp"
#include "interface/meshblock_data.hpp"
#include "interface/state_descriptor.hpp"
#include "kokkos_abstraction.hpp"
#include "solvers/mg_solver.hpp"
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
template <class u, class rhs, class equations>
class CGSolver {
 public:
  PARTHENON_INTERNALSOLVERVARIABLE(u, x);
  PARTHENON_INTERNALSOLVERVARIABLE(u, r);
  PARTHENON_INTERNALSOLVERVARIABLE(u, v);
  PARTHENON_INTERNALSOLVERVARIABLE(u, p);

  using internal_types_tl = TypeList<x, r, v, p>;
  using preconditioner_t = MGSolver<u, rhs, equations>;
  using all_internal_types_tl =
      concatenate_type_lists_t<internal_types_tl,
                               typename preconditioner_t::internal_types_tl>;

  std::vector<std::string> GetInternalVariableNames() const {
    std::vector<std::string> names;
    if (params_.precondition) {
      all_internal_types_tl::IterateTypes(
          [&names](auto t) { names.push_back(decltype(t)::name()); });
    } else {
      internal_types_tl::IterateTypes(
          [&names](auto t) { names.push_back(decltype(t)::name()); });
    }
    return names;
  }

  CGSolver(StateDescriptor *pkg, CGParams params_in,
                 equations eq_in = equations(), std::vector<int> shape = {},
                 const std::string &container = "base")
      : preconditioner(pkg, params_in.mg_params, eq_in, shape, container),
        params_(params_in), iter_counter(0), eqs_(eq_in), container_(container) {
    using namespace refinement_ops;
    auto m_no_ghost =
        Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy}, shape);
    pkg->AddField(x::name(), m_no_ghost);
    pkg->AddField(r::name(), m_no_ghost);
    pkg->AddField(v::name(), m_no_ghost);
    pkg->AddField(p::name(), m_no_ghost);
  }

  template <class TL_t>
  TaskID AddSetupTasks(TL_t &tl, TaskID dependence, int partition, Mesh *pmesh) {
    return preconditioner.AddSetupTasks(tl, dependence, partition, pmesh);
  }

  TaskID AddTasks(TaskList &tl, TaskID dependence, Mesh *pmesh, const int partition) {
    using namespace utils;
    TaskID none;
    auto &md = pmesh->mesh_data.GetOrAdd(container_, partition);
    std::string label = container_ + "cg_comm_" + std::to_string(partition);
    auto &md_comm =
        pmesh->mesh_data.AddShallow(label, md, std::vector<std::string>{u::name()});
    iter_counter = 0;
    bool multilevel = pmesh->multilevel;

    // Initialization: u <- 0, r <- rhs, p <- 0, ru <- 1
    auto zero_u = tl.AddTask(dependence, TF(SetToZero<u>), md);
    auto zero_v = tl.AddTask(dependence, TF(SetToZero<v>), md);
    auto zero_x = tl.AddTask(dependence, TF(SetToZero<x>), md);
    auto zero_p = tl.AddTask(dependence, TF(SetToZero<p>), md);
    auto copy_r = tl.AddTask(dependence, TF(CopyData<rhs, r>), md);
    auto get_rhs2 = none;
    if (params_.relative_residual)
      get_rhs2 = DotProduct<rhs, rhs>(dependence, tl, &rhs2, md);
    auto initialize = tl.AddTask(
        TaskQualifier::once_per_region | TaskQualifier::local_sync,
        zero_u | zero_v | zero_x | zero_p | copy_r | get_rhs2,
        "zero factors",
        [](CGSolver *solver) {
          solver->iter_counter = -1;
          solver->ru.val = std::numeric_limits<Real>::max();
          return TaskStatus::complete;
        },
        this);

    if (params_.print_per_step && Globals::my_rank == 0) {
      initialize = tl.AddTask(
          TaskQualifier::once_per_region, initialize, "print to screen",
          [&](CGSolver *solver, std::shared_ptr<Real> res_tol,
              bool relative_residual) {
            Real tol =
                relative_residual
                    ? *res_tol * std::sqrt(solver->rhs2.val / pmesh->GetTotalCells())
                    : *res_tol;
            printf("# [0] v-cycle\n# [1] rms-residual (tol = %e) \n# [2] rms-error\n",
                   tol);
            return TaskStatus::complete;
          },
          this, params_.residual_tolerance, params_.relative_residual);
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
      auto set_rhs = itl.AddTask(precon, TF(CopyData<r, rhs>), md);
      auto zero_u = itl.AddTask(precon, TF(SetToZero<u>), md);
      precon =
          preconditioner.AddLinearOperatorTasks(itl, set_rhs | zero_u, partition, pmesh);
    } else {
      precon = itl.AddTask(precon, TF(CopyData<r, u>), md);
    }

    // 2. beta <- r dot u / r dot u {old}
    auto get_ru = DotProduct<r, u>(precon, itl, &ru, md); 
    
    // 3. p <- u + beta p
    auto correct_p = itl.AddTask(
        get_ru, "p <- u + beta p",
        [](CGSolver *solver, std::shared_ptr<MeshData<Real>> &md) {
          Real beta = solver->iter_counter > 0 ? solver->ru.val / solver->ru_old : 0.0;
          return AddFieldsAndStore<u, p, p>(md, 1.0, beta);
        },
        this, md);
    
    // 4. v <- A p
    auto copy_u = itl.AddTask(correct_p, TF(CopyData<p, u>), md);
    auto comm =
        AddBoundaryExchangeTasks<BoundaryType::any>(copy_u, itl, md_comm, multilevel);
    auto get_v = eqs_.template Ax<u, v>(itl, comm, md);

    // 5. alpha <- r dot u / p dot v (calculate denominator) 
    auto get_pAp = DotProduct<p, v>(get_v, itl, &pAp, md);

    // 6. x <- x + alpha p 
    auto correct_x = itl.AddTask(
        get_pAp, "x <- x + alpha p",
        [](CGSolver *solver, std::shared_ptr<MeshData<Real>> &md) {
          Real alpha = solver->ru.val / solver->pAp.val;
          return AddFieldsAndStore<x, p, x>(md, 1.0, alpha);
        },
        this, md); 
    
    // 6. r <- r - alpha A p 
    auto correct_r = itl.AddTask(
        get_pAp, "r <- r - alpha A p",
        [](CGSolver *solver, std::shared_ptr<MeshData<Real>> &md) {
          Real alpha = solver->ru.val / solver->pAp.val;
          return AddFieldsAndStore<r, v, r>(md, 1.0, -alpha);
        },
        this, md);

    // 7. Check and print out residual
    auto get_res = DotProduct<r, r>(correct_r, itl, &residual, md);

    auto print = itl.AddTask(
        TaskQualifier::once_per_region, get_res,
        [&](CGSolver *solver, Mesh *pmesh) {
          Real rms_res = std::sqrt(solver->residual.val / pmesh->GetTotalCells());
          if (Globals::my_rank == 0 && solver->params_.print_per_step)
            printf("%i %e\n", solver->iter_counter, rms_res);
          return TaskStatus::complete;
        },
        this, pmesh);
    
    auto check = itl.AddTask(
        TaskQualifier::completion, get_res | correct_x, "completion",
        [](CGSolver *solver, Mesh *pmesh, int max_iter,
                    std::shared_ptr<Real> res_tol, bool relative_residual) {
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

    return tl.AddTask(solver_id, TF(CopyData<x, u>), md);
  }

  Real GetSquaredResidualSum() const { return residual.val; }
  int GetCurrentIterations() const { return iter_counter; }

  Real GetFinalResidual() const { return final_residual; }
  int GetFinalIterations() const { return final_iteration; }

  CGParams &GetParams() { return params_; }

 protected:
  preconditioner_t preconditioner;
  CGParams params_;
  int iter_counter;
  AllReduce<Real> ru, pAp, residual, rhs2;
  Real ru_old;
  equations eqs_;
  Real final_residual;
  int final_iteration;
  std::string container_;
};

} // namespace solvers
} // namespace parthenon

#endif // SOLVERS_CG_SOLVER_HPP_
