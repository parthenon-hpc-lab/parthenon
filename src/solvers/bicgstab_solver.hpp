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
#ifndef SOLVERS_BICGSTAB_SOLVER_HPP_
#define SOLVERS_BICGSTAB_SOLVER_HPP_

#include <cstdio>
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

enum class Preconditioner { None, Diagonal, Multigrid };
struct BiCGSTABParams {
  MGParams mg_params;
  int max_iters = 1000;
  std::shared_ptr<Real> residual_tolerance = std::make_shared<Real>(1.e-12);
  Preconditioner precondition_type = Preconditioner::Multigrid;
  bool print_per_step = false;
  bool relative_residual = false;
  BiCGSTABParams() = default;
  BiCGSTABParams(ParameterInput *pin, const std::string &input_block) {
    max_iters = pin->GetOrAddInteger(input_block, "max_iterations", max_iters);
    *residual_tolerance =
        pin->GetOrAddReal(input_block, "residual_tolerance", *residual_tolerance);
    bool precondition = pin->GetOrAddBoolean(input_block, "precondition", true);
    std::string precondition_str =
        pin->GetOrAddString(input_block, "preconditioner", "Multigrid");
    if (precondition && precondition_str == "Multigrid") {
      precondition_type = Preconditioner::Multigrid;
    } else if (precondition && precondition_str == "Diagonal") {
      precondition_type = Preconditioner::Diagonal;
    } else {
      precondition_type = Preconditioner::None;
    }
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
class BiCGSTABSolver : public SolverBase {
 public:
  PARTHENON_INTERNALSOLVERVARIABLE(u, rhat0);
  PARTHENON_INTERNALSOLVERVARIABLE(u, v);
  PARTHENON_INTERNALSOLVERVARIABLE(u, h);
  PARTHENON_INTERNALSOLVERVARIABLE(u, s);
  PARTHENON_INTERNALSOLVERVARIABLE(u, t);
  PARTHENON_INTERNALSOLVERVARIABLE(u, r);
  PARTHENON_INTERNALSOLVERVARIABLE(u, p);
  PARTHENON_INTERNALSOLVERVARIABLE(u, x);
  PARTHENON_INTERNALSOLVERVARIABLE(u, diag);

  using internal_types_tl = TypeList<rhat0, v, h, s, t, r, p, x>;
  using preconditioner_t = MGSolver<u, rhs, equations>;
  using all_internal_types_tl =
      concatenate_type_lists_t<internal_types_tl,
                               typename preconditioner_t::internal_types_tl>;

  std::vector<std::string> GetInternalVariableNames() const {
    std::vector<std::string> names;
    if (params_.precondition_type == Preconditioner::Multigrid) {
      all_internal_types_tl::IterateTypes(
          [&names](auto t) { names.push_back(decltype(t)::name()); });
    } else {
      internal_types_tl::IterateTypes(
          [&names](auto t) { names.push_back(decltype(t)::name()); });
    }
    return names;
  }

  BiCGSTABSolver(StateDescriptor *pkg, BiCGSTABParams params_in,
                 equations eq_in = equations(), std::vector<int> shape = {},
                 const std::string &container = "base")
      : preconditioner(pkg, params_in.mg_params, eq_in, shape, container),
        params_(params_in), iter_counter(0), eqs_(eq_in), container_(container) {
    using namespace refinement_ops;
    auto m_no_ghost =
        Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy}, shape);
    pkg->AddField(rhat0::name(), m_no_ghost);
    pkg->AddField(v::name(), m_no_ghost);
    pkg->AddField(h::name(), m_no_ghost);
    pkg->AddField(s::name(), m_no_ghost);
    pkg->AddField(t::name(), m_no_ghost);
    pkg->AddField(r::name(), m_no_ghost);
    pkg->AddField(p::name(), m_no_ghost);
    pkg->AddField(x::name(), m_no_ghost);
    pkg->AddField(diag::name(), m_no_ghost);
  }

  TaskID AddSetupTasks(TaskList &tl, TaskID dependence, int partition, Mesh *pmesh) {
    if (params_.precondition_type == Preconditioner::Multigrid) {
      return preconditioner.AddSetupTasks(tl, dependence, partition, pmesh);
    } else if (params_.precondition_type == Preconditioner::Diagonal) {
      auto partitions = pmesh->GetDefaultBlockPartitions();
      auto &md = pmesh->mesh_data.Add(container_, partitions[partition]);
      return tl.AddTask(dependence, &equations::template SetDiagonal<diag>, &eqs_, md);
    } else {
      return dependence;
    }
  }

  TaskID AddTasks(TaskList &tl, TaskID dependence, const int partition, Mesh *pmesh) {
    using namespace utils;
    TaskID none;
    auto &md = pmesh->mesh_data.GetOrAdd(container_, partition);
    std::string label = container_ + "bicg_comm_" + std::to_string(partition);
    auto &md_comm =
        pmesh->mesh_data.AddShallow(label, md, std::vector<std::string>{u::name()});
    iter_counter = 0;
    bool multilevel = pmesh->multilevel;

    // Initialization: x <- 0, r <- rhs, rhat0 <- rhs,
    // rhat0r_old <- (rhat0, r), p <- r, u <- 0
    auto zero_x = tl.AddTask(dependence, TF(SetToZero<x>), md);
    auto zero_u_init = tl.AddTask(dependence, TF(SetToZero<u>), md);
    auto copy_r = tl.AddTask(dependence, TF(CopyData<rhs, r>), md);
    auto copy_p = tl.AddTask(dependence, TF(CopyData<rhs, p>), md);
    auto copy_rhat0 = tl.AddTask(dependence, TF(CopyData<rhs, rhat0>), md);
    auto get_rhat0r_init = DotProduct<rhat0, r>(dependence, tl, &rhat0r, md);
    auto get_rhs2 = get_rhat0r_init;
    if (params_.relative_residual || params_.print_per_step)
      get_rhs2 = DotProduct<rhs, rhs>(dependence, tl, &rhs2, md);
    auto initialize = tl.AddTask(
        TaskQualifier::once_per_region | TaskQualifier::local_sync,
        zero_x | zero_u_init | copy_r | copy_p | copy_rhat0 | get_rhat0r_init | get_rhs2,
        "zero factors",
        [](BiCGSTABSolver *solver) {
          solver->iter_counter = -1;
          return TaskStatus::complete;
        },
        this);
    tl.AddTask(
        TaskQualifier::once_per_region, initialize, "print to screen",
        [&](BiCGSTABSolver *solver, std::shared_ptr<Real> res_tol, bool relative_residual,
            Mesh *pm) {
          if (Globals::my_rank == 0 && params_.print_per_step) {
            Real tol = relative_residual
                           ? *res_tol * std::sqrt(solver->rhs2.val / pm->GetTotalCells())
                           : *res_tol;
            printf("# [0] v-cycle\n# [1] rms-residual (tol = %e) \n# [2] rms-error\n",
                   tol);
            printf("0 %e\n", std::sqrt(solver->rhs2.val / pm->GetTotalCells()));
          }
          return TaskStatus::complete;
        },
        this, params_.residual_tolerance, params_.relative_residual, pmesh);

    // BEGIN ITERATIVE TASKS
    auto [itl, solver_id] = tl.AddSublist(initialize, {1, params_.max_iters});

    auto sync = itl.AddTask(TaskQualifier::local_sync, none,
                            []() { return TaskStatus::complete; });
    auto reset = itl.AddTask(
        TaskQualifier::once_per_region, sync, "update values",
        [](BiCGSTABSolver *solver) {
          solver->rhat0r_old = solver->rhat0r.val;
          solver->iter_counter++;
          return TaskStatus::complete;
        },
        this);

    // 1. u <- M p
    auto precon1 = reset;
    if (params_.precondition_type == Preconditioner::Multigrid) {
      auto set_rhs = itl.AddTask(precon1, TF(CopyData<p, rhs>), md);
      auto zero_u = itl.AddTask(precon1, TF(SetToZero<u>), md);
      precon1 =
          preconditioner.AddLinearOperatorTasks(itl, set_rhs | zero_u, partition, pmesh);
    } else if (params_.precondition_type == Preconditioner::Diagonal) {
      precon1 = itl.AddTask(precon1, TF(ADividedByB<p, diag, u>), md);
    } else {
      precon1 = itl.AddTask(precon1, TF(CopyData<p, u>), md);
    }

    // 2. v <- A u
    auto comm =
        AddBoundaryExchangeTasks<BoundaryType::any>(precon1, itl, md_comm, multilevel);
    auto get_v = eqs_.template Ax<u, v>(itl, comm, md);

    // 3. rhat0v <- (rhat0, v)
    auto get_rhat0v = DotProduct<rhat0, v>(get_v, itl, &rhat0v, md);

    // 4. h <- x + alpha u (alpha = rhat0r_old / rhat0v)
    auto correct_h = itl.AddTask(
        get_rhat0v, "h <- x + alpha u",
        [](BiCGSTABSolver *solver, std::shared_ptr<MeshData<Real>> &md) {
          Real alpha = solver->rhat0r_old / solver->rhat0v.val;
          return AddFieldsAndStore<x, u, h>(md, 1.0, alpha);
        },
        this, md);

    // 5. s <- r - alpha v (alpha = rhat0r_old / rhat0v)
    auto correct_s = itl.AddTask(
        get_rhat0v, "s <- r - alpha v",
        [](BiCGSTABSolver *solver, std::shared_ptr<MeshData<Real>> &md) {
          Real alpha = solver->rhat0r_old / solver->rhat0v.val;
          return AddFieldsAndStore<r, v, s>(md, 1.0, -alpha);
        },
        this, md);

    // Check and print out residual
    auto get_res = DotProduct<s, s>(correct_s, itl, &residual, md);

    auto print = itl.AddTask(
        TaskQualifier::once_per_region, get_res,
        [&](BiCGSTABSolver *solver, Mesh *pmesh) {
          Real rms_res = std::sqrt(solver->residual.val / pmesh->GetTotalCells());
          if (Globals::my_rank == 0 && solver->params_.print_per_step)
            printf("%i %e\n", solver->iter_counter * 2 + 1, rms_res);
          return TaskStatus::complete;
        },
        this, pmesh);

    // 6. u <- M s
    auto precon2 = correct_s;
    if (params_.precondition_type == Preconditioner::Multigrid) {
      auto set_rhs = itl.AddTask(precon2, TF(CopyData<s, rhs>), md);
      auto zero_u = itl.AddTask(precon2, TF(SetToZero<u>), md);
      precon2 =
          preconditioner.AddLinearOperatorTasks(itl, set_rhs | zero_u, partition, pmesh);
    } else if (params_.precondition_type == Preconditioner::Diagonal) {
      precon2 = itl.AddTask(precon2, TF(ADividedByB<s, diag, u>), md);
    } else {
      precon2 = itl.AddTask(precon2, TF(CopyData<s, u>), md);
    }

    // 7. t <- A u
    auto pre_t_comm =
        AddBoundaryExchangeTasks<BoundaryType::any>(precon2, itl, md_comm, multilevel);
    auto get_t = eqs_.template Ax<u, t>(itl, pre_t_comm, md);

    // 8. omega <- (t,s) / (t,t)
    auto get_ts = DotProduct<t, s>(get_t, itl, &ts, md);
    auto get_tt = DotProduct<t, t>(get_t, itl, &tt, md);

    // 9. x <- h + omega u
    auto correct_x = itl.AddTask(
        get_tt | get_ts, "x <- h + omega u",
        [](BiCGSTABSolver *solver, std::shared_ptr<MeshData<Real>> &md) {
          Real omega = solver->ts.val / solver->tt.val;
          return AddFieldsAndStore<h, u, x>(md, 1.0, omega);
        },
        this, md);

    // 10. r <- s - omega t
    auto correct_r = itl.AddTask(
        get_tt | get_ts, "r <- s - omega t",
        [](BiCGSTABSolver *solver, std::shared_ptr<MeshData<Real>> &md) {
          Real omega = solver->ts.val / solver->tt.val;
          return AddFieldsAndStore<s, t, r>(md, 1.0, -omega);
        },
        this, md);

    // Check and print out residual
    auto get_res2 = DotProduct<r, r>(correct_r, itl, &residual, md);

    get_res2 = itl.AddTask(
        TaskQualifier::once_per_region, get_res2,
        [&](BiCGSTABSolver *solver, Mesh *pmesh) {
          Real rms_err = std::sqrt(solver->residual.val / pmesh->GetTotalCells());
          if (Globals::my_rank == 0 && solver->params_.print_per_step)
            printf("%i %e\n", solver->iter_counter * 2 + 2, rms_err);
          return TaskStatus::complete;
        },
        this, pmesh);

    // 11. rhat0r <- (rhat0, r)
    auto get_rhat0r = DotProduct<rhat0, r>(correct_r, itl, &rhat0r, md);

    // 12. beta <- rhat0r / rhat0r_old * alpha / omega
    // 13. p <- r + beta * (p - omega * v)
    auto update_p = itl.AddTask(
        get_rhat0r | get_res2, "p <- r + beta * (p - omega * v)",
        [](BiCGSTABSolver *solver, std::shared_ptr<MeshData<Real>> &md) {
          Real alpha = solver->rhat0r_old / solver->rhat0v.val;
          Real omega = solver->ts.val / solver->tt.val;
          Real beta = solver->rhat0r.val / solver->rhat0r_old * alpha / omega;
          AddFieldsAndStore<p, v, p>(md, 1.0, -omega);
          return AddFieldsAndStore<r, p, p>(md, 1.0, beta);
          return TaskStatus::complete;
        },
        this, md);

    // 14. rhat0r_old <- rhat0r, zero all reductions
    auto check = itl.AddTask(
        TaskQualifier::completion, update_p | correct_x, "rhat0r_old <- rhat0r",
        [partition](BiCGSTABSolver *solver, Mesh *pmesh, int max_iter,
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

  BiCGSTABParams &GetParams() { return params_; }

 protected:
  preconditioner_t preconditioner;
  BiCGSTABParams params_;
  int iter_counter;
  AllReduce<Real> rtr, pAp, rhat0v, rhat0r, ts, tt, residual, rhs2;
  Real rhat0r_old;
  equations eqs_;
  std::string container_;
};

} // namespace solvers

} // namespace parthenon

#endif // SOLVERS_BICGSTAB_SOLVER_HPP_
