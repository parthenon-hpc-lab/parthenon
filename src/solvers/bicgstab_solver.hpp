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

#include "bvals/comms/bvals_in_one.hpp"
#include "interface/mesh_data.hpp"
#include "interface/meshblock_data.hpp"
#include "interface/state_descriptor.hpp"
#include "kokkos_abstraction.hpp"
#include "solvers/mg_solver.hpp"
#include "solvers/solver_base.hpp"
#include "solvers/solver_utils.hpp"
#include "tasks/tasks.hpp"
#include "utils/reductions.hpp"
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

struct BiCGSTABSolverCounter {
  static inline std::size_t id{0};
};

// The equations_t class must include a template method
//
//   template <class x_t, class y_t, class TL_t>
//   TaskID Ax(TL_t &tl, TaskID depends_on, std::shared_ptr<MeshData<Real>> &md)
//
// that takes a field associated with x_t and applies
// the matrix A to it and stores the result in y_t.
template <class equations_t, class preconditioner_t = MGSolver<equations_t>>
class BiCGSTABSolver : public SolverBase, BiCGSTABSolverCounter {
  using FieldTL = typename equations_t::IndependentVars;

  // Internal containers for solver which create deep copies of sol_fields
  std::string container_rhat0, container_v, container_h, container_s;
  std::string container_t, container_r, container_p, container_x, container_diag;
  BValOnMDFunc_t BCFunc;

 public:
  BiCGSTABSolver(const std::string &container_base, const std::string &container_u,
                 const std::string &container_rhs, ParameterInput *pin,
                 const std::string &input_block, equations_t eq_in = equations_t())
      : preconditioner(container_base, container_u, container_rhs, pin, input_block,
                       eq_in),
        SolverBase(container_base, container_u, container_rhs), params_(pin, input_block),
        iter_counter(0), eqs_(eq_in) {
    FieldTL::IterateTypes(
        [this](auto t) { this->sol_fields.push_back(decltype(t)::name()); });
    std::string solver_id = "bicgstab" + std::to_string(id++);
    container_rhat0 = solver_id + "_rhat0";
    container_v = solver_id + "_v";
    container_h = solver_id + "_h";
    container_s = solver_id + "_s";
    container_t = solver_id + "_t";
    container_r = solver_id + "_r";
    container_p = solver_id + "_p";
    container_x = solver_id + "_x";
    container_diag = solver_id + "_diag";
    preconditioner.initial_guess_is_zero = true;
    if constexpr (has_SetBoundary<equations_t>::value) {
      BCFunc = equations_t::SetBoundary;
    } else {
      BCFunc = ApplyBoundaryConditionsOnCoarseOrFineMD;
    }
  }

  TaskID Ax(TaskList &tl, TaskID dependence, std::shared_ptr<MeshData<Real>> &md_mat,
            std::shared_ptr<MeshData<Real>> &md_in,
            std::shared_ptr<MeshData<Real>> &md_out) {
    return eqs_.Ax(tl, dependence, md_mat, md_in, md_out);
  }

  TaskID AddSetupTasks(TaskList &tl, TaskID dependence, int partition, Mesh *pmesh) {
    if (params_.precondition_type == Preconditioner::Multigrid) {
      return preconditioner.AddSetupTasks(tl, dependence, partition, pmesh);
    } else if (params_.precondition_type == Preconditioner::Diagonal) {
      auto partitions = pmesh->GetDefaultBlockPartitions();
      auto &md = pmesh->mesh_data.Add(container_base, partitions[partition]);
      auto &md_diag = pmesh->mesh_data.Add(container_diag, md, sol_fields);
      return tl.AddTask(dependence, &equations_t::SetDiagonal, &eqs_, md, md_diag);
    } else {
      return dependence;
    }
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
    auto &md_rhat0 = pmesh->mesh_data.Add(container_rhat0, md_u, sol_fields);
    auto &md_v = pmesh->mesh_data.Add(container_v, md_u, sol_fields);
    auto &md_h = pmesh->mesh_data.Add(container_h, md_u, sol_fields);
    auto &md_s = pmesh->mesh_data.Add(container_s, md_u, sol_fields);
    auto &md_t = pmesh->mesh_data.Add(container_t, md_u, sol_fields);
    auto &md_r = pmesh->mesh_data.Add(container_r, md_u, sol_fields);
    auto &md_p = pmesh->mesh_data.Add(container_p, md_u, sol_fields);
    auto &md_x = pmesh->mesh_data.Add(container_x, md_u, sol_fields);
    auto &md_diag = pmesh->mesh_data.Add(container_diag, md_u, sol_fields);

    iter_counter = 0;
    bool multilevel = pmesh->multilevel;

    // Initialization: x <- 0, r <- rhs, rhat0 <- rhs,
    // rhat0r_old <- (rhat0, r), p <- r, u <- 0
    auto zero_x = tl.AddTask(dependence, TF(SetToZero<FieldTL>), md_x);
    auto zero_u_init = tl.AddTask(dependence, TF(SetToZero<FieldTL>), md_u);
    auto copy_r = tl.AddTask(dependence, TF(CopyData<FieldTL>), md_rhs, md_r);
    auto copy_p = tl.AddTask(dependence, TF(CopyData<FieldTL>), md_rhs, md_p);
    auto copy_rhat0 = tl.AddTask(dependence, TF(CopyData<FieldTL>), md_rhs, md_rhat0);
    auto get_rhs2_rhat0r_init = DoubleDotProduct<FieldTL>(dependence, tl, &res_rhat0r, md_r, md_rhat0);
    auto initialize = tl.AddTask(
        TaskQualifier::once_per_region | TaskQualifier::local_sync,
        zero_x | zero_u_init | copy_r | copy_p | copy_rhat0 | get_rhs2_rhat0r_init,
        "zero factors",
        [](BiCGSTABSolver *solver) {
          solver->iter_counter = -1;
          solver->rhs2 = solver->res_rhat0r.val[0];
          return TaskStatus::complete;
        },
        this);
    tl.AddTask(
        TaskQualifier::once_per_region, initialize, "print to screen",
        [&](BiCGSTABSolver *solver, std::shared_ptr<Real> res_tol, bool relative_residual,
            Mesh *pm) {
          if (Globals::my_rank == 0 && params_.print_per_step) {
            Real tol = relative_residual
                           ? *res_tol * std::sqrt(solver->rhs2 / pm->GetTotalCells())
                           : *res_tol;
            printf("# [0] v-cycle\n# [1] rms-residual (tol = %e) \n# [2] rms-error\n",
                   tol);
            printf("0 %e\n", std::sqrt(solver->rhs2 / pm->GetTotalCells()));
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
          solver->rhat0r_old = solver->res_rhat0r.val[1];
          solver->iter_counter++;
          return TaskStatus::complete;
        },
        this);

    // 1. u <- M p
    auto precon1 = reset;
    if (params_.precondition_type == Preconditioner::Multigrid) {
      auto timer = solver_timings.GetOrAddAndRegister("BiCGSTAB: Precon setup", itl);
      timer->StartCollectingTasks();
      auto zero_u = itl.AddTask(precon1, TF(SetToZero<FieldTL>), md_u);
      timer->StopCollectingTasks();
      preconditioner.SetRHSContainerLabel(container_p);
      precon1 =
          preconditioner.AddLinearOperatorTasks(itl, zero_u, partition, pmesh);
    } else if (params_.precondition_type == Preconditioner::Diagonal) {
      precon1 = itl.AddTask(precon1, TF(ADividedByB<FieldTL>), md_p, md_diag, md_u);
    } else {
      precon1 = itl.AddTask(precon1, TF(CopyData<FieldTL>), md_p, md_u);
    }

    // 2. v <- A u
    auto timer_Auv = solver_timings.GetOrAddAndRegister("BiCGSTAB: Au -> v", itl);
    auto timer_comm = solver_timings.GetOrAddAndRegister("BiCGSTAB: Boundary", itl);
    timer_Auv->StartCollectingTasks();
    timer_comm->StartCollectingTasks();
    auto comm = AddBoundaryExchangeTasks<BoundaryType::any>(precon1, itl, md_u,
                                                            multilevel, BCFunc);
    timer_comm->StopCollectingTasks();
    auto get_v = eqs_.Ax(itl, comm, md_base, md_u, md_v);
    timer_Auv->StopCollectingTasks();
    
    // 3. rhat0v <- (rhat0, v)
    auto timer_alpha = solver_timings.GetOrAddAndRegister("BiCGSTAB: alpha update", itl);
    timer_alpha->StartCollectingTasks();
    auto get_rhat0v = DotProduct<FieldTL>(get_v, itl, &rhat0v, md_rhat0, md_v);

    // 4. h <- x + alpha u (alpha = rhat0r_old / rhat0v)
    auto correct_h = itl.AddTask(
        get_rhat0v, "h <- x + alpha u",
        [](BiCGSTABSolver *solver, std::shared_ptr<MeshData<Real>> &md_x,
           std::shared_ptr<MeshData<Real>> &md_u, std::shared_ptr<MeshData<Real>> &md_h) {
          Real alpha = solver->rhat0r_old / solver->rhat0v.val;
          return AddFieldsAndStore<FieldTL>(md_x, md_u, md_h, 1.0, alpha);
        },
        this, md_x, md_u, md_h);

    // 5. s <- r - alpha v (alpha = rhat0r_old / rhat0v)
    auto correct_s = itl.AddTask(
        get_rhat0v, "s <- r - alpha v",
        [](BiCGSTABSolver *solver, std::shared_ptr<MeshData<Real>> &md_r,
           std::shared_ptr<MeshData<Real>> &md_v, std::shared_ptr<MeshData<Real>> &md_s) {
          Real alpha = solver->rhat0r_old / solver->rhat0v.val;
          return AddFieldsAndStore<FieldTL>(md_r, md_v, md_s, 1.0, -alpha);
        },
        this, md_r, md_v, md_s);

    // Check and print out residual
    if (params_.print_per_step) {
      auto get_res = DotProduct<FieldTL>(correct_s, itl, &residual, md_s, md_s);

      auto print = itl.AddTask(
          TaskQualifier::once_per_region, get_res,
          [&](BiCGSTABSolver *solver, Mesh *pmesh) {
            Real rms_res = std::sqrt(solver->residual.val / pmesh->GetTotalCells());
            if (Globals::my_rank == 0 && solver->params_.print_per_step)
              printf("%i %e\n", solver->iter_counter * 2 + 1, rms_res);
            return TaskStatus::complete;
          },
          this, pmesh);
    }
    timer_alpha->StopCollectingTasks();

    // 6. u <- M s
    auto precon2 = correct_s;
    if (params_.precondition_type == Preconditioner::Multigrid) {
      auto timer = solver_timings.GetOrAddAndRegister("BiCGSTAB: Precon setup", itl);
      timer->StartCollectingTasks();
      auto zero_u = itl.AddTask(precon2, TF(SetToZero<FieldTL>), md_u);
      timer->StopCollectingTasks();
      preconditioner.SetRHSContainerLabel(container_s);
      precon2 =
          preconditioner.AddLinearOperatorTasks(itl, zero_u, partition, pmesh);
    } else if (params_.precondition_type == Preconditioner::Diagonal) {
      precon2 = itl.AddTask(precon2, TF(ADividedByB<FieldTL>), md_s, md_diag, md_u);
    } else {
      precon2 = itl.AddTask(precon2, TF(CopyData<FieldTL>), md_s, md_u);
    }

    // 7. t <- A u
    auto timer_Aut = solver_timings.GetOrAddAndRegister("BiCGSTAB: Au -> t", itl);
    timer_comm->StartCollectingTasks();
    timer_Aut->StartCollectingTasks();
    auto pre_t_comm = AddBoundaryExchangeTasks<BoundaryType::any>(precon2, itl, md_u,
                                                                  multilevel, BCFunc);
    timer_comm->StopCollectingTasks();
    auto get_t = eqs_.Ax(itl, pre_t_comm, md_base, md_u, md_t);
    timer_Aut->StopCollectingTasks();

    // 8. omega <- (t,s) / (t,t)
    auto timer_omega = solver_timings.GetOrAddAndRegister("BiCGSTAB: omega update", itl);
    timer_omega->StartCollectingTasks();
    auto get_tt_ts = DoubleDotProduct<FieldTL>(get_t, itl, &tt_ts, md_t, md_s);

    // 9. x <- h + omega u
    auto correct_x = itl.AddTask(
        get_tt_ts, "x <- h + omega u",
        [](BiCGSTABSolver *solver, std::shared_ptr<MeshData<Real>> &md_h,
           std::shared_ptr<MeshData<Real>> &md_u, std::shared_ptr<MeshData<Real>> &md_x) {
          Real omega = solver->tt_ts.val[1] / solver->tt_ts.val[0];
          return AddFieldsAndStore<FieldTL>(md_h, md_u, md_x, 1.0, omega);
        },
        this, md_h, md_u, md_x);

    // 10. r <- s - omega t
    auto correct_r = itl.AddTask(
        get_tt_ts, "r <- s - omega t",
        [](BiCGSTABSolver *solver, std::shared_ptr<MeshData<Real>> &md_s,
           std::shared_ptr<MeshData<Real>> &md_t, std::shared_ptr<MeshData<Real>> &md_r) {
          Real omega = solver->tt_ts.val[1] / solver->tt_ts.val[0];
          return AddFieldsAndStore<FieldTL>(md_s, md_t, md_r, 1.0, -omega);
        },
        this, md_s, md_t, md_r);
    timer_omega->StopCollectingTasks();

    // Check and print out residual
    auto timer_res = solver_timings.GetOrAddAndRegister("BiCGSTAB: residual", itl);
    timer_res->StartCollectingTasks();
    // 11. rhat0r <- (rhat0, r) and residual
    auto get_res2_rhat0r = DoubleDotProduct<FieldTL>(correct_r, itl, &res_rhat0r, md_r, md_rhat0);
    get_res2_rhat0r = itl.AddTask(
        TaskQualifier::once_per_region, get_res2_rhat0r,
        [&](BiCGSTABSolver *solver, Mesh *pmesh) {
          Real rms_err = std::sqrt(solver->res_rhat0r.val[0] / pmesh->GetTotalCells());
          if (Globals::my_rank == 0 && solver->params_.print_per_step)
            printf("%i %e\n", solver->iter_counter * 2 + 2, rms_err);
          return TaskStatus::complete;
        },
        this, pmesh);

    // 12. beta <- rhat0r / rhat0r_old * alpha / omega
    // 13. p <- r + beta * (p - omega * v)
    auto update_p = itl.AddTask(
        get_res2_rhat0r, "p <- r + beta * (p - omega * v)",
        [](BiCGSTABSolver *solver, std::shared_ptr<MeshData<Real>> &md_p,
           std::shared_ptr<MeshData<Real>> &md_v, std::shared_ptr<MeshData<Real>> &md_r) {
          Real alpha = solver->rhat0r_old / solver->rhat0v.val;
          Real omega = solver->tt_ts.val[1] / solver->tt_ts.val[0];
          Real beta = solver->res_rhat0r.val[1] / solver->rhat0r_old * alpha / omega;
          AddFieldsAndStore<FieldTL>(md_p, md_v, md_p, 1.0, -omega);
          return AddFieldsAndStore<FieldTL>(md_r, md_p, md_p, 1.0, beta);
          return TaskStatus::complete;
        },
        this, md_p, md_v, md_r);

    // 14. rhat0r_old <- rhat0r, zero all reductions
    auto check = itl.AddTask(
        TaskQualifier::completion, update_p | correct_x, "rhat0r_old <- rhat0r",
        [partition](BiCGSTABSolver *solver, Mesh *pmesh, int max_iter,
                    std::shared_ptr<Real> res_tol, bool relative_residual) {
          Real rms_res = std::sqrt(solver->res_rhat0r.val[0] / pmesh->GetTotalCells());
          solver->final_residual = rms_res;
          solver->final_iteration = solver->iter_counter;
          Real tol = relative_residual
                         ? *res_tol * std::sqrt(solver->rhs2 / pmesh->GetTotalCells())
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
    timer_res->StopCollectingTasks();
    return tl.AddTask(solver_id, TF(CopyData<FieldTL>), md_x, md_u);
  }

  Real GetSquaredResidualSum() const { return res_rhat0r.val[0]; }
  int GetCurrentIterations() const { return iter_counter; }

  BiCGSTABParams &GetParams() { return params_; }

 protected:
  preconditioner_t preconditioner;
  BiCGSTABParams params_;
  int iter_counter;
  AllReduce<Real> rhat0v;
  AllReduce<utils::summable_array_t<Real, 2>> tt_ts, res_rhat0r;
  Real rhat0r_old, rhs2;
  equations_t eqs_;
  std::string container_;
};

} // namespace solvers

} // namespace parthenon

#endif // SOLVERS_BICGSTAB_SOLVER_HPP_
