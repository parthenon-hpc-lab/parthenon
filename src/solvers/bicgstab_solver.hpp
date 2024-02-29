//========================================================================================
// (C) (or copyright) 2023. Triad National Security, LLC. All rights reserved.
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

namespace parthenon {

namespace solvers {

struct BiCGSTABParams {
  int max_iters = 10;
  Real residual_tolerance = 1.e-12;
  Real restart_threshold = -1.0;
  bool precondition = true;
};

// The equations class must include a template method
//
//   template <class x_t, class y_t, class TL_t>
//   TaskID Ax(TL_t &tl, TaskID depends_on, std::shared_ptr<MeshData<Real>> &md)
//
// that takes a field associated with x_t and applies
// the matrix A to it and stores the result in y_t.
template <class x, class rhs, class equations>
class BiCGSTABSolver {
 public:
  PARTHENON_INTERNALSOLVERVARIABLE(x, rhat0);
  PARTHENON_INTERNALSOLVERVARIABLE(x, v);
  PARTHENON_INTERNALSOLVERVARIABLE(x, h);
  PARTHENON_INTERNALSOLVERVARIABLE(x, s);
  PARTHENON_INTERNALSOLVERVARIABLE(x, t);
  PARTHENON_INTERNALSOLVERVARIABLE(x, r);
  PARTHENON_INTERNALSOLVERVARIABLE(x, p);
  PARTHENON_INTERNALSOLVERVARIABLE(x, u);

  BiCGSTABSolver(StateDescriptor *pkg, BiCGSTABParams params_in,
                 equations eq_in = equations(), std::vector<int> shape = {})
      : preconditioner(pkg, MGParams(), eq_in, shape), params_(params_in),
        iter_counter(0), eqs_(eq_in) {
    using namespace refinement_ops;
    auto mu = Metadata({Metadata::Cell, Metadata::Independent, Metadata::FillGhost,
                        Metadata::WithFluxes, Metadata::GMGRestrict},
                       shape);
    mu.RegisterRefinementOps<ProlongateSharedLinear, RestrictAverage>();
    auto m_no_ghost =
        Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy}, shape);
    pkg->AddField(u::name(), mu);
    pkg->AddField(rhat0::name(), m_no_ghost);
    pkg->AddField(v::name(), m_no_ghost);
    pkg->AddField(h::name(), m_no_ghost);
    pkg->AddField(s::name(), m_no_ghost);
    pkg->AddField(t::name(), m_no_ghost);
    pkg->AddField(r::name(), m_no_ghost);
    pkg->AddField(p::name(), m_no_ghost);
  }

  TaskID AddTasks(TaskList &tl, TaskID dependence, Mesh *pmesh, const int partition) {
    using namespace utils;
    TaskID none;
    auto &md = pmesh->mesh_data.GetOrAdd("base", partition);
    iter_counter = 0;

    // Initialization: x <- 0, r <- rhs, rhat0 <- rhs,
    // rhat0r_old <- (rhat0, r), p <- r, u <- 0
    auto zero_x = tl.AddTask(dependence, SetToZero<x>, md);
    auto zero_u_init = tl.AddTask(dependence, SetToZero<u>, md);
    auto copy_r = tl.AddTask(dependence, CopyData<rhs, r>, md);
    auto copy_p = tl.AddTask(dependence, CopyData<rhs, p>, md);
    auto copy_rhat0 = tl.AddTask(dependence, CopyData<rhs, rhat0>, md);
    auto get_rhat0r_init = DotProduct<rhat0, r>(dependence, tl, &rhat0r, md);
    auto initialize = tl.AddTask(
        TaskQualifier::once_per_region | TaskQualifier::local_sync,
        zero_x | zero_u_init | copy_r | copy_p | copy_rhat0 | get_rhat0r_init,
        [](BiCGSTABSolver *solver) {
          solver->rhat0r_old = solver->rhat0r.val;
          solver->rhat0r.val = 0.0;
          solver->rhat0v.val = 0.0;
          solver->ts.val = 0.0;
          solver->tt.val = 0.0;
          solver->residual.val = 0.0;
          return TaskStatus::complete;
        },
        this);
    tl.AddTask(TaskQualifier::once_per_region, dependence, [&]() {
      if (Globals::my_rank == 0)
        printf("# [0] v-cycle\n# [1] rms-residual\n# [2] rms-error\n");
      return TaskStatus::complete;
    });

    // BEGIN ITERATIVE TASKS
    auto [itl, solver_id] = tl.AddSublist(initialize, {1, params_.max_iters});

    // 1. u <- M p
    auto precon1 = none;
    if (params_.precondition) {
      auto set_rhs = itl.AddTask(precon1, CopyData<p, rhs>, md);
      auto zero_u = itl.AddTask(precon1, SetToZero<u>, md);
      precon1 =
          preconditioner.AddLinearOperatorTasks(itl, set_rhs | zero_u, partition, pmesh);
    } else {
      precon1 = itl.AddTask(none, CopyData<p, u>, md);
    }

    // 2. v <- A u
    auto comm = AddBoundaryExchangeTasks<BoundaryType::any>(precon1, itl, md, true);
    auto get_v = eqs_.template Ax<u, v>(itl, comm, md);

    // 3. rhat0v <- (rhat0, v)
    auto get_rhat0v = DotProduct<rhat0, v>(get_v, itl, &rhat0v, md);

    // 4. h <- x + alpha u (alpha = rhat0r_old / rhat0v)
    auto correct_h = itl.AddTask(
        get_rhat0v,
        [](BiCGSTABSolver *solver, std::shared_ptr<MeshData<Real>> &md) {
          Real alpha = solver->rhat0r_old / solver->rhat0v.val;
          return AddFieldsAndStore<x, u, h>(md, 1.0, alpha);
        },
        this, md);

    // 5. s <- r - alpha v (alpha = rhat0r_old / rhat0v)
    auto correct_s = itl.AddTask(
        get_rhat0v,
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
          if (Globals::my_rank == 0)
            printf("%i %e\n", solver->iter_counter * 2 + 1, rms_res);
          return TaskStatus::complete;
        },
        this, pmesh);

    // 6. u <- M s
    auto precon2 = correct_s;
    if (params_.precondition) {
      auto set_rhs = itl.AddTask(precon2, CopyData<s, rhs>, md);
      auto zero_u = itl.AddTask(precon2, SetToZero<u>, md);
      precon2 =
          preconditioner.AddLinearOperatorTasks(itl, set_rhs | zero_u, partition, pmesh);
    } else {
      precon2 = itl.AddTask(precon2, CopyData<s, u>, md);
    }

    // 7. t <- A u
    auto pre_t_comm = AddBoundaryExchangeTasks<BoundaryType::any>(precon2, itl, md, true);
    auto get_t = eqs_.template Ax<u, t>(itl, pre_t_comm, md);

    // 8. omega <- (t,s) / (t,t)
    auto get_ts = DotProduct<t, s>(get_t, itl, &ts, md);
    auto get_tt = DotProduct<t, t>(get_t, itl, &tt, md);

    // 9. x <- h + omega u
    auto correct_x = itl.AddTask(
        TaskQualifier::local_sync, get_tt | get_ts,
        [](BiCGSTABSolver *solver, std::shared_ptr<MeshData<Real>> &md) {
          Real omega = solver->ts.val / solver->tt.val;
          return AddFieldsAndStore<h, u, x>(md, 1.0, omega);
        },
        this, md);

    // 10. r <- s - omega t
    auto correct_r = itl.AddTask(
        get_tt | get_ts,
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
          if (Globals::my_rank == 0)
            printf("%i %e\n", solver->iter_counter * 2 + 2, rms_err);
          return TaskStatus::complete;
        },
        this, pmesh);

    // 11. rhat0r <- (rhat0, r)
    auto get_rhat0r = DotProduct<rhat0, r>(correct_r, itl, &rhat0r, md);

    // 12. beta <- rhat0r / rhat0r_old * alpha / omega
    // 13. p <- r + beta * (p - omega * v)
    auto update_p = itl.AddTask(
        TaskQualifier::local_sync, get_rhat0r | get_res2,
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
        TaskQualifier::completion | TaskQualifier::once_per_region |
            TaskQualifier::global_sync,
        update_p | correct_x,
        [](BiCGSTABSolver *solver, Mesh *pmesh, Real res_tol) {
          solver->iter_counter++;
          Real rms_res = std::sqrt(solver->residual.val / pmesh->GetTotalCells());
          solver->final_residual = rms_res;
          solver->final_iteration = solver->iter_counter;
          if (rms_res < res_tol) {
            return TaskStatus::complete;
          }
          solver->rhat0r_old = solver->rhat0r.val;
          solver->rhat0r.val = 0.0;
          solver->rhat0v.val = 0.0;
          solver->ts.val = 0.0;
          solver->tt.val = 0.0;
          solver->residual.val = 0.0;
          return TaskStatus::iterate;
        },
        this, pmesh, params_.residual_tolerance);

    return solver_id;
  }

  Real GetSquaredResidualSum() const { return residual.val; }
  int GetCurrentIterations() const { return iter_counter; }

  Real GetFinalResidual() const { return final_residual; }
  int GetFinalIterations() const { return final_iteration; }

 protected:
  MGSolver<u, rhs, equations> preconditioner;
  BiCGSTABParams params_;
  int iter_counter;
  AllReduce<Real> rtr, pAp, rhat0v, rhat0r, ts, tt, residual;
  Real rhat0r_old;
  equations eqs_;
  Real final_residual;
  int final_iteration;
};

} // namespace solvers

} // namespace parthenon

#endif // SOLVERS_BICGSTAB_SOLVER_HPP_
