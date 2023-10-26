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
#include "tasks/task_id.hpp"
#include "tasks/task_list.hpp"

namespace parthenon {

namespace solvers {

struct BiCGSTABParams {
  int max_iters = 10;
  Real residual_tolerance = 1.e-12;
  Real restart_threshold = -1.0;
  bool precondition = true;
};

template <class x, class rhs, class equations>
class BiCGSTABSolver {
 public:
  INTERNALSOLVERVARIABLE(x, rhat0);
  INTERNALSOLVERVARIABLE(x, v);
  INTERNALSOLVERVARIABLE(x, h);
  INTERNALSOLVERVARIABLE(x, s);
  INTERNALSOLVERVARIABLE(x, t);
  INTERNALSOLVERVARIABLE(x, r);
  INTERNALSOLVERVARIABLE(x, p);
  INTERNALSOLVERVARIABLE(x, u);

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

  TaskID AddTasks(TaskList &tl, IterativeTasks &itl, TaskID dependence, int i,
                  Mesh *pmesh, TaskRegion &region, int &reg_dep_id) {
    using namespace utils;
    auto &md = pmesh->mesh_data.GetOrAdd("base", i);
    iter_counter = 0;

    // Initialization: x <- 0, r <- rhs, rhat0 <- rhs,
    // rhat0r_old <- (rhat0, r), p <- r, u <- 0
    // TODO(LFR): Fix this to calculate the actual residual
    auto zero_x = tl.AddTask(dependence, SetToZero<x>, md);
    auto zero_u_init = tl.AddTask(dependence, SetToZero<u>, md);
    auto copy_r = tl.AddTask(dependence, CopyData<rhs, r>, md);
    auto copy_p = tl.AddTask(dependence, CopyData<rhs, p>, md);
    auto copy_rhat0 = tl.AddTask(dependence, CopyData<rhs, rhat0>, md);
    auto get_rhat0r_init =
        DotProduct<rhat0, r>(dependence, region, tl, i, reg_dep_id, &rhat0r, md);
    auto initialize = tl.AddTask(
        zero_x | zero_u_init | copy_r | copy_p | copy_rhat0 | get_rhat0r_init,
        [](BiCGSTABSolver *solver, int partition) {
          if (partition != 0) return TaskStatus::complete;
          solver->rhat0r_old = solver->rhat0r.val;
          solver->rhat0r.val = 0.0;
          solver->rhat0v.val = 0.0;
          solver->ts.val = 0.0;
          solver->tt.val = 0.0;
          solver->residual.val = 0.0;
          return TaskStatus::complete;
        },
        this, i);
    region.AddRegionalDependencies(reg_dep_id, i, initialize);
    reg_dep_id++;
    if (i == 0) {
      tl.AddTask(dependence, [&]() {
        if (Globals::my_rank == 0)
          printf("# [0] v-cycle\n# [1] rms-residual\n# [2] rms-error\n");
        return TaskStatus::complete;
      });
    }

    // BEGIN ITERATIVE TASKS

    // 1. u <- M p
    auto precon1 = initialize;
    if (params_.precondition) {
      auto set_rhs = itl.AddTask(precon1, CopyData<p, rhs>, md);
      auto zero_u = itl.AddTask(precon1, SetToZero<u>, md);
      precon1 = preconditioner.AddLinearOperatorTasks(region, itl, set_rhs | zero_u, i,
                                                      reg_dep_id, pmesh);
    } else {
      precon1 = itl.AddTask(initialize, CopyData<p, u>, md);
    }

    // 2. v <- A u
    auto comm = AddBoundaryExchangeTasks<BoundaryType::any>(precon1, itl, md, true);
    auto get_v = eqs_.template Ax<u, v>(itl, comm, md);

    // 3. rhat0v <- (rhat0, v)
    auto get_rhat0v =
        DotProduct<rhat0, v>(get_v, region, itl, i, reg_dep_id, &rhat0v, md);

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
    auto get_res = DotProduct<s, s>(correct_s, region, itl, i, reg_dep_id, &residual, md);

    auto print = itl.AddTask(
        get_res,
        [&](BiCGSTABSolver *solver, Mesh *pmesh, int partition) {
          if (partition != 0) return TaskStatus::complete;
          Real rms_res = std::sqrt(solver->residual.val / pmesh->GetTotalCells());
          if (Globals::my_rank == 0)
            printf("%i %e\n", solver->iter_counter * 2 + 1, rms_res);
          return TaskStatus::complete;
        },
        this, pmesh, i);

    // 6. u <- M s
    auto precon2 = correct_s;
    if (params_.precondition) {
      auto set_rhs = itl.AddTask(precon2, CopyData<s, rhs>, md);
      auto zero_u = itl.AddTask(precon2, SetToZero<u>, md);
      precon2 = preconditioner.AddLinearOperatorTasks(region, itl, set_rhs | zero_u, i,
                                                      reg_dep_id, pmesh);
    } else {
      precon2 = itl.AddTask(precon2, CopyData<s, u>, md);
    }

    // 7. t <- A u
    auto pre_t_comm = AddBoundaryExchangeTasks<BoundaryType::any>(precon2, itl, md, true);
    auto get_t = eqs_.template Ax<u, t>(itl, pre_t_comm, md);

    // 8. omega <- (t,s) / (t,t)
    auto get_ts = DotProduct<t, s>(get_t, region, itl, i, reg_dep_id, &ts, md);
    auto get_tt = DotProduct<t, t>(get_t, region, itl, i, reg_dep_id, &tt, md);

    // 9. x <- h + omega u
    auto correct_x = itl.AddTask(
        get_tt | get_ts,
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
    auto get_res2 =
        DotProduct<r, r>(correct_r, region, itl, i, reg_dep_id, &residual, md);

    if (i == 0) {
      get_res2 = itl.AddTask(
          get_res2,
          [&](BiCGSTABSolver *solver, Mesh *pmesh) {
            Real rms_err = std::sqrt(solver->residual.val / pmesh->GetTotalCells());
            if (Globals::my_rank == 0)
              printf("%i %e\n", solver->iter_counter * 2 + 2, rms_err);
            return TaskStatus::complete;
          },
          this, pmesh);
    }

    // 11. rhat0r <- (rhat0, r)
    auto get_rhat0r =
        DotProduct<rhat0, r>(correct_r, region, itl, i, reg_dep_id, &rhat0r, md);

    // 12. beta <- rhat0r / rhat0r_old * alpha / omega
    // 13. p <- r + beta * (p - omega * v)
    auto update_p = itl.AddTask(
        get_rhat0r | get_res2,
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
    region.AddRegionalDependencies(reg_dep_id, i, update_p | correct_x);
    auto check = itl.SetCompletionTask(
        update_p | correct_x,
        [](BiCGSTABSolver *solver, Mesh *pmesh, int partition, int max_iter,
           Real res_tol) {
          if (partition != 0) return TaskStatus::complete;
          solver->iter_counter++;
          Real rms_res = std::sqrt(solver->residual.val / pmesh->GetTotalCells());
          if (rms_res < res_tol || solver->iter_counter >= max_iter) {
            solver->final_residual = rms_res;
            solver->final_iteration = solver->iter_counter;
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
        this, pmesh, i, params_.max_iters, params_.residual_tolerance);
    region.AddGlobalDependencies(reg_dep_id, i, check);
    reg_dep_id++;

    return check;
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
