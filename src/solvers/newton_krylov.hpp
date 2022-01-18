//========================================================================================
// (C) (or copyright) 2021. Triad National Security, LLC. All rights reserved.
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
#ifndef SOLVERS_NEWTON_KRYLOV_HPP_
#define SOLVERS_NEWTON_KRYLOV_HPP_

#include <memory>
#include <string>
#include <vector>

#include "interface/mesh_data.hpp"
#include "interface/meshblock_data.hpp"
#include "interface/state_descriptor.hpp"
#include "kokkos_abstraction.hpp"
#include "solvers/solver_utils.hpp"
#include "tasks/task_id.hpp"
#include "tasks/task_list.hpp"

namespace parthenon {

namespace solvers {

struct NewtonKrylov_Counter {
  static int global_num_newton_solvers;
};

template <typename LinSolverType, typename DataType>
class NewtonKrylov : NewtonKrylov_Counter {
 public:
  NewtonKrylov() {}
  NewtonKrylov(StateDescriptor *pkg, const Real err, std::shared_ptr<LinSolverType> lin)
      : solver_name("NewtonKrylov" + std::to_string(global_num_newton_solvers++)),
        error_tol(err), linear_solver(lin),
        max_iters(pkg->Param<int>("newton_max_iterations")),
        check_interval(pkg->Param<int>("newton_check_interval")),
        fail_flag(pkg->Param<bool>("newton_abort_on_fail")),
        warn_flag(pkg->Param<bool>("newton_warn_on_fail")),
        sol_name(pkg->Param<std::string>("sol_name")),
        ResidualFunc(pkg->Param<decltype(ResidualFunc)>("ResidualFunc")),
        JacobianFunc(pkg->Param<decltype(JacobianFunc)>("JacobianFunc")) {}

  std::string label() { return "newton_" + linear_solver->label(); }
  std::vector<std::string> SolverState() { return {sol_name}; }

  TaskStatus DoNothing() { return TaskStatus::complete; }
  TaskStatus AddKrylovTasks(TaskID begin, const int i, TaskRegion *tr,
                            IterativeTasks *lsolver, std::shared_ptr<DataType> &md,
                            std::shared_ptr<DataType> &mout) {
    krylov[i] = linear_solver->createTaskList(begin, i, *tr, *lsolver, md, mout);
    return TaskStatus::complete;
  }
  TaskStatus CheckKrylovTasks(const int &i, TaskList *tl) {
    return (tl->CheckDependencies(krylov[i]) ? TaskStatus::complete
                                             : TaskStatus::incomplete);
  }
  TaskStatus Update(DataType *u, DataType *du) {
    const auto &ib = u->GetBoundsI(IndexDomain::interior);
    const auto &jb = u->GetBoundsJ(IndexDomain::interior);
    const auto &kb = u->GetBoundsK(IndexDomain::interior);

    PackIndexMap imap;
    const std::vector<std::string> vars({sol_name});
    const auto &v = u->PackVariables(vars, imap);
    const int ivlo = imap[sol_name].first;
    const int ivhi = imap[sol_name].second;

    const auto &dv = du->PackVariables(vars, imap);
    const Real alp = alpha_ls;
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "NewtonKrylov::Update", DevExecSpace(), 0, v.GetDim(5) - 1,
        ivlo, ivhi, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int n, const int k, const int j, const int i) {
          v(b, n, k, j, i) += alp * dv(b, n, k, j, i);
        });
    return TaskStatus::complete;
  }
  TaskStatus Copy(DataType *u, DataType *du) {
    const auto &ib = u->GetBoundsI(IndexDomain::interior);
    const auto &jb = u->GetBoundsJ(IndexDomain::interior);
    const auto &kb = u->GetBoundsK(IndexDomain::interior);

    PackIndexMap imap;
    const std::vector<std::string> vars({sol_name});
    const auto &dv = du->PackVariables(vars, imap);
    const int ivlo = imap[sol_name].first;
    const int ivhi = imap[sol_name].second;
    PackIndexMap imap2;
    const std::vector<std::string> vars2({"delta"});
    const auto &v = u->PackVariables(vars2, imap2);
    const int idel = imap2["delta"].first;

    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "NewtonKrylov::Update", DevExecSpace(), 0, v.GetDim(5) - 1,
        kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          v(b, idel, k, j, i) = dv(b, ivlo, k, j, i);
        });
    return TaskStatus::complete;
  }
  TaskStatus CheckConvergence() {
    std::cout << "Newton err: " << l2res.val << std::endl;
    old_res = l2res.val;
    if (l2res.val < error_tol) return TaskStatus::complete;
    return TaskStatus::iterate;
  }
  template <typename T>
  TaskID GetResidual(const TaskID &dep, T &tasks, const int i, RegionCounter &reg,
                     TaskRegion &tr, DataType *md) {
    // zero residual
    TaskID none(0);
    auto zres = (i == 0 ? tasks.AddTask(
                              none,
                              [](Real *val) {
                                *val = 0.0;
                                return TaskStatus::complete;
                              },
                              &l2res.val)
                        : tasks.AddTask(none, &NewtonKrylov::DoNothing, this));
    tr.AddRegionalDependencies(reg.ID(), i, zres);

    // compute new residual
    auto new_res = tasks.AddTask(dep, ResidualFunc, md, &l2res.val);
    tr.AddRegionalDependencies(reg.ID(), i, new_res);

    // do the global reduciton
    auto start_global_res =
        (i == 0 ? tasks.AddTask(new_res, &AllReduce<Real>::StartReduce, &l2res, MPI_SUM)
                : new_res);
    auto finish_global_res =
        tasks.AddTask(start_global_res, &AllReduce<Real>::CheckReduce, &l2res);
    return finish_global_res;
  }

  TaskStatus CheckLineSearch() {
    search_iters++;
    const Real c = 1.e-4;
    const Real afac = 0.5;
    bool check = (0.5 * l2res.val < 0.5 * old_res - c * l2res.val);
    if (!check) {
      // scale delta x
      if (alpha_ls > 0.0)
        alpha_ls *= -afac;
      else
        alpha_ls *= afac;
      return TaskStatus::iterate;
    } else {
      return TaskStatus::complete;
    }
  }

  /*auto lsearch_tasks = solver.AddTask(none,
     &NewtonKrylov<LinSolverType,DataType>::AddSearchTasks, this, lsolver_complete, i,
     &tr, &lsearch, md, mdelta);*/
  /*TaskStatus AddKrylovTasks(TaskID begin,
                            const int &i,
                            TaskRegion *tr,
                            IterativeTasks *lsolver,
                            std::shared_ptr<DataType> &md,
                            std::shared_ptr<DataType> &mout*/
  TaskStatus AddSearchTasks(const TaskID &begin, const int &i, TaskRegion *tr,
                            IterativeTasks *ls, std::shared_ptr<DataType> &md,
                            std::shared_ptr<DataType> &mdelta) {
    TaskID none(0);
    RegionCounter reg(solver_name + "_lsearch");
    alpha_ls = 1.0;
    search_iters = 0;
    // update the guess
    auto update = ls->AddTask(begin, &NewtonKrylov<LinSolverType, DataType>::Update, this,
                              md.get(), mdelta.get());
    // share \Delta x
    auto start_recv =
        ls->AddTask(none, &DataType::StartReceiving, md.get(), BoundaryCommSubset::all);
    auto send =
        ls->AddTask(update, parthenon::cell_centered_bvars::SendBoundaryBuffers, md);
    auto recv = ls->AddTask(start_recv,
                            parthenon::cell_centered_bvars::ReceiveBoundaryBuffers, md);
    auto setb =
        ls->AddTask(recv | update, parthenon::cell_centered_bvars::SetBoundaries, md);
    auto clear = ls->AddTask(send | setb, &DataType::ClearBoundary, md.get(),
                             BoundaryCommSubset::all);

    // apply physical boundary conditions
    auto copy = ls->AddTask(setb, &NewtonKrylov<LinSolverType, DataType>::Copy, this,
                            md.get(), mdelta.get());
    auto new_res = GetResidual(setb | copy, *ls, i, reg, *tr, md.get());

    line_search[i] = ls->SetCompletionTask(
        new_res, &NewtonKrylov<LinSolverType, DataType>::CheckLineSearch, this);
    tr->AddRegionalDependencies(reg.ID(), i, line_search[i]);

    return TaskStatus::complete;
  }
  TaskStatus CheckSearchTasks(const int &i, TaskList *tl) {
    return (tl->CheckDependencies(line_search[i]) ? TaskStatus::complete
                                                  : TaskStatus::incomplete);
  }

  TaskID createTaskList(TaskID begin, const int i, TaskRegion &tr,
                        std::shared_ptr<DataType> md, std::shared_ptr<DataType> mdelta) {
    TaskID none(0);
    TaskList &tl = tr[i];
    RegionCounter reg(solver_name);

    // get the initial residual
    auto get_res0 = GetResidual(begin, tl, i, reg, tr, md.get());
    auto save_res = (i == 0 ? tl.AddTask(
                                  none,
                                  [](Real *old_val, Real *val) {
                                    *old_val = *val;
                                    return TaskStatus::complete;
                                  },
                                  &old_res, &l2res.val)
                            : tl.AddTask(none, &NewtonKrylov::DoNothing, this));
    tr.AddRegionalDependencies(reg.ID(), i, save_res);

    // set up the iterative task list for the nonlinear solver (outer iteration)
    auto &solver = tl.AddIteration(solver_name);
    solver.SetMaxIterations(max_iters);
    solver.SetCheckInterval(check_interval);
    solver.SetFailWithMaxIterations(fail_flag);
    solver.SetWarnWithMaxIterations(warn_flag);
    // Evaluate Jacobian
    auto get_jac = solver.AddTask(begin, JacobianFunc, md.get());

    // set up the iterative task list for the linear solver (inner interation)
    auto &lsolver = tl.AddIteration(solver_name + "_lsolver");
    lsolver.SetMaxIterations(linear_solver->MaxIters());
    lsolver.SetCheckInterval(linear_solver->CheckInterval());
    lsolver.SetFailWithMaxIterations(linear_solver->GetFail());
    lsolver.SetWarnWithMaxIterations(linear_solver->GetWarn());
    // add linear solver tasks to list for each outer iteration
    auto lin_tasks =
        solver.AddTask(none, &NewtonKrylov<LinSolverType, DataType>::AddKrylovTasks, this,
                       get_jac | get_res0, i, &tr, &lsolver, md, mdelta);

    // check if the linear system has been solved
    auto lsolver_complete = solver.AddTask(
        lin_tasks, &NewtonKrylov<LinSolverType, DataType>::CheckKrylovTasks, this, i,
        &tl);

    // line search
    auto &lsearch = tl.AddIteration(solver_name + "_lsearch");
    lsearch.SetMaxIterations(50);
    lsearch.SetCheckInterval(1);
    lsearch.SetFailWithMaxIterations(false);
    lsearch.SetWarnWithMaxIterations(false);
    auto lsearch_tasks =
        solver.AddTask(none, &NewtonKrylov<LinSolverType, DataType>::AddSearchTasks, this,
                       lsolver_complete, i, &tr, &lsearch, md, mdelta);
    auto lsearch_complete = solver.AddTask(
        lsearch_tasks, &NewtonKrylov<LinSolverType, DataType>::CheckSearchTasks, this, i,
        &tl);

    // check stopping criteria
    auto converged = solver.SetCompletionTask(
        lsearch_complete, &NewtonKrylov<LinSolverType, DataType>::CheckConvergence, this);
    tr.AddGlobalDependencies(reg.ID(), i, converged);

    return converged;
  }

 private:
  std::string solver_name;
  Real error_tol;
  std::shared_ptr<LinSolverType> linear_solver;
  int max_iters;
  int check_interval;
  bool fail_flag;
  bool warn_flag;
  std::string sol_name;
  std::function<TaskStatus(DataType *, Real *)> ResidualFunc;
  std::function<TaskStatus(DataType *)> JacobianFunc;
  AllReduce<Real> l2res;
  Real old_res, alpha_ls;
  int search_iters;
  std::map<int, TaskID> krylov;
  std::map<int, TaskID> line_search;
};

} // namespace solvers

} // namespace parthenon

#endif // SOLVERS_NEWTON_KRYLOV_HPP_
