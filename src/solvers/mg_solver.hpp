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
#ifndef SOLVERS_MG_SOLVER_HPP_
#define SOLVERS_MG_SOLVER_HPP_

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "interface/mesh_data.hpp"
#include "interface/meshblock_data.hpp"
#include "interface/state_descriptor.hpp"
#include "kokkos_abstraction.hpp"
#include "solvers/solver_utils.hpp"

#include "tasks/tasks.hpp"

namespace parthenon {

namespace solvers {

struct MGParams {
  int max_iters = 10;
  Real residual_tolerance = 1.e-12;
  bool do_FAS = true;
  std::string smoother = "SRJ2";
  bool two_by_two_diagonal = false;
  int max_coarsenings = std::numeric_limits<int>::max();
};

// The equations class must include a template method
//
//   template <class x_t, class y_t, class TL_t>
//   TaskID Ax(TL_t &tl, TaskID depends_on, std::shared_ptr<MeshData<Real>> &md)
//
// that takes a field associated with x_t and applies
// the matrix A to it and stores the result in y_t. Additionally,
// it must include a template method
//
//  template <class diag_t>
//  TaskStatus SetDiagonal(std::shared_ptr<MeshData<Real>> &md)
//
// That stores the (possibly approximate) diagonal of matrix A in the field
// associated with the type diag_t. This is used for Jacobi iteration.
template <class u, class rhs, class equations>
class MGSolver {
 public:
  PARTHENON_INTERNALSOLVERVARIABLE(
      u, res_err); // residual on the way up and error on the way down
  PARTHENON_INTERNALSOLVERVARIABLE(u, temp); // Temporary storage
  PARTHENON_INTERNALSOLVERVARIABLE(u, u0);   // Storage for initial solution during FAS
  PARTHENON_INTERNALSOLVERVARIABLE(u, D);    // Storage for (approximate) diagonal
  std::vector<std::string> GetInternalVariableNames() const {
    return {res_err::name(), temp::name(), u0::name(), D::name()};
  }

  MGSolver(StateDescriptor *pkg, MGParams params_in, equations eq_in = equations(),
           std::vector<int> shape = {})
      : params_(params_in), iter_counter(0), eqs_(eq_in) {
    using namespace parthenon::refinement_ops;
    // The ghost cells of res_err need to be filled, but this is accomplished by
    // copying res_err into u, communicating, then copying u back into res_err
    // across all zones in a block
    auto mres_err =
        Metadata({Metadata::Cell, Metadata::Independent, Metadata::GMGRestrict,
                  Metadata::GMGProlongate, Metadata::OneCopy},
                 shape);
    mres_err.RegisterRefinementOps<ProlongateSharedLinear, RestrictAverage>();
    pkg->AddField(res_err::name(), mres_err);

    auto mtemp =
        Metadata({Metadata::Cell, Metadata::Independent, Metadata::OneCopy}, shape);
    mtemp.RegisterRefinementOps<ProlongateSharedLinear, RestrictAverage>();
    pkg->AddField(temp::name(), mtemp);

    auto mu0 = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy}, shape);
    pkg->AddField(u0::name(), mu0);
    auto Dshape = shape;
    if (params_.two_by_two_diagonal) {
      Dshape = std::vector<int>{4};
    }
    auto mD = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy}, Dshape);
    pkg->AddField(D::name(), mD);
  }

  TaskID AddTasks(TaskList &tl, TaskID dependence, Mesh *pmesh, const int partition) {
    using namespace utils;
    TaskID none;
    auto [itl, solve_id] = tl.AddSublist(dependence, {1, this->params_.max_iters});
    iter_counter = 0;
    itl.AddTask(
        TaskQualifier::once_per_region, none,
        [](int *iter_counter) {
          if (*iter_counter > 0 || Globals::my_rank != 0) return TaskStatus::complete;
          printf("# [0] v-cycle\n# [1] rms-residual\n# [2] rms-error\n");
          return TaskStatus::complete;
        },
        &iter_counter);
    auto mg_finest = AddLinearOperatorTasks(itl, none, partition, pmesh);
    auto &md = pmesh->mesh_data.GetOrAdd("base", partition);
    auto comm = AddBoundaryExchangeTasks<BoundaryType::any>(mg_finest, itl, md,
                                                            pmesh->multilevel);
    auto calc_pointwise_res = eqs_.template Ax<u, res_err>(itl, comm, md);
    calc_pointwise_res = itl.AddTask(
        calc_pointwise_res, AddFieldsAndStoreInteriorSelect<rhs, res_err, res_err>, md,
        1.0, -1.0, false);
    auto get_res = DotProduct<res_err, res_err>(calc_pointwise_res, itl, &residual, md);

    auto check = itl.AddTask(
        TaskQualifier::once_per_region | TaskQualifier::completion |
            TaskQualifier::global_sync,
        get_res,
        [](MGSolver *solver, Mesh *pmesh) {
          solver->iter_counter++;
          Real rms_res = std::sqrt(solver->residual.val / pmesh->GetTotalCells());
          if (Globals::my_rank == 0) printf("%i %e\n", solver->iter_counter, rms_res);
          solver->final_residual = rms_res;
          solver->final_iteration = solver->iter_counter;
          if (rms_res > solver->params_.residual_tolerance) return TaskStatus::iterate;
          return TaskStatus::complete;
        },
        this, pmesh);

    return solve_id;
  }

  TaskID AddLinearOperatorTasks(TaskList &tl, TaskID dependence, int partition,
                                Mesh *pmesh) {
    using namespace utils;
    iter_counter = 0;

    int min_level = std::max(pmesh->GetGMGMaxLevel() - params_.max_coarsenings,
                             pmesh->GetGMGMinLevel());
    int max_level = pmesh->GetGMGMaxLevel();

    return AddMultiGridTasksPartitionLevel(tl, dependence, partition, max_level,
                                           min_level, max_level, pmesh);
  }

  template <class TL_t>
  TaskID AddSetupTasks(TL_t &tl, TaskID dependence, int partition, Mesh *pmesh) {
    using namespace utils;

    int min_level = std::max(pmesh->GetGMGMaxLevel() - params_.max_coarsenings,
                             pmesh->GetGMGMinLevel());
    int max_level = pmesh->GetGMGMaxLevel();

    return AddMultiGridSetupPartitionLevel(tl, dependence, partition, max_level,
                                           min_level, max_level, pmesh);
  }

  Real GetSquaredResidualSum() const { return residual.val; }
  int GetCurrentIterations() const { return iter_counter; }
  Real GetFinalResidual() const { return final_residual; }
  int GetFinalIterations() const { return final_iteration; }

 protected:
  MGParams params_;
  int iter_counter;
  AllReduce<Real> residual;
  equations eqs_;
  Real final_residual;
  int final_iteration;
  // These functions apparently have to be public to compile with cuda since
  // they contain device side lambdas
 public:
  template <class rhs_t, class Axold_t, class D_t, class xold_t, class xnew_t>
  TaskStatus Jacobi(std::shared_ptr<MeshData<Real>> &md, double weight) {
    using namespace parthenon;
    const int ndim = md->GetMeshPointer()->ndim;
    using TE = parthenon::TopologicalElement;
    TE te = TE::CC;
    IndexRange ib = md->GetBoundsI(IndexDomain::interior, te);
    IndexRange jb = md->GetBoundsJ(IndexDomain::interior, te);
    IndexRange kb = md->GetBoundsK(IndexDomain::interior, te);

    int nblocks = md->NumBlocks();
    std::vector<bool> include_block(nblocks, true);

    static auto desc =
        parthenon::MakePackDescriptor<xold_t, xnew_t, Axold_t, rhs_t, D_t>(md.get());
    auto pack = desc.GetPack(md.get(), include_block);
    if (params_.two_by_two_diagonal) {
      parthenon::par_for(
          DEFAULT_LOOP_PATTERN, "CaclulateFluxes", DevExecSpace(), 0,
          pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
          KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
            const auto &coords = pack.GetCoordinates(b);

            const Real D11 = pack(b, te, D_t(0), k, j, i);
            const Real D22 = pack(b, te, D_t(1), k, j, i);
            const Real D12 = pack(b, te, D_t(2), k, j, i);
            const Real D21 = pack(b, te, D_t(3), k, j, i);
            const Real det = D11 * D22 - D12 * D21;

            const Real Du0 = D11 * pack(b, te, xold_t(0), k, j, i) +
                             D12 * pack(b, te, xold_t(1), k, j, i);
            const Real Du1 = D21 * pack(b, te, xold_t(0), k, j, i) +
                             D22 * pack(b, te, xold_t(1), k, j, i);

            const Real t0 =
                pack(b, te, rhs_t(0), k, j, i) - pack(b, te, Axold_t(0), k, j, i) + Du0;
            const Real t1 =
                pack(b, te, rhs_t(1), k, j, i) - pack(b, te, Axold_t(1), k, j, i) + Du1;

            const Real v0 = (D22 * t0 - D12 * t1) / det;
            const Real v1 = (-D21 * t0 + D11 * t1) / det;

            pack(b, te, xnew_t(0), k, j, i) =
                weight * v0 + (1.0 - weight) * pack(b, te, xold_t(0), k, j, i);
            pack(b, te, xnew_t(1), k, j, i) =
                weight * v1 + (1.0 - weight) * pack(b, te, xold_t(1), k, j, i);
          });
    } else {
      const int scratch_size = 0;
      const int scratch_level = 0;
      const int nghost = Globals::nghost;
      parthenon::par_for_outer(
          DEFAULT_OUTER_LOOP_PATTERN, "Jacobi", DevExecSpace(), scratch_size,
          scratch_level, 0, pack.GetNBlocks() - 1, kb.s, kb.e,
          KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int k) {
            const int nvars =
                pack.GetUpperBound(b, xnew_t()) - pack.GetLowerBound(b, xnew_t()) + 1;
            for (int c = 0; c < nvars; ++c) {
              Real *Ax = &pack(b, te, Axold_t(c), k, jb.s, ib.s);
              Real *diag = &pack(b, te, D_t(c), k, jb.s, ib.s);
              Real *prhs = &pack(b, te, rhs_t(c), k, jb.s, ib.s);
              Real *xo = &pack(b, te, xold_t(c), k, jb.s, ib.s);
              Real *xn = &pack(b, te, xnew_t(c), k, jb.s, ib.s);
              // Use ptr arithmetic to get the number of points we need to go over
              // (including ghost zones) to get from (k, jb.s, ib.s) to (k, jb.e, ib.e)
              const int npoints = &pack(b, te, Axold_t(c), k, jb.e, ib.e) - Ax + 1;
              parthenon::par_for_inner(
                  DEFAULT_INNER_LOOP_PATTERN, member, 0, npoints - 1, [&](const int idx) {
                    const Real off_diag = Ax[idx] - diag[idx] * xo[idx];
                    const Real val = prhs[idx] - off_diag;
                    xn[idx] = weight * val / diag[idx] + (1.0 - weight) * xo[idx];
                  });
            }
          });
    }
    return TaskStatus::complete;
  }

  template <parthenon::BoundaryType comm_boundary, class in_t, class out_t, class TL_t>
  TaskID AddJacobiIteration(TL_t &tl, TaskID depends_on, bool multilevel, Real omega,
                            std::shared_ptr<MeshData<Real>> &md,
                            std::shared_ptr<MeshData<Real>> &md_comm) {
    using namespace utils;

    auto comm =
        AddBoundaryExchangeTasks<comm_boundary>(depends_on, tl, md_comm, multilevel);
    auto mat_mult = eqs_.template Ax<in_t, out_t>(tl, comm, md);
    return tl.AddTask(mat_mult, &MGSolver::Jacobi<rhs, out_t, D, in_t, out_t>, this, md,
                      omega);
  }

  template <parthenon::BoundaryType comm_boundary, class TL_t>
  TaskID AddSRJIteration(TL_t &tl, TaskID depends_on, int stages, bool multilevel,
                         std::shared_ptr<MeshData<Real>> &md,
                         std::shared_ptr<MeshData<Real>> &md_comm) {
    using namespace utils;
    int ndim = md->GetParentPointer()->ndim;

    std::array<std::array<Real, 3>, 3> omega_M1{
        {{1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}}};
    // Damping factors from Yang & Mittal (2017)
    std::array<std::array<Real, 3>, 3> omega_M2{
        {{0.8723, 0.5395, 0.0000}, {1.3895, 0.5617, 0.0000}, {1.7319, 0.5695, 0.0000}}};
    std::array<std::array<Real, 3>, 3> omega_M3{
        {{0.9372, 0.6667, 0.5173}, {1.6653, 0.8000, 0.5264}, {2.2473, 0.8571, 0.5296}}};

    if (stages == 0) return depends_on;
    auto omega = omega_M1;
    if (stages == 2) omega = omega_M2;
    if (stages == 3) omega = omega_M3;
    // This copy is to set the coarse blocks in temp to the values in u so that
    // fine-coarse boundaries of temp are correctly updated during communication
    depends_on = tl.AddTask(depends_on, CopyData<u, temp, false>, md);
    auto jacobi1 = AddJacobiIteration<comm_boundary, u, temp>(
        tl, depends_on, multilevel, omega[ndim - 1][0], md, md_comm);
    auto copy1 = tl.AddTask(jacobi1, CopyData<temp, u, true>, md);
    if (stages < 2) return copy1;
    auto jacobi2 = AddJacobiIteration<comm_boundary, u, temp>(
        tl, copy1, multilevel, omega[ndim - 1][1], md, md_comm);
    auto copy2 = tl.AddTask(jacobi2, CopyData<temp, u, true>, md);
    if (stages < 3) return copy2;
    auto jacobi3 = AddJacobiIteration<comm_boundary, u, temp>(
        tl, copy2, multilevel, omega[ndim - 1][2], md, md_comm);
    return tl.AddTask(jacobi3, CopyData<temp, u, true>, md);
  }

  template <class TL_t>
  TaskID AddMultiGridSetupPartitionLevel(TL_t &tl, TaskID dependence, int partition,
                                         int level, int min_level, int max_level,
                                         Mesh *pmesh) {
    using namespace utils;

    bool multilevel = (level != min_level);

    auto &md = pmesh->gmg_mesh_data[level].GetOrAdd(level, "base", partition);

    auto task_out = dependence;
    if (level < max_level) {
      task_out =
          tl.AddTask(task_out, ReceiveBoundBufs<BoundaryType::gmg_restrict_recv>, md);
      task_out = tl.AddTask(task_out, SetBounds<BoundaryType::gmg_restrict_recv>, md);
    }

    // If we are finer than the coarsest level:
    if (level > min_level) {
      task_out = tl.AddTask(task_out, SendBoundBufs<BoundaryType::gmg_restrict_send>, md);
      task_out = AddMultiGridSetupPartitionLevel(tl, task_out, partition, level - 1,
                                                 min_level, max_level, pmesh);
    }

    // The boundaries are not up to date on return
    return task_out;
  }

  TaskID AddMultiGridTasksPartitionLevel(TaskList &tl, TaskID dependence, int partition,
                                         int level, int min_level, int max_level,
                                         Mesh *pmesh) {
    using namespace utils;
    auto smoother = params_.smoother;
    bool do_FAS = params_.do_FAS;
    int pre_stages, post_stages;
    if (smoother == "none") {
      pre_stages = 0;
      post_stages = 0;
    } else if (smoother == "SRJ1") {
      pre_stages = 1;
      post_stages = 1;
    } else if (smoother == "SRJ2") {
      pre_stages = 2;
      post_stages = 2;
    } else if (smoother == "SRJ3") {
      pre_stages = 3;
      post_stages = 3;
    } else {
      PARTHENON_FAIL("Unknown solver type.");
    }

    bool multilevel = (level != min_level);

    auto &md = pmesh->gmg_mesh_data[level].GetOrAdd(level, "base", partition);
    std::string label =
        "mg_comm_" + std::to_string(level) + "_" + std::to_string(partition);
    auto &md_comm = pmesh->gmg_mesh_data[level].AddShallow(
        label, md, std::vector<std::string>{u::name(), res_err::name()});

    // 0. Receive residual from coarser level if there is one
    auto set_from_finer = dependence;
    if (level < max_level) {
      // Fill fields with restricted values
      auto recv_from_finer = tl.AddTask(
          dependence, ReceiveBoundBufs<BoundaryType::gmg_restrict_recv>, md_comm);
      set_from_finer = tl.AddTask( // TaskQualifier::local_sync, // is this required?
          recv_from_finer, SetBounds<BoundaryType::gmg_restrict_recv>, md_comm);
      // 1. Copy residual from dual purpose communication field to the rhs, should be
      // actual RHS for finest level
      if (!do_FAS) {
        auto zero_u = tl.AddTask(set_from_finer, SetToZero<u, true>, md);
        auto copy_rhs = tl.AddTask(set_from_finer, CopyData<res_err, rhs, true>, md);
        set_from_finer = zero_u | copy_rhs;
      } else {
        // TODO(LFR): Determine if this boundary exchange task is required, I think it is
        // to make sure that the boundaries of the restricted u are up to date before
        // calling Ax. That being said, at least in one case commenting this line out
        // didn't seem to impact the solution.
        set_from_finer = AddBoundaryExchangeTasks<BoundaryType::gmg_same>(
            set_from_finer, tl, md_comm, multilevel);
        set_from_finer = tl.AddTask(set_from_finer, CopyData<u, u0, true>, md);
        // This should set the rhs only in blocks that correspond to interior nodes, the
        // RHS of leaf blocks that are on this GMG level should have already been set on
        // entry into multigrid
        set_from_finer = eqs_.template Ax<u, temp>(tl, set_from_finer, md);
        set_from_finer = tl.AddTask(
            set_from_finer, AddFieldsAndStoreInteriorSelect<temp, res_err, rhs, true>, md,
            1.0, 1.0, true);
      }
    } else {
      set_from_finer = tl.AddTask(set_from_finer, CopyData<u, u0, true>, md);
    }

    // 2. Do pre-smooth and fill solution on this level
    set_from_finer =
        tl.AddTask(set_from_finer, &equations::template SetDiagonal<D>, &eqs_, md);
    auto pre_smooth = AddSRJIteration<BoundaryType::gmg_same>(
        tl, set_from_finer, pre_stages, multilevel, md, md_comm);
    // If we are finer than the coarsest level:
    auto post_smooth = pre_smooth;
    if (level > min_level) {
      // 3. Communicate same level boundaries so that u is up to date everywhere
      auto comm_u = AddBoundaryExchangeTasks<BoundaryType::gmg_same>(pre_smooth, tl,
                                                                     md_comm, multilevel);

      // 4. Caclulate residual and store in communication field
      auto residual = eqs_.template Ax<u, temp>(tl, comm_u, md);
      residual =
          tl.AddTask(residual, AddFieldsAndStoreInteriorSelect<rhs, temp, res_err, true>,
                     md, 1.0, -1.0, false);

      // 5. Restrict communication field and send to next level
      auto communicate_to_coarse =
          tl.AddTask(residual, SendBoundBufs<BoundaryType::gmg_restrict_send>, md_comm);

      auto coarser = AddMultiGridTasksPartitionLevel(
          tl, communicate_to_coarse, partition, level - 1, min_level, max_level, pmesh);

      // 6. Receive error field into communication field and prolongate
      auto recv_from_coarser = tl.AddTask(
          coarser, ReceiveBoundBufs<BoundaryType::gmg_prolongate_recv>, md_comm);
      auto set_from_coarser = tl.AddTask(
          recv_from_coarser, SetBounds<BoundaryType::gmg_prolongate_recv>, md_comm);
      auto prolongate = tl.AddTask( // TaskQualifier::local_sync, // is this required?
          set_from_coarser, ProlongateBounds<BoundaryType::gmg_prolongate_recv>, md_comm);

      // 7. Correct solution on this level with res_err field and store in
      //    communication field
      auto update_sol =
          tl.AddTask(prolongate, AddFieldsAndStore<u, res_err, u, true>, md, 1.0, 1.0);

      // 8. Post smooth using communication field and stored RHS
      post_smooth = AddSRJIteration<BoundaryType::gmg_same>(tl, update_sol, post_stages,
                                                            multilevel, md, md_comm);
    } else {
      post_smooth = tl.AddTask(pre_smooth, CopyData<u, res_err, true>, md);
    }

    // 9. Send communication field to next finer level (should be error field for that
    // level)
    TaskID last_task = post_smooth;
    if (level < max_level) {
      auto copy_over = post_smooth;
      if (!do_FAS) {
        copy_over = tl.AddTask(post_smooth, CopyData<u, res_err, true>, md);
      } else {
        auto calc_err = tl.AddTask(post_smooth, AddFieldsAndStore<u, u0, res_err, true>,
                                   md, 1.0, -1.0);
        copy_over = calc_err;
      }
      // This is required to make sure boundaries of res_err are up to date before
      // prolongation
      copy_over = tl.AddTask(copy_over, CopyData<u, temp, false>, md);
      copy_over = tl.AddTask(copy_over, CopyData<res_err, u, false>, md);
      auto boundary = AddBoundaryExchangeTasks<BoundaryType::gmg_same>(
          copy_over, tl, md_comm, multilevel);
      auto copy_back = tl.AddTask(boundary, CopyData<u, res_err, true>, md);
      copy_back = tl.AddTask(copy_back, CopyData<temp, u, false>, md);
      last_task =
          tl.AddTask(copy_back, SendBoundBufs<BoundaryType::gmg_prolongate_send>, md);
    }
    // The boundaries are not up to date on return
    return last_task;
  }
};

} // namespace solvers

} // namespace parthenon

#endif // SOLVERS_MG_SOLVER_HPP_
