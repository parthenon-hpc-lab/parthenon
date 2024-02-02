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
#ifndef SOLVERS_MG_SOLVER_HPP_
#define SOLVERS_MG_SOLVER_HPP_

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

  MGSolver(StateDescriptor *pkg, MGParams params_in, equations eq_in = equations(),
           std::vector<int> shape = {})
      : params_(params_in), iter_counter(0), eqs_(eq_in) {
    using namespace parthenon::refinement_ops;
    auto mres_err =
        Metadata({Metadata::Cell, Metadata::Independent, Metadata::FillGhost,
                  Metadata::GMGRestrict, Metadata::GMGProlongate, Metadata::OneCopy},
                 shape);
    mres_err.RegisterRefinementOps<ProlongateSharedLinear, RestrictAverage>();
    pkg->AddField(res_err::name(), mres_err);

    auto mtemp = Metadata({Metadata::Cell, Metadata::Independent, Metadata::FillGhost,
                           Metadata::WithFluxes, Metadata::OneCopy},
                          shape);
    mtemp.RegisterRefinementOps<ProlongateSharedLinear, RestrictAverage>();
    pkg->AddField(temp::name(), mtemp);

    auto mu0 = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy}, shape);
    pkg->AddField(u0::name(), mu0);
    pkg->AddField(D::name(), mu0);
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
    auto calc_pointwise_res = eqs_.template Ax<u, res_err>(itl, mg_finest, md);
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

    int min_level = 0;
    int max_level = pmesh->GetGMGMaxLevel();

    return AddMultiGridTasksPartitionLevel(tl, dependence, partition, max_level,
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
  enum class GSType { all, red, black };

  template <class rhs_t, class Axold_t, class D_t, class xold_t, class xnew_t>
  static TaskStatus Jacobi(std::shared_ptr<MeshData<Real>> &md, double weight,
                           GSType gs_type = GSType::all) {
    using namespace parthenon;
    const int ndim = md->GetMeshPointer()->ndim;
    using TE = parthenon::TopologicalElement;
    TE te = TE::CC;
    IndexRange ib = md->GetBoundsI(IndexDomain::interior, te);
    IndexRange jb = md->GetBoundsJ(IndexDomain::interior, te);
    IndexRange kb = md->GetBoundsK(IndexDomain::interior, te);

    int nblocks = md->NumBlocks();
    std::vector<bool> include_block(nblocks, true);

    auto desc =
        parthenon::MakePackDescriptor<xold_t, xnew_t, Axold_t, rhs_t, D_t>(md.get());
    auto pack = desc.GetPack(md.get(), include_block);
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "CaclulateFluxes", DevExecSpace(), 0, pack.GetNBlocks() - 1,
        kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          const auto &coords = pack.GetCoordinates(b);
          if ((i + j + k) % 2 == 1 && gs_type == GSType::red) return;
          if ((i + j + k) % 2 == 0 && gs_type == GSType::black) return;

          const int nvars =
              pack.GetUpperBound(b, D_t()) - pack.GetLowerBound(b, D_t()) + 1;
          for (int c = 0; c < nvars; ++c) {
            Real diag_elem = pack(b, te, D_t(c), k, j, i);

            // Get the off-diagonal contribution to Ax = (D + L + U)x = y
            Real off_diag = pack(b, te, Axold_t(c), k, j, i) -
                            diag_elem * pack(b, te, xold_t(c), k, j, i);

            Real val = pack(b, te, rhs_t(c), k, j, i) - off_diag;
            pack(b, te, xnew_t(c), k, j, i) =
                weight * val / diag_elem +
                (1.0 - weight) * pack(b, te, xold_t(c), k, j, i);
          }
        });
    return TaskStatus::complete;
  }

  template <parthenon::BoundaryType comm_boundary, class in_t, class out_t, class TL_t>
  TaskID AddJacobiIteration(TL_t &tl, TaskID depends_on, bool multilevel, Real omega,
                            std::shared_ptr<MeshData<Real>> &md) {
    using namespace utils;

    auto comm = AddBoundaryExchangeTasks<comm_boundary>(depends_on, tl, md, multilevel);
    auto mat_mult = eqs_.template Ax<in_t, out_t>(tl, comm, md);
    return tl.AddTask(mat_mult, Jacobi<rhs, out_t, D, in_t, out_t>, md, omega,
                      GSType::all);
  }

  template <parthenon::BoundaryType comm_boundary, class TL_t>
  TaskID AddSRJIteration(TL_t &tl, TaskID depends_on, int stages, bool multilevel,
                         std::shared_ptr<MeshData<Real>> &md) {
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
    auto jacobi1 = AddJacobiIteration<comm_boundary, u, temp>(tl, depends_on, multilevel,
                                                              omega[ndim - 1][0], md);
    if (stages < 2) {
      return tl.AddTask(jacobi1, CopyData<temp, u, true>, md);
    }
    auto jacobi2 = AddJacobiIteration<comm_boundary, temp, u>(tl, jacobi1, multilevel,
                                                              omega[ndim - 1][1], md);
    if (stages < 3) return jacobi2;
    auto jacobi3 = AddJacobiIteration<comm_boundary, u, temp>(tl, jacobi2, multilevel,
                                                              omega[ndim - 1][2], md);
    return tl.AddTask(jacobi3, CopyData<temp, u, true>, md);
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

    // 0. Receive residual from coarser level if there is one
    auto set_from_finer = dependence;
    if (level < max_level) {
      // Fill fields with restricted values
      auto recv_from_finer =
          tl.AddTask(dependence, ReceiveBoundBufs<BoundaryType::gmg_restrict_recv>, md);
      set_from_finer = tl.AddTask( // TaskQualifier::local_sync, // is this required?
          recv_from_finer, SetBounds<BoundaryType::gmg_restrict_recv>, md);
      // 1. Copy residual from dual purpose communication field to the rhs, should be
      // actual RHS for finest level
      auto copy_u = tl.AddTask(set_from_finer, CopyData<u, u0, true>, md);
      if (!do_FAS) {
        auto zero_u = tl.AddTask(copy_u, SetToZero<u, true>, md);
        auto copy_rhs = tl.AddTask(set_from_finer, CopyData<res_err, rhs, true>, md);
        set_from_finer = zero_u | copy_u | copy_rhs;
      } else {
        set_from_finer = AddBoundaryExchangeTasks<BoundaryType::gmg_same>(
            set_from_finer, tl, md, multilevel);
        // This should set the rhs only in blocks that correspond to interior nodes, the
        // RHS of leaf blocks that are on this GMG level should have already been set on
        // entry into multigrid
        set_from_finer = eqs_.template Ax<u, temp>(tl, set_from_finer, md);
        set_from_finer = tl.AddTask(
            set_from_finer, AddFieldsAndStoreInteriorSelect<temp, res_err, rhs, true>, md,
            1.0, 1.0, true);
        set_from_finer = set_from_finer | copy_u;
      }
    } else {
      set_from_finer = tl.AddTask(set_from_finer, CopyData<u, u0, true>, md);
    }

    // 2. Do pre-smooth and fill solution on this level
    set_from_finer =
        tl.AddTask(set_from_finer, &equations::template SetDiagonal<D>, &eqs_, md);
    auto pre_smooth = AddSRJIteration<BoundaryType::gmg_same>(tl, set_from_finer,
                                                              pre_stages, multilevel, md);
    // If we are finer than the coarsest level:
    auto post_smooth = pre_smooth;
    if (level > min_level) {
      // 3. Communicate same level boundaries so that u is up to date everywhere
      auto comm_u = AddBoundaryExchangeTasks<BoundaryType::gmg_same>(pre_smooth, tl, md,
                                                                     multilevel);

      // 4. Caclulate residual and store in communication field
      auto residual = eqs_.template Ax<u, temp>(tl, comm_u, md);
      residual =
          tl.AddTask(residual, AddFieldsAndStoreInteriorSelect<rhs, temp, res_err, true>,
                     md, 1.0, -1.0, false);

      // 5. Restrict communication field and send to next level
      auto communicate_to_coarse =
          tl.AddTask(residual, SendBoundBufs<BoundaryType::gmg_restrict_send>, md);

      auto coarser = AddMultiGridTasksPartitionLevel(
          tl, communicate_to_coarse, partition, level - 1, min_level, max_level, pmesh);

      // 6. Receive error field into communication field and prolongate
      auto recv_from_coarser =
          tl.AddTask(coarser, ReceiveBoundBufs<BoundaryType::gmg_prolongate_recv>, md);
      auto set_from_coarser =
          tl.AddTask(recv_from_coarser, SetBounds<BoundaryType::gmg_prolongate_recv>, md);
      auto prolongate = tl.AddTask( // TaskQualifier::local_sync, // is this required?
          set_from_coarser, ProlongateBounds<BoundaryType::gmg_prolongate_recv>, md);

      // 7. Correct solution on this level with res_err field and store in
      //    communication field
      auto update_sol =
          tl.AddTask(prolongate, AddFieldsAndStore<u, res_err, u, true>, md, 1.0, 1.0);

      // 8. Post smooth using communication field and stored RHS
      post_smooth = AddSRJIteration<BoundaryType::gmg_same>(tl, update_sol, post_stages,
                                                            multilevel, md);
    } else {
      post_smooth = tl.AddTask(pre_smooth, CopyData<u, res_err, true>, md);
    }

    // 9. Send communication field to next finer level (should be error field for that
    // level)
    TaskID last_task;
    if (level < max_level) {
      auto copy_over = post_smooth;
      if (!do_FAS) {
        copy_over = tl.AddTask(post_smooth, CopyData<u, res_err, true>, md);
      } else {
        auto calc_err = tl.AddTask(post_smooth, AddFieldsAndStore<u, u0, res_err, true>,
                                   md, 1.0, -1.0);
        copy_over = calc_err;
      }
      auto boundary =
          AddBoundaryExchangeTasks<BoundaryType::gmg_same>(copy_over, tl, md, multilevel);
      last_task =
          tl.AddTask(boundary, SendBoundBufs<BoundaryType::gmg_prolongate_send>, md);
    } else {
      last_task = AddBoundaryExchangeTasks<BoundaryType::gmg_same>(post_smooth, tl, md,
                                                                   multilevel);
    }
    return last_task;
  }
};

} // namespace solvers

} // namespace parthenon

#endif // SOLVERS_MG_SOLVER_HPP_
