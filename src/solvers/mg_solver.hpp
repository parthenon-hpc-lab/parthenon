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
#include "solvers/internal_prolongation.hpp"
#include "solvers/solver_base.hpp"
#include "solvers/solver_utils.hpp"
#include "tasks/tasks.hpp"
#include "utils/robust.hpp"
#include "utils/type_list.hpp"

namespace parthenon {

namespace solvers {

struct MGParams {
  int max_iters = 1000;
  Real residual_tolerance = 1.e-12;
  bool do_FAS = true;
  std::string smoother = "SRJ2";
  bool two_by_two_diagonal = false;
  int max_coarsenings = std::numeric_limits<int>::max();
  std::string prolongation = "OldLinear";

  MGParams() = default;
  MGParams(ParameterInput *pin, const std::string &input_block) {
    max_iters = pin->GetOrAddInteger(input_block, "max_iterations", max_iters);
    residual_tolerance =
        pin->GetOrAddReal(input_block, "residual_tolerance", residual_tolerance);
    do_FAS = pin->GetOrAddBoolean(input_block, "do_FAS", do_FAS);
    smoother = pin->GetOrAddString(input_block, "smoother", smoother);
    prolongation = pin->GetOrAddString(input_block, "prolongation", prolongation);
    two_by_two_diagonal =
        pin->GetOrAddBoolean(input_block, "two_by_two_diagonal", two_by_two_diagonal);
    max_coarsenings =
        pin->GetOrAddInteger(input_block, "max_coarsenings", max_coarsenings);
  }
};

// The equations_t class must include a template method
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
template <class equations_t, class prolongator_t = ProlongationBlockInteriorDefault>
class MGSolver : public SolverBase {
  static inline std::size_t id{0};

 public:
  using FieldTL = typename equations_t::IndependentVars;

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
  std::string container_res_err, container_temp, container_u0, container_diag;

  MGSolver(const std::string &container_base, const std::string &container_u,
           const std::string &container_rhs, ParameterInput *pin,
           const std::string &input_block, equations_t eq_in = equations_t())
      : MGSolver(container_base, container_u, container_rhs, MGParams(pin, input_block),
                 eq_in, prolongator_t(pin, input_block)) {}

  MGSolver(const std::string &container_base, const std::string &container_u,
           const std::string &container_rhs, MGParams params_in,
           equations_t eq_in = equations_t(), prolongator_t prol_in = prolongator_t())
      : container_base(container_base), container_u(container_u),
        container_rhs(container_rhs), params_(params_in), iter_counter(0), eqs_(eq_in),
        prolongator_(prol_in) {
    FieldTL::IterateTypes(
        [this](auto t) { this->sol_fields.push_back(decltype(t)::name()); });
    std::string solver_id = "mg" + std::to_string(id++);
    container_res_err = solver_id + "_res_err";
    container_temp = solver_id + "_temp";
    container_u0 = solver_id + "_u0";
    container_diag = solver_id + "_diag";
  }

  TaskID AddTasks(TaskList &tl, TaskID dependence, const int partition, Mesh *pmesh) {
    using namespace utils;
    TaskID none;
    auto [itl, solve_id] = tl.AddSublist(dependence, {1, this->params_.max_iters});
    iter_counter = -1;
    auto update_iter = itl.AddTask(
        TaskQualifier::local_sync | TaskQualifier::once_per_region, none, "print",
        [](int *iter_counter) {
          (*iter_counter)++;
          if (*iter_counter > 1 || Globals::my_rank != 0) return TaskStatus::complete;
          printf("# [0] v-cycle\n# [1] rms-residual\n# [2] rms-error\n");
          return TaskStatus::complete;
        },
        &iter_counter);
    auto mg_finest = AddLinearOperatorTasks(itl, update_iter, partition, pmesh);

    auto partitions = pmesh->GetDefaultBlockPartitions(GridIdentifier::leaf());
    if (partition >= partitions.size())
      PARTHENON_FAIL("Does not work with non-default partitioning.");
    auto &md = pmesh->mesh_data.Add(container_base, partitions[partition]);
    auto &md_u = pmesh->mesh_data.Add(container_u, md, sol_fields);
    auto &md_res_err = pmesh->mesh_data.Add(container_res_err, md, sol_fields);
    auto &md_rhs = pmesh->mesh_data.Add(container_rhs, md, sol_fields);
    auto comm = AddBoundaryExchangeTasks<BoundaryType::any>(mg_finest, itl, md_u,
                                                            pmesh->multilevel);
    auto calc_pointwise_res = eqs_.Ax(itl, comm, md, md_u, md_res_err);
    calc_pointwise_res =
        itl.AddTask(calc_pointwise_res, TF(AddFieldsAndStoreInteriorSelect<FieldTL>),
                    md_rhs, md_res_err, md_res_err, 1.0, -1.0, false);
    auto get_res =
        DotProduct<FieldTL>(calc_pointwise_res, itl, &residual, md_res_err, md_res_err);

    auto check = itl.AddTask(
        TaskQualifier::completion, get_res, "Check residual",
        [partition](MGSolver *solver, Mesh *pmesh) {
          Real rms_res = std::sqrt(solver->residual.val / pmesh->GetTotalCells());
          if (Globals::my_rank == 0 && partition == 0)
            printf("%i %e\n", solver->iter_counter, rms_res);
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
    // We require a local pre- and post-MG sync since multigrid iterations require
    // communication across blocks and partitions on the multigrid levels do not
    // necessarily contain the same blocks as partitions on the leaf grid. This
    // means that without the syncs, leaf partitions can receive messages erroneously
    // receive messages and/or update block data during a MG step.
    auto pre_sync = tl.AddTask(TaskQualifier::local_sync, dependence,
                               []() { return TaskStatus::complete; });
    auto mg = pre_sync;
    for (int level = max_level; level >= min_level; --level) {
      mg = mg | AddMultiGridTasksPartitionLevel(tl, dependence, partition, level,
                                                min_level, max_level, pmesh);
    }
    auto post_sync =
        tl.AddTask(TaskQualifier::local_sync, mg, []() { return TaskStatus::complete; });
    return post_sync;
  }

  TaskID AddSetupTasks(TaskList &tl, TaskID dependence, int partition, Mesh *pmesh) {
    using namespace utils;

    int min_level = std::max(pmesh->GetGMGMaxLevel() - params_.max_coarsenings,
                             pmesh->GetGMGMinLevel());
    int max_level = pmesh->GetGMGMaxLevel();

    auto mg_setup = dependence;
    for (int level = max_level; level >= min_level; --level) {
      mg_setup =
          mg_setup | AddMultiGridSetupPartitionLevel(tl, dependence, partition, level,
                                                     min_level, max_level, pmesh);
    }
    return mg_setup;
  }

  Real GetSquaredResidualSum() const { return residual.val; }
  int GetCurrentIterations() const { return iter_counter; }

 protected:
  MGParams params_;
  int iter_counter;
  AllReduce<Real> residual;
  equations_t eqs_;
  prolongator_t prolongator_;
  std::string container_;

  // These functions apparently have to be public to compile with cuda since
  // they contain device side lambdas
 public:
  TaskStatus Jacobi(std::shared_ptr<MeshData<Real>> &md_rhs,
                    std::shared_ptr<MeshData<Real>> &md_Ax,
                    std::shared_ptr<MeshData<Real>> &md_diag,
                    std::shared_ptr<MeshData<Real>> &md_xold,
                    std::shared_ptr<MeshData<Real>> &md_xnew, double weight) {
    using namespace parthenon;
    const int ndim = md_rhs->GetMeshPointer()->ndim;
    using TE = parthenon::TopologicalElement;
    TE te = TE::CC;
    IndexRange ib = md_rhs->GetBoundsI(IndexDomain::interior, te);
    IndexRange jb = md_rhs->GetBoundsJ(IndexDomain::interior, te);
    IndexRange kb = md_rhs->GetBoundsK(IndexDomain::interior, te);

    int nblocks = md_rhs->NumBlocks();
    std::vector<bool> include_block(nblocks, true);
    if (md_rhs->grid.type == GridType::two_level_composite) {
      int current_level = md_rhs->grid.logical_level;
      for (int b = 0; b < nblocks; ++b) {
        include_block[b] =
            md_rhs->GetBlockData(b)->GetBlockPointer()->loc.level() == current_level;
      }
    }
    static auto desc = parthenon::MakePackDescriptorFromTypeList<FieldTL>(md_rhs.get());
    auto pack_rhs = desc.GetPack(md_rhs.get(), include_block);
    auto pack_Ax = desc.GetPack(md_Ax.get(), include_block);
    auto pack_diag = desc.GetPack(md_diag.get(), include_block);
    auto pack_xold = desc.GetPack(md_xold.get(), include_block);
    auto pack_xnew = desc.GetPack(md_xnew.get(), include_block);
    const int scratch_size = 0;
    const int scratch_level = 0;
    parthenon::par_for_outer(
        DEFAULT_OUTER_LOOP_PATTERN, "Jacobi", DevExecSpace(), scratch_size, scratch_level,
        0, pack_rhs.GetNBlocks() - 1, kb.s, kb.e,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int k) {
          const int nvars = pack_rhs.GetUpperBound(b) - pack_rhs.GetLowerBound(b) + 1;
          for (int c = 0; c < nvars; ++c) {
            Real *Ax = &pack_Ax(b, te, c, k, jb.s, ib.s);
            Real *diag = &pack_diag(b, te, c, k, jb.s, ib.s);
            Real *prhs = &pack_rhs(b, te, c, k, jb.s, ib.s);
            Real *xo = &pack_xold(b, te, c, k, jb.s, ib.s);
            Real *xn = &pack_xnew(b, te, c, k, jb.s, ib.s);
            // Use ptr arithmetic to get the number of points we need to go over
            // (including ghost zones) to get from (k, jb.s, ib.s) to (k, jb.e, ib.e)
            const int npoints = &pack_Ax(b, te, c, k, jb.e, ib.e) - Ax + 1;
            parthenon::par_for_inner(
                DEFAULT_INNER_LOOP_PATTERN, member, 0, npoints - 1, [&](const int idx) {
                  const Real off_diag = Ax[idx] - diag[idx] * xo[idx];
                  const Real val = prhs[idx] - off_diag;
                  xn[idx] =
                      weight * robust::ratio(val, diag[idx]) + (1.0 - weight) * xo[idx];
                });
          }
        });
    return TaskStatus::complete;
  }

  template <parthenon::BoundaryType comm_boundary>
  TaskID AddJacobiIteration(TaskList &tl, TaskID depends_on, bool multilevel, Real omega,
                            int partition, int level,
                            std::shared_ptr<MeshData<Real>> &md_in,
                            std::shared_ptr<MeshData<Real>> &md_out) {
    using namespace utils;
    auto pmesh = md_in->GetMeshPointer();
    auto partitions =
        pmesh->GetDefaultBlockPartitions(GridIdentifier::two_level_composite(level));
    auto &md_base = pmesh->mesh_data.Add(container_base, partitions[partition]);
    auto &md_rhs = pmesh->mesh_data.Add(container_rhs, partitions[partition], sol_fields);
    auto &md_diag = pmesh->mesh_data.Add(container_diag, md_base, sol_fields);

    auto comm =
        AddBoundaryExchangeTasks<comm_boundary>(depends_on, tl, md_in, multilevel);
    auto mat_mult = eqs_.Ax(tl, comm, md_base, md_in, md_out);
    return tl.AddTask(mat_mult, TF(&MGSolver::Jacobi), this, md_rhs, md_out, md_diag,
                      md_in, md_out, omega);
  }

  template <parthenon::BoundaryType comm_boundary, class TL_t>
  TaskID AddSRJIteration(TL_t &tl, TaskID depends_on, int stages, bool multilevel,
                         int partition, int level, Mesh *pmesh) {
    using namespace utils;

    const int ndim = pmesh->ndim;
    auto partitions =
        pmesh->GetDefaultBlockPartitions(GridIdentifier::two_level_composite(level));
    auto &md_base = pmesh->mesh_data.Add(container_base, partitions[partition]);
    auto &md_u = pmesh->mesh_data.Add(container_u, md_base, sol_fields);
    auto &md_temp = pmesh->mesh_data.Add(container_temp, md_base, sol_fields);

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
    depends_on = tl.AddTask(depends_on, TF(CopyData<FieldTL, false>), md_u, md_temp);
    auto jacobi1 = AddJacobiIteration<comm_boundary>(
        tl, depends_on, multilevel, omega[ndim - 1][0], partition, level, md_u, md_temp);
    auto copy1 = tl.AddTask(jacobi1, TF(CopyData<FieldTL, true>), md_temp, md_u);
    if (stages < 2) return copy1;
    auto jacobi2 = AddJacobiIteration<comm_boundary>(
        tl, copy1, multilevel, omega[ndim - 1][1], partition, level, md_u, md_temp);
    auto copy2 = tl.AddTask(jacobi2, TF(CopyData<FieldTL, true>), md_temp, md_u);
    if (stages < 3) return copy2;
    auto jacobi3 = AddJacobiIteration<comm_boundary>(
        tl, copy2, multilevel, omega[ndim - 1][2], partition, level, md_u, md_temp);
    return tl.AddTask(jacobi3, TF(CopyData<FieldTL, true>), md_temp, md_u);
  }

  template <class TL_t>
  TaskID AddMultiGridSetupPartitionLevel(TL_t &tl, TaskID dependence, int partition,
                                         int level, int min_level, int max_level,
                                         Mesh *pmesh) {
    using namespace utils;

    auto partitions =
        pmesh->GetDefaultBlockPartitions(GridIdentifier::two_level_composite(level));
    if (partition >= partitions.size()) return dependence;
    auto &md = pmesh->mesh_data.Add(container_base, partitions[partition]);
    auto &md_diag = pmesh->mesh_data.Add(container_diag, md, sol_fields);

    auto task_out = dependence;
    if (level < max_level) {
      task_out =
          tl.AddTask(task_out, TF(ReceiveBoundBufs<BoundaryType::gmg_restrict_recv>), md);
      task_out = tl.AddTask(task_out, TF(SetBounds<BoundaryType::gmg_restrict_recv>), md);
    }
    task_out = tl.AddTask(task_out, TF(&equations_t::SetDiagonal), &eqs_, md, md_diag);
    // If we are finer than the coarsest level:
    if (level > min_level) {
      task_out =
          tl.AddTask(task_out, TF(SendBoundBufs<BoundaryType::gmg_restrict_send>), md);
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
      PARTHENON_FAIL("Unknown smoother type.");
    }

//    auto decorate_task_name = [partition, level](const std::string &in, auto b) {
//      return std::make_tuple(in + "(p:" + std::to_string(partition) +
//                                 ", l:" + std::to_string(level) + ")",
//                             1, b);
//    };

// #define BTF(...) decorate_task_name(TF(__VA_ARGS__))
#define BTF(...) TF(__VA_ARGS__)
    bool multilevel = (level != min_level);

    auto partitions =
        pmesh->GetDefaultBlockPartitions(GridIdentifier::two_level_composite(level));
    if (partition >= partitions.size()) return dependence;
    auto &md = pmesh->mesh_data.Add(container_base, partitions[partition]);
    auto &md_u = pmesh->mesh_data.Add(container_u, partitions[partition], sol_fields);
    auto &md_rhs = pmesh->mesh_data.Add(container_rhs, partitions[partition], sol_fields);
    auto &md_res_err = pmesh->mesh_data.Add(container_res_err, md, sol_fields);
    auto &md_temp = pmesh->mesh_data.Add(container_temp, md, sol_fields);
    auto &md_u0 = pmesh->mesh_data.Add(container_u0, md, sol_fields);
    auto &md_diag = pmesh->mesh_data.Add(container_diag, md, sol_fields);

    // 0. Receive residual from coarser level if there is one
    auto set_from_finer = dependence;
    if (level < max_level) {
      // Fill fields with restricted values
      // TODO(LFR): Need to make sure that this communication pattern is ok, since we are
      //            trying to concurrently communicate on two stages
      auto recv_from_finer = tl.AddTask(
          dependence, TF(ReceiveBoundBufs<BoundaryType::gmg_restrict_recv>), md_u);
      set_from_finer = tl.AddTask(recv_from_finer,
                                  BTF(SetBounds<BoundaryType::gmg_restrict_recv>), md_u);
      recv_from_finer =
          tl.AddTask(set_from_finer,
                     TF(ReceiveBoundBufs<BoundaryType::gmg_restrict_recv>), md_res_err);
      set_from_finer = tl.AddTask(
          recv_from_finer, BTF(SetBounds<BoundaryType::gmg_restrict_recv>), md_res_err);
      // 1. Copy residual from dual purpose communication field to the rhs, should be
      // actual RHS for finest level
      if (!do_FAS) {
        auto zero_u = tl.AddTask(set_from_finer, BTF(SetToZero<FieldTL, true>), md_u);
        auto copy_rhs =
            tl.AddTask(set_from_finer, BTF(CopyData<FieldTL, true>), md_res_err, md_rhs);
        set_from_finer = zero_u | copy_rhs;
      } else {
        // TODO(LFR): Determine if this boundary exchange task is required, I think it is
        // to make sure that the boundaries of the restricted u are up to date before
        // calling Ax. That being said, at least in one case commenting this line out
        // didn't seem to impact the solution.
        set_from_finer = AddBoundaryExchangeTasks<BoundaryType::gmg_same>(
            set_from_finer, tl, md_u, multilevel);
        set_from_finer =
            tl.AddTask(set_from_finer, BTF(CopyData<FieldTL, true>), md_u, md_u0);
        // This should set the rhs only in blocks that correspond to interior nodes, the
        // RHS of leaf blocks that are on this GMG level should have already been set on
        // entry into multigrid
        set_from_finer = eqs_.Ax(tl, set_from_finer, md, md_u, md_temp);
        set_from_finer = tl.AddTask(set_from_finer,
                                    BTF(AddFieldsAndStoreInteriorSelect<FieldTL, true>),
                                    md_temp, md_res_err, md_rhs, 1.0, 1.0, true);
      }
    } else {
      set_from_finer =
          tl.AddTask(set_from_finer, BTF(CopyData<FieldTL, true>), md_u, md_u0);
    }

    // 2. Do pre-smooth and fill solution on this level
    auto pre_smooth = AddSRJIteration<BoundaryType::gmg_same>(
        tl, set_from_finer, pre_stages, multilevel, partition, level, pmesh);
    // If we are finer than the coarsest level:
    auto post_smooth = pre_smooth;
    if (level > min_level) {
      // 3. Communicate same level boundaries so that u is up to date everywhere
      auto comm_u = AddBoundaryExchangeTasks<BoundaryType::gmg_same>(pre_smooth, tl, md_u,
                                                                     multilevel);

      // 4. Caclulate residual and store in communication field
      auto residual = eqs_.Ax(tl, comm_u, md, md_u, md_temp);
      residual = tl.AddTask(residual, BTF(AddFieldsAndStoreInteriorSelect<FieldTL, true>),
                            md_rhs, md_temp, md_res_err, 1.0, -1.0, false);

      // 5. Restrict communication field and send to next level
      // TODO(LFR): Other place where we are receiving two stage communication
      auto communicate_to_coarse =
          tl.AddTask(residual, BTF(SendBoundBufs<BoundaryType::gmg_restrict_send>), md_u);
      communicate_to_coarse =
          tl.AddTask(communicate_to_coarse,
                     BTF(SendBoundBufs<BoundaryType::gmg_restrict_send>), md_res_err);

      // 6. Receive error field into communication field and prolongate
      auto recv_from_coarser =
          tl.AddTask(communicate_to_coarse,
                     TF(ReceiveBoundBufs<BoundaryType::gmg_prolongate_recv>), md_res_err);
      auto set_from_coarser =
          tl.AddTask(recv_from_coarser, BTF(SetBounds<BoundaryType::gmg_prolongate_recv>),
                     md_res_err);
      auto prolongate =
          prolongator_.template Prolongate<FieldTL>(tl, set_from_coarser, md_res_err);

      // 7. Correct solution on this level with res_err field and store in
      //    communication field
      auto update_sol = tl.AddTask(prolongate, BTF(AddFieldsAndStore<FieldTL, true>),
                                   md_u, md_res_err, md_u, 1.0, 1.0);

      // 8. Post smooth using communication field and stored RHS
      post_smooth = AddSRJIteration<BoundaryType::gmg_same>(
          tl, update_sol, post_stages, multilevel, partition, level, pmesh);

    } else {
      post_smooth =
          tl.AddTask(pre_smooth, BTF(CopyData<FieldTL, true>), md_u, md_res_err);
    }

    // 9. Send communication field to next finer level (should be error field for that
    // level)
    TaskID last_task = post_smooth;
    if (level < max_level) {
      auto copy_over = post_smooth;
      if (!do_FAS) {
        copy_over =
            tl.AddTask(post_smooth, BTF(CopyData<FieldTL, true>), md_u, md_res_err);
      } else {
        auto calc_err = tl.AddTask(post_smooth, BTF(AddFieldsAndStore<FieldTL, true>),
                                   md_u, md_u0, md_res_err, 1.0, -1.0);
        copy_over = calc_err;
      }
      // This is required to make sure boundaries of res_err are up to date before
      // prolongation
      auto boundary = AddBoundaryExchangeTasks<BoundaryType::gmg_same>(
          copy_over, tl, md_res_err, multilevel);
      last_task = tl.AddTask(
          boundary, BTF(SendBoundBufs<BoundaryType::gmg_prolongate_send>), md_res_err);
    }
    // The boundaries are not up to date on return
    return last_task;
#undef BTF
  }
};

} // namespace solvers

} // namespace parthenon

#endif // SOLVERS_MG_SOLVER_HPP_
