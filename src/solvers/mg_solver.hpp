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

struct MGParams { 
  int max_iters = 10; 
  Real residual_tolerance = 1.e-12;  
  bool do_FAS = true;
  std::string smoother = "SRJ2";
}; 

#define MGVARIABLE(base, varname)                                                        \
  struct varname : public parthenon::variable_names::base_t<false> {                     \
    template <class... Ts>                                                               \
    KOKKOS_INLINE_FUNCTION varname(Ts &&...args)                                         \
        : parthenon::variable_names::base_t<false>(std::forward<Ts>(args)...) {}         \
    static std::string name() { return base::name() + "." #varname; }                    \
  }

template <class u, class rhs> 
class MGSolver {
  MGVARIABLE(u, res_err); // residual on the way up and error on the way down
  MGVARIABLE(u, temp); // Temporary storage 
  MGVARIABLE(u, u0); // Storage for initial solution during FAS
  MGVARIABLE(u, D); // Storage for (approximate) diagonal
 public:
  MGSolver(StateDescriptor *pkg, MGParams params_in) : params_(params_in), iter_counter(0) { 
    using namespace parthenon::refinement_ops;
    auto mres_err = Metadata({Metadata::Cell, Metadata::Independent, Metadata::FillGhost,
                              Metadata::GMGRestrict, Metadata::GMGProlongate, Metadata::OneCopy});
    mres_err.RegisterRefinementOps<ProlongateSharedLinear, RestrictAverage>();
    pkg->AddField(res_err::name(), mres_err);

    auto mtemp = Metadata(
      {Metadata::Cell, Metadata::Independent, Metadata::FillGhost, Metadata::WithFluxes, Metadata::OneCopy});
    mtemp.RegisterRefinementOps<ProlongateSharedLinear, RestrictAverage>();
    pkg->AddField(temp::name(), mtemp);

    auto mu0 = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy});
    pkg->AddField(u0::name(), mu0);
    pkg->AddField(D::name(), mu0);
  }

  TaskID AddTasks(TaskList &tl, TaskID dependence, int partition, Mesh *pmesh, TaskRegion& region, int reg_dep_id) { 
    TaskID none(0);
    auto iter_tl = tl.AddIteration("MG." + u::name());
    using namespace impl;
    iter_counter = 0;

    int min_level = 0;
    int max_level = pmesh->GetGMGMaxLevel();
    
    auto &md = pmesh->mesh_data.GetOrAdd("base", partition);
    if (partition == 0) {
      tl.AddTask(
          dependence,
          [&]() {
            printf("# [0] v-cycle\n# [1] rms-residual\n# [2] rms-error\n");
            return TaskStatus::complete;
          });
    } 

    for (int level = max_level - 1; level >= min_level; --level)
      AddMultiGridTasksPartitionLevel(iter_tl, none, partition, level, min_level, max_level, level == min_level, pmesh); 
    auto mg_finest = AddMultiGridTasksPartitionLevel(iter_tl, dependence, partition, max_level, min_level, max_level, false, pmesh);
    
    auto calc_pointwise_res = Axpy<u, rhs, res_err>(iter_tl, mg_finest, md, -1.0, 1.0, false);
    auto get_res = DotProduct<res_err, res_err>(calc_pointwise_res, region, iter_tl, partition,
                                                reg_dep_id, &residual, md);

    auto check = iter_tl.SetCompletionTask(get_res, [](MGSolver *solver, int part, Mesh *pmesh){
      if (part != 0) TaskStatus::complete; 
      solver->iter_counter++;
      Real rms_res = std::sqrt(solver->residual.val / pmesh->GetTotalCells());
      printf("%i %e (%i)\n", solver->iter_counter, rms_res, pmesh->GetTotalCells()); 
      if (rms_res > solver->params_.residual_tolerance && solver->iter_counter < solver->params_.max_iters) return TaskStatus::iterate;
      return TaskStatus::complete;
      }, this, partition, pmesh);
    region.AddGlobalDependencies(reg_dep_id, partition, check);

    return check;
  }

 protected: 
  MGParams params_;
  int iter_counter;
  AllReduce<Real> residual;

  template <parthenon::BoundaryType comm_boundary, class in_t, class out_t, class TL_t>
  TaskID AddJacobiIteration(TL_t &tl, TaskID depends_on, bool multilevel, Real omega,
                            std::shared_ptr<MeshData<Real>> &md) {
    using namespace impl;
    TaskID none(0);
  
    auto comm = AddBoundaryExchangeTasks<comm_boundary>(depends_on, tl, md, multilevel);
    auto flux = tl.AddTask(comm, CalculateFluxes<in_t, true>, md);
    auto mat_mult = tl.AddTask(flux, FluxMultiplyMatrix<in_t, out_t, true>, md, false);
    auto diag = tl.AddTask(flux, SetDiagonal<D>, md);
    return tl.AddTask(mat_mult | diag, Jacobi<rhs, out_t, D, in_t, out_t, true>, md, omega,
                      GSType::all);
  }
  
  template <parthenon::BoundaryType comm_boundary, class TL_t>
  TaskID AddSRJIteration(TL_t &tl, TaskID depends_on, int stages, bool multilevel,
                         std::shared_ptr<MeshData<Real>> &md) {
    using namespace impl;
    int ndim = md->GetParentPointer()->ndim;
  
    std::array<std::array<Real, 3>, 3> omega_M1{
        {{1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}}};
    // Damping factors from Yang & Mittal (2017)
    std::array<std::array<Real, 3>, 3> omega_M2{
        {{0.8723, 0.5395, 0.0000}, {1.3895, 0.5617, 0.0000}, {1.7319, 0.5695, 0.0000}}};
    std::array<std::array<Real, 3>, 3> omega_M3{
        {{0.9372, 0.6667, 0.5173}, {1.6653, 0.8000, 0.5264}, {2.2473, 0.8571, 0.5296}}};
  
    auto omega = omega_M1;
    if (stages == 2) omega = omega_M2;
    if (stages == 3) omega = omega_M3;
    // This copy is to set the boundaries of temp that will not be updated by boundary
    // communication to the values in u
    depends_on = tl.AddTask(depends_on, CopyData<u, temp>, md);
    auto jacobi1 = AddJacobiIteration<comm_boundary, u, temp>(tl, depends_on, multilevel,
                                                              omega[ndim - 1][0], md);
    if (stages < 2) {
      return tl.AddTask(jacobi1, CopyData<temp, u>, md);
    }
    auto jacobi2 = AddJacobiIteration<comm_boundary, temp, u>(tl, jacobi1, multilevel,
                                                              omega[ndim - 1][1], md);
    if (stages < 3) return jacobi2;
    auto jacobi3 = AddJacobiIteration<comm_boundary, u, temp>(tl, jacobi2, multilevel,
                                                              omega[ndim - 1][2], md);
    return tl.AddTask(jacobi3, CopyData<temp, u>, md);
  }
  
  template <class x_t, class y_t, class out_t, bool only_md_level = false, class TL_t>
  TaskID Axpy(TL_t &tl, TaskID depends_on, std::shared_ptr<MeshData<Real>> &md,
              Real weight_Ax, Real weight_y, bool only_interior, bool do_flux_cor = false) {
    using namespace impl;
    auto flux_res = tl.AddTask(depends_on, CalculateFluxes<x_t, only_md_level>, md);
    if (do_flux_cor && !only_md_level) {
      auto start_flxcor = tl.AddTask(flux_res, StartReceiveFluxCorrections, md);
      auto send_flxcor = tl.AddTask(flux_res, LoadAndSendFluxCorrections, md);
      auto recv_flxcor = tl.AddTask(send_flxcor, ReceiveFluxCorrections, md);
      flux_res = tl.AddTask(recv_flxcor, SetFluxCorrections, md);
    }
    auto Ax_res = tl.AddTask(flux_res, FluxMultiplyMatrix<x_t, temp, only_md_level>, md,
                             only_interior);
    return tl.AddTask(Ax_res,
                      AddFieldsAndStoreInteriorSelect<temp, y_t, out_t, only_md_level>, md,
                      weight_Ax, weight_y, only_interior);
  }
  
  template <class a_t, class b_t, class TL_t>
  TaskID DotProduct(TaskID dependency_in, TaskRegion &region, TL_t &tl, int partition,
                    int &reg_dep_id, AllReduce<Real> *adotb,
                    std::shared_ptr<MeshData<Real>> &md) {
    using namespace impl;
    auto zero_adotb = (partition == 0 ? tl.AddTask(
                                            dependency_in,
                                            [](AllReduce<Real> *r) {
                                              r->val = 0.0;
                                              return TaskStatus::complete;
                                            },
                                            adotb)
                                      : dependency_in);
    region.AddRegionalDependencies(reg_dep_id, partition, zero_adotb);
    reg_dep_id++;
    auto get_adotb = tl.AddTask(zero_adotb, DotProductLocal<a_t, b_t>, md, &(adotb->val));
    region.AddRegionalDependencies(reg_dep_id, partition, get_adotb);
    reg_dep_id++;
    auto start_global_adotb =
        (partition == 0
             ? tl.AddTask(get_adotb, &AllReduce<Real>::StartReduce, adotb, MPI_SUM)
             : get_adotb);
    auto finish_global_adotb =
        tl.AddTask(start_global_adotb, &AllReduce<Real>::CheckReduce, adotb);
    region.AddRegionalDependencies(reg_dep_id, partition, finish_global_adotb);
    reg_dep_id++;
    return finish_global_adotb;
  }
  
  template <class TL_t>
  TaskID AddMultiGridTasksPartitionLevel(TL_t &tl, TaskID dependence, int partition, int level, int min_level,
                                             int max_level, bool final, Mesh *pmesh) {
    using namespace impl;  
    auto smoother = params_.smoother;
    bool do_FAS = params_.do_FAS;
    int pre_stages, post_stages;
    if (smoother == "SRJ1") {
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
  
    int ndim = pmesh->ndim;
    bool multilevel = (level != min_level);
    TaskID last_task;
    
    auto &md = pmesh->gmg_mesh_data[level].GetOrAdd(level, "base", partition);
    // 0. Receive residual from coarser level if there is one
    auto set_from_finer = dependence;
    if (level < max_level) {
      // Fill fields with restricted values
      auto recv_from_finer =
          tl.AddTask(dependence, ReceiveBoundBufs<BoundaryType::gmg_restrict_recv>, md);
      set_from_finer =
          tl.AddTask(recv_from_finer, SetBounds<BoundaryType::gmg_restrict_recv>, md);
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
        set_from_finer =
            Axpy<u, res_err, rhs, true>(tl, set_from_finer, md, 1.0, 1.0, true);
        set_from_finer = set_from_finer | copy_u;
      }
    } else {
      set_from_finer = tl.AddTask(set_from_finer, CopyData<u, u0, true>, md);
    }
  
    // 2. Do pre-smooth and fill solution on this level
    auto pre_smooth = AddSRJIteration<BoundaryType::gmg_same>(tl, set_from_finer,
                                                              pre_stages, multilevel, md);
    // If we are finer than the coarsest level:
    auto post_smooth = pre_smooth;
    if (level > min_level) {
      // 3. Communicate same level boundaries so that u is up to date everywhere
      auto comm_u = AddBoundaryExchangeTasks<BoundaryType::gmg_same>(pre_smooth, tl, md,
                                                                     multilevel);
  
      // 4. Caclulate residual and store in communication field
      auto residual = Axpy<u, rhs, res_err, true>(tl, comm_u, md, -1.0, 1.0, false);
  
      // 5. Restrict communication field and send to next level
      auto communicate_to_coarse =
          tl.AddTask(residual, SendBoundBufs<BoundaryType::gmg_restrict_send>, md);
  
      // 6. Receive error field into communication field and prolongate
      auto recv_from_coarser = tl.AddTask(
          communicate_to_coarse, ReceiveBoundBufs<BoundaryType::gmg_prolongate_recv>, md);
      auto set_from_coarser =
          tl.AddTask(recv_from_coarser, SetBounds<BoundaryType::gmg_prolongate_recv>, md);
      auto prolongate = tl.AddTask(
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
      last_task = tl.AddTask(boundary, SendBoundBufs<BoundaryType::gmg_prolongate_send>, md);
    } else {
      last_task = AddBoundaryExchangeTasks<BoundaryType::gmg_same>(post_smooth, tl, md, multilevel);
    }
    return last_task;
  }

};

} // namespace solvers

} // namespace parthenon

#endif // SOLVERS_MG_SOLVER_HPP_
