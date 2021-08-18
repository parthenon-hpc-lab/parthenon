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

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

// Local Includes
#include "bvals/cc/bvals_cc_in_one.hpp"
#include "interface/metadata.hpp"
#include "interface/update.hpp"
#include "mesh/meshblock_pack.hpp"
#include "mesh/refinement_cc_in_one.hpp"
#include "parthenon/driver.hpp"
#include "poisson_cg_driver.hpp"
#include "poisson_cg_package.hpp"
#include "refinement/refinement.hpp"


using namespace parthenon::driver::prelude;

namespace poisson_example {

parthenon::DriverStatus PoissonDriver::Execute() {
  pouts->MakeOutputs(pmesh, pinput);
  ConstructAndExecuteTaskLists<>(this);
  pouts->MakeOutputs(pmesh, pinput);
  return DriverStatus::complete;
}

TaskCollection PoissonDriver::MakeTaskCollection(BlockList_t &blocks) {
  using namespace parthenon::Update;
  TaskCollection tc;
  TaskID none(0);

  for (int i = 0; i < blocks.size(); i++) {
    auto &pmb = blocks[i];
    auto &base = pmb->meshblock_data.Get();
    
    pmb->meshblock_data.Add("delta", base);
  }

  int max_iters = pmesh->packages.Get("poisson_package")->Param<int>("max_iterations");
  int check_interval =
      pmesh->packages.Get("poisson_package")->Param<int>("check_interval");
  bool fail_flag =
      pmesh->packages.Get("poisson_package")->Param<bool>("fail_without_convergence");
  bool warn_flag =
      pmesh->packages.Get("poisson_package")->Param<bool>("warn_without_convergence");

  
  const int num_partitions = pmesh->DefaultNumPartitions();
  TaskRegion &solver_region = tc.AddRegion(num_partitions);

  // setup some reductions
  // initialize to zero
  total_mass.val = 0.0;
  max_rank.val = 0;
  // we'll also demonstrate how to reduce a vector
  vec_reduce.val.resize(10);
  for (int i = 0; i < 10; i++)
    vec_reduce.val[i] = 0;

  std::cout << "max_iters: " << max_iters
            << " checkinterval: " << check_interval
            << " fail flag: " << fail_flag
            << " warn_flag: " << warn_flag
            << " num_partitions: " << num_partitions<<std::endl;
  
  for (int i = 0; i < num_partitions; i++) {
    // make/get a mesh_data container for the state
    auto &md = pmesh->mesh_data.GetOrAdd("base", i);
    auto &mdelta = pmesh->mesh_data.GetOrAdd("delta", i);

    TaskList &tl = solver_region[i];
    int j(0);
    
    //--- Demo a few reductions
    // pass a pointer to the variable being reduced into
    auto loc_red = tl.AddTask(none, poisson_package::SumMass<MeshData<Real>>, md.get(),
                              &total_mass.val);
    // make it a regional dependency so dependent tasks can't execute until all lists do
    // this
    solver_region.AddRegionalDependencies(j, i, loc_red);
    j++;
    

    auto rank_red = tl.AddTask(
        none,
        [](int *max_rank) {
          *max_rank = std::max(*max_rank, parthenon::Globals::my_rank);
          return TaskStatus::complete;
        },
        &max_rank.val);
    solver_region.AddRegionalDependencies(j, i, rank_red);
    j++;
    
    // start a non-blocking MPI_Iallreduce
    auto start_global_reduce =
        (i == 0 ? tl.AddTask(loc_red, &AllReduce<Real>::StartReduce, &total_mass, MPI_SUM)
                : none);

    auto start_rank_reduce =
        (i == 0 ? tl.AddTask(rank_red, &Reduce<int>::StartReduce, &max_rank, 0, MPI_MAX)
                : none);

    // test the reduction until it completes
    auto finish_global_reduce =
        tl.AddTask(start_global_reduce, &AllReduce<Real>::CheckReduce, &total_mass);
    solver_region.AddRegionalDependencies(j, i, finish_global_reduce);
    j++;
    
    auto finish_rank_reduce =
        tl.AddTask(start_rank_reduce, &Reduce<int>::CheckReduce, &max_rank);
    solver_region.AddRegionalDependencies(j, i, finish_rank_reduce);
    j++;
    
    // notice how we must always pass a pointer to the reduction value
    // since tasks capture args by value, this would print zero if we just passed in
    // the val since the tasks that compute the value haven't actually executed yet
    auto report_mass = (i == 0 && parthenon::Globals::my_rank == 0
                            ? tl.AddTask(
                                  finish_global_reduce,
                                  [](Real *mass) {
                                    std::cout << "Total mass = " << *mass << std::endl;
                                    return TaskStatus::complete;
                                  },
                                  &total_mass.val)
                            : none);
    auto report_rank = (i == 0 && parthenon::Globals::my_rank == 0
                            ? tl.AddTask(
                                  finish_rank_reduce,
                                  [](int *max_rank) {
                                    std::cout << "Max rank = " << *max_rank << std::endl;
                                    return TaskStatus::complete;
                                  },
                                  &max_rank.val)
                            : none);
    

    auto setrhs = tl.AddTask(none, poisson_package::SetRHS<MeshData<Real>>, md.get());
    
//////////////////////////////////////////////////////
    
    auto &solver = tl.AddIteration("poisson solver");
    solver.SetMaxIterations(max_iters);
    solver.SetCheckInterval(check_interval);
    solver.SetFailWithMaxIterations(fail_flag);
    solver.SetWarnWithMaxIterations(warn_flag);

    auto mat_elem =
      tl.AddTask(none, poisson_package::SetMatrixElements<MeshData<Real>>, md.get());

    //parthenon::solvers::CG_Solver cgsol;
    auto begin = none;//setrhs|mat_elem;
    
    auto beta = cgsol.createCGTaskList(begin,i,j, tc, tl, solver_region, solver, md, mdelta);

#if 0    
    ////////////////////////////////////////////////////////////////////////////////
    //CG
    // this will move to somewhere..
    
    // initialization only happens once.
    auto mat_elem =
        tl.AddTask(none, poisson_package::SetMatrixElements<MeshData<Real>>, md.get());

    
    auto rz0=  tl.AddTask(none,
                          [](Real *val, Real *beta, int *cntr) {
                            *val=0;
                            *beta=0;
                            *cntr=0;
                            return TaskStatus::complete;
                          },
                          &r_dot_z.val, &betak.val, &cg_cntr);
    // x=0;
    // b = dV*rho
    // r=b-Ax;
    // z = Minv*r;
    auto res0 =
      tl.AddTask( rz0 | mat_elem, poisson_package::DiagScaling<MeshData<Real>>, md.get(), mdelta.get(), &r_dot_z.val);

    // r.z;
    auto start_global_rz =
      (i == 0 ? tl.AddTask(res0,  &AllReduce<Real>::StartReduce, &r_dot_z, MPI_SUM)
                 : none);

    auto finish_global_rz =
        tl.AddTask(start_global_rz, &AllReduce<Real>::CheckReduce, &r_dot_z);

    //synch.
    solver_region.AddRegionalDependencies(1, i, finish_global_rz);


    /////////////////////////////////////////////
    // Iteration starts here.
    // p = beta*p+z;
    auto axpy1 = solver.AddTask(none, poisson_package::Axpy1<MeshData<Real>>, md.get(), &betak.val);
    
    // matvec Ap = J*p
    auto pAp0=  solver.AddTask(none,
                               [](Real *val) {
                                 std::cout <<"in pap0"<<std::endl;
                                 
                                 *val=0;
                                 return TaskStatus::complete;
                               },
                               &p_dot_ap.val);
    
    auto start_recv = solver.AddTask(pAp0, &MeshData<Real>::StartReceiving, md.get(),
                                     BoundaryCommSubset::all);
    
    //ghost exchange.
    auto send =
        solver.AddTask(pAp0, parthenon::cell_centered_bvars::SendBoundaryBuffers, md);

    auto recv = solver.AddTask(
        start_recv, parthenon::cell_centered_bvars::ReceiveBoundaryBuffers, md);

    auto setb =
        solver.AddTask(recv, parthenon::cell_centered_bvars::SetBoundaries, md);

    auto clear = solver.AddTask(send | setb, &MeshData<Real>::ClearBoundary, md.get(),
                                BoundaryCommSubset::all);


    auto matvec = solver.AddTask(pAp0, poisson_package::MatVec<MeshData<Real>>, md.get(), &p_dot_ap.val);

    // reduce p.Ap
    auto start_global_pAp =
        (i == 0 ? tl.AddTask(matvec, &AllReduce<Real>::StartReduce, &p_dot_ap, MPI_SUM)
                : none);

    auto finish_global_pAp =
        tl.AddTask(start_global_pAp, &AllReduce<Real>::CheckReduce, &p_dot_ap);

    solver_region.AddRegionalDependencies(2, i, finish_global_pAp);

    // alpha = r.z/p.Ap
    auto alpha=  solver.AddTask(finish_global_pAp,
                                [](Real *val,Real *rz, Real *pAp, Real *rznew) {
                                  std::cout <<"in aplha"<<std::endl;
                                  
                                  *val=(*rz)/(*pAp);
                                  *rznew=0;
                                  std::cout <<" alpha: " << *val<<std::endl;
                                  
                                  return TaskStatus::complete;
                                },
                                &alphak.val, &r_dot_z.val, &p_dot_ap.val, &r_dot_z_new.val );
    
    // x = x+alpha*p
    // r = r-alpha*Apk
    // z = M^-1*r
    // r.z-new

    auto double_axpy = solver.AddTask(alpha, poisson_package::DoubleAxpy<MeshData<Real>>, md.get(), mdelta.get(),
                                      &alphak.val, &r_dot_z_new.val);
    
    // reduce p.Ap
    auto start_global_rz_new =
        (i == 0 ? tl.AddTask(double_axpy, &AllReduce<Real>::StartReduce, &r_dot_z_new, MPI_SUM)
                : none);

    auto finish_global_rz_new =
        tl.AddTask(start_global_rz_new, &AllReduce<Real>::CheckReduce, &r_dot_z_new);

    solver_region.AddRegionalDependencies(3, i, finish_global_rz_new);

    // beta= rz_new/rz
    // and check convergence..
    auto beta = solver.SetCompletionTask(finish_global_rz_new,
                                         [](Real *beta, Real *rz_new, Real *rz, Real *res_global, Real *gres0, int* cntr)
                                         {
                                           std::cout <<"in beta "<<std::endl;
                                           
                                           *beta=(*rz_new)/(*rz);
                                           *res_global = sqrt(*rz_new);
                                           
                                           if( *cntr == 0 )
                                             *gres0 = *res_global;

                                           *cntr= *cntr+1;
                                           (*rz) = (*rz_new);
                                           
                                           Real err_tol = 1e-9;//pkg->Param<Real>("error_tolerance");
                                           //Real res0    = 1000;//pkg->Param<Real>("res0");
                                           auto status = (*res_global/(*gres0) < err_tol ?
                                                          TaskStatus::complete : TaskStatus::iterate);
                                           std::cout <<*cntr<<" res: " << *res_global/(*gres0)<< "  " << *res_global
                                           << " err: " << err_tol<<std::endl;
                                           
                                           return status;
                                         },
                                         &betak.val, &r_dot_z_new.val, &r_dot_z.val, &res_global.val, &global_res0, &cg_cntr);

    solver_region.AddRegionalDependencies(4, i, beta);
#endif

    // mark task so that dependent tasks (below) won't execute
    // until all task lists have completed it

    auto print = none;
    if (i == 0) { // only print donce
      print = tl.AddTask(beta, poisson_package::PrintComplete);
    }
    //--- End of tasks related to solving the Poisson eq

    // do a vector reduction (everything below here), just for fun
    // first fill it in
    auto fill_vec = tl.AddTask(
        none,
        [](std::vector<int> *vec) {
          auto &v = *vec;
          for (int n = 0; n < v.size(); n++)
            v[n] += n;
          return TaskStatus::complete;
        },
        &vec_reduce.val);
    solver_region.AddRegionalDependencies(5, i, fill_vec);

    TaskID start_vec_reduce =
        (i == 0
             ? tl.AddTask(fill_vec, &parthenon::AllReduce<std::vector<int>>::StartReduce,
                          &vec_reduce, MPI_SUM)
             : none);
    // test the reduction until it completes
    TaskID finish_vec_reduce =
        tl.AddTask(start_vec_reduce, &parthenon::AllReduce<std::vector<int>>::CheckReduce,
                   &vec_reduce);
    solver_region.AddRegionalDependencies(6, i, finish_vec_reduce);

    auto report_vec =
        (i == 0 && parthenon::Globals::my_rank == 0
             ? tl.AddTask(
                   finish_vec_reduce,
                   [num_partitions](std::vector<int> *vec) {
                     auto &v = *vec;
                     std::cout << "Vec reduction: ";
                     for (int n = 0; n < v.size(); n++) {
                       std::cout << v[n] << " ";
                     }
                     std::cout << std::endl;
                     std::cout << "Should be:     ";
                     for (int n = 0; n < v.size(); n++) {
                       std::cout << n * num_partitions * parthenon::Globals::nranks
                                 << " ";
                     }
                     std::cout << std::endl;
                     return TaskStatus::complete;
                   },
                   &vec_reduce.val)
             : none);
  }
  std::cout <<"end of maketask collection "<<std::endl;
  
  return tc;
}

} // namespace poisson_example
