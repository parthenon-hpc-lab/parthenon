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


  std::string pkg_name = "poisson_package";
  auto error_tol =  pmesh->packages.Get("poisson_package")->Param<Real>("error_tolerance");
  
  auto spm_name = pmesh->packages.Get("poisson_package")->Param<std::string>("spm_name");
  auto sol_name = pmesh->packages.Get("poisson_package")->Param<std::string>("sol_name");
  auto rhs_name = pmesh->packages.Get("poisson_package")->Param<std::string>("rhs_name");
  cgsol.init(pkg_name, error_tol);
  
  
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

    auto begin = none;//setrhs|mat_elem;

    
    // create task list for solver.
    auto beta = cgsol.createCGTaskList(begin,i,j, tc, tl, solver_region, solver, md, mdelta);


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
  std::cout <<"end of maketaskcollection "<<std::endl;
  
  return tc;
}

} // namespace poisson_example
