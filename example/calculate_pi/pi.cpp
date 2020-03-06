//========================================================================================
// (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
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


#include <iostream>
#ifdef MPI_PARALLEL
#include <mpi.h>
#endif
#include "pi.hpp"

namespace parthenon {

// can be used to set global properties that all meshblocks want to know about
void ProcessProperties(std::vector<std::shared_ptr<PropertiesInterface>>& properties, ParameterInput *pin) {}

void InitializePhysics(std::map<std::string, std::shared_ptr<StateDescriptor>>& physics, ParameterInput *pin) {
  // only have one package for this app, but will typically have more things added to 
  physics["PiCalculator"] = PiCalculator::Initialize(pin);
}

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  // nothing to do here for this app
}

DriverStatus CalculatePi::Execute() {
  // this is where the main work is orchestrated
  // No evolution in this driver.  Just calculates something once.
  // For evolution, look at the EvolutionDriver

  int nmb = pmesh->GetNumMeshBlocksThisRank(Globals::my_rank);
  std::vector<TaskList> task_lists;
  task_lists.resize(nmb);

  int i=0;
  MeshBlock *pmb = pmesh->pblock;
  while (pmb != nullptr) {
    task_lists[i] = MakeTaskList(pmb);
    i++;
    pmb = pmb->next;
  }
  int complete_cnt = 0;
  while (complete_cnt != nmb) {
    for (auto & tl : task_lists) {
      if (!tl.IsComplete()) {
          auto status = tl.DoAvailable();
          if (status == TaskListStatus::complete) {
            complete_cnt++;
          }
      }
    }
  }

  // All the blocks are done, now do a global reduce and spit out the answer
  // first sum over blocks on this rank
  Real area = 0.0;
  pmb = pmesh->pblock;
  while (pmb != nullptr) {
    Variable<Real>& v = pmb->real_container.Get("in_or_out");
    Real block_area = v(0,0,0);
    area += block_area;
    pmb = pmb->next;
  }
#ifdef MPI_PARALLEL
  Real pi_val;
  MPI_Reduce(&area, &pi_val, 1, MPI_ATHENA_REAL, MPI_SUM, 0, MPI_COMM_WORLD);
#else
  Real pi_val = area;
 #endif
  if(Globals::my_rank == 0) {
    std::cout << std::endl << std::endl << "PI = " << pi_val << "    rel error = " << (pi_val-M_PI)/M_PI << std::endl << std::endl;
  }
  pmesh->mbcnt = pmesh->nbtotal; // this is how many blocks were processed
  return DriverStatus::complete;
}

TaskList CalculatePi::MakeTaskList(MeshBlock *pmb) {
  // make a task list for this mesh block
  using namespace PiCalculator;
  TaskList tl;

  TaskID none(0);
  auto get_area = tl.AddTask(ComputeArea, none, pmb);

  // could add more tasks like:
  // auto next_task = tl.AddTask(FuncPtr, get_area, pmb);
  // for a task that executes the function FuncPtr (with argument MeshBlock *pmb) that depends on task get_area
  return std::move(tl);
}

// This defines a "physics" package
namespace PiCalculator {

  void SetInOrOut(Container<Real>& rc) {
    MeshBlock *pmb = rc.pmy_block;
    Coordinates *pcoord = pmb->pcoord.get();
    Variable<Real>& v = rc.Get("in_or_out");
    const auto& radius = pmb->physics["PiCalculator"]->Param<Real>("radius");
    for (int k=0; k<pmb->ncells3; k++) {
      for (int j=0; j<pmb->ncells2; j++) {
        for (int i=0; i<pmb->ncells1; i++) {
          Real rsq = std::pow(pcoord->x1v(i),2) + std::pow(pcoord->x2v(j),2);
          if (rsq < radius*radius) {
            v(k,j,i) = 1.0;
          } else {
            v(k,j,i) = 0.0;
          }
        }
      }
    }
  }

  int CheckRefinement(Container<Real>& rc) {
    MeshBlock *pmb = rc.pmy_block;
    Variable<Real>& v = rc.Get("in_or_out");
    int delta_level = -1;
    Real vmin = 1.0;
    Real vmax = 0.0;
    for (int k=0; k<pmb->ncells3; k++) {
      for (int j=0; j<pmb->ncells2; j++) {
        for (int i=0; i<pmb->ncells1; i++) {
          vmin = (v(k,j,i) < vmin ? v(k,j,i) : vmin);
          vmax = (v(k,j,i) > vmax ? v(k,j,i) : vmax);
        }
      }
    }
    if (vmax > 0.95 && vmin < 0.05) {
      delta_level = 1;
    }
    return delta_level;
  }

  std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
    auto package = std::make_shared<StateDescriptor>("PiCalculator");
    Params& params = package->AllParams();

    Real radius = pin->GetOrAddReal("Pi", "radius", 1.0);
    params.Add("radius", radius);

    std::string field_name("in_or_out");
    Metadata m({Metadata::cell, Metadata::derived, Metadata::graphics});
    package->AddField(field_name, m, DerivedOwnership::unique);

    package->FillDerived = SetInOrOut;
    package->CheckRefinement = CheckRefinement;
    return package;
  }

  TaskStatus ComputeArea(MeshBlock *pmb) {
    Container<Real>& rc = pmb->real_container;
    int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
    int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;
    Coordinates *pcoord = pmb->pcoord.get();
    Variable<Real>& v = rc.Get("in_or_out");
    const auto& radius = pmb->physics["PiCalculator"]->Param<Real>("radius");
    Real area = 0.0;
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie; i++) {
          area += v(k,j,i)*pcoord->dx1f(i)*pcoord->dx2f(j);
        }
      }
    }
    //std::cout << "area = " << area << std::endl;
    area /= (radius*radius);
    // just stash the area somewhere for later
    v(0,0,0) = area;
    return TaskStatus::success;
  }
}

} // namespace parthenon

