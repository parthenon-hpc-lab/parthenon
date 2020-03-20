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
#include <string>
#include <utility>
#ifdef MPI_PARALLEL
#include <mpi.h>
#endif
#include "pi.hpp"

namespace parthenon {

// can be used to set global properties that all meshblocks want to know about
// no need in this app so use the weak version that ships with parthenon
//Properties_t ParthenonManager::ProcessProperties(std::unique_ptr<ParameterInput>& pin) {
//  Properties_t props;
//  return std::move(props);
//}

Packages_t ParthenonManager::ProcessPackages(std::unique_ptr<ParameterInput>& pin) {
  Packages_t packages;
  // only have one package for this app, but will typically have more things added to
  packages["PiCalculator"] = PiCalculator::Initialize(pin.get());
  return std::move(packages);
}

// this should set up initial conditions of independent variables on the block
// this app only has one variable of derived type, so nothing to do here.
// in this case, just use the weak version
//void MeshBlock::ProblemGenerator(ParameterInput *pin) {
//  // nothing to do here for this app
//}

// applications can register functions to fill shared derived quantities
// before and/or after all the package FillDerived call backs
// in this case, just use the weak version that sets these to nullptr
//void ParthenonManager::SetFillDerivedFunctions() {
//  FillDerivedVariables::SetFillDerivedFunctions(nullptr,nullptr);
//}

} // namespace parthenon

DriverStatus CalculatePi::Execute() {
  // this is where the main work is orchestrated
  // No evolution in this driver.  Just calculates something once.
  // For evolution, look at the EvolutionDriver

  ConstructAndExecuteBlockTasks<>(this);

  // All the blocks are done, now do a global reduce and spit out the answer
  // first sum over blocks on this rank
  Real area = 0.0;
  MeshBlock* pmb = pmesh->pblock;
  while (pmb != nullptr) {
    Variable<Real>& v = pmb->real_container.Get("in_or_out");
    // NOTE: the MeshBlock integrated indicator function, divided
    // by r0^2, was stashed in v(0,0,0) in ComputeArea.
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
  if(my_rank == 0) {
    std::cout << std::endl << std::endl << "PI = " << pi_val <<
                 "    rel error = " << (pi_val-M_PI)/M_PI << std::endl << std::endl;
  }
  pmesh->mbcnt = pmesh->nbtotal; // this is how many blocks were processed
  return DriverStatus::complete;
}

TaskList CalculatePi::MakeTaskList(MeshBlock *pmb) {
  // make a task list for this mesh block
  using PiCalculator::ComputeArea;
  TaskList tl;

  // make some lambdas that over overkill here but clean things up for more realistic code
  auto AddBlockTask = [pmb,&tl](BlockTaskFunc func, TaskID dependencies) {
    return tl.AddTask<BlockTask>(func, dependencies, pmb);
  };

  TaskID none(0);
  auto get_area = AddBlockTask(ComputeArea, none);

  // could add more tasks like:
  // auto next_task = tl.AddTask(FuncPtr, get_area, pmb);
  // for a task that executes the function FuncPtr (with argument MeshBlock *pmb)
  // that depends on task get_area
  return std::move(tl);
}

// This defines a "physics" package
// In this case, PiCalculator provides the functions required to set up
// an indicator function in_or_out(x,y) = (r < r0 ? 1 : 0), and compute the area
// of a circle of radius r0 as A = \int d^x in_or_out(x,y) over the domain. Then
// pi \approx A/r0^2
namespace PiCalculator {

  void SetInOrOut(Container<Real>& rc) {
    MeshBlock *pmb = rc.pmy_block;
    int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
    int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;
    Coordinates *pcoord = pmb->pcoord.get();
    Variable<Real>& v = rc.Get("in_or_out");
    const auto& radius = pmb->packages["PiCalculator"]->Param<Real>("radius");
    // Set an indicator function that indicates whether the cell center
    // is inside or outside of the circle we're interating the area of.
    // see the CheckRefinement routine below for an explanation of the loop bounds
    for (int k=ks; k<=ke; k++) {
      for (int j=js-1; j<=je+1; j++) {
        for (int i=is-1; i<=ie+1; i++) {
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
    // tag cells for refinement or derefinement
    // each package can define its own refinement tagging
    // function and they are all called by parthenon
    MeshBlock *pmb = rc.pmy_block;
    int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
    int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;
    Variable<Real>& v = rc.Get("in_or_out");
    int delta_level = -1;
    Real vmin = 1.0;
    Real vmax = 0.0;
    // loop over all real cells and one layer of ghost cells and refine
    // if the edge of the circle is found.  The one layer of ghost cells
    // catches the case where the edge is between the cell centers of
    // the first/last real cell and the first ghost cell
    for (int k=ks; k<=ke; k++) {
      for (int j=js-1; j<=je+1; j++) {
        for (int i=is-1; i<=ie+1; i++) {
          vmin = (v(k,j,i) < vmin ? v(k,j,i) : vmin);
          vmax = (v(k,j,i) > vmax ? v(k,j,i) : vmax);
        }
      }
    }
    // was the edge of the circle found?
    if (vmax > 0.95 && vmin < 0.05) { // then yes
      delta_level = 1;
    }
    return delta_level;
  }

  std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
    auto package = std::make_shared<StateDescriptor>("PiCalculator");
    Params& params = package->AllParams();

    Real radius = pin->GetOrAddReal("Pi", "radius", 1.0);
    params.Add("radius", radius);

    // add a variable called in_or_out that will hold the value of the indicator function
    std::string field_name("in_or_out");
    Metadata m({Metadata::cell, Metadata::derived, Metadata::graphics});
    package->AddField(field_name, m, DerivedOwnership::unique);

    // All the package FillDerived and CheckRefinement functions are called by parthenon
    package->FillDerived = SetInOrOut;
    package->CheckRefinement = CheckRefinement;
    return package;
  }

  TaskStatus ComputeArea(MeshBlock *pmb) {
    // compute 1/r0^2 \int d^2x in_or_out(x,y) over the block's domain
    Container<Real>& rc = pmb->real_container;
    int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
    int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;
    Coordinates *pcoord = pmb->pcoord.get();
    Variable<Real>& v = rc.Get("in_or_out");
    const auto& radius = pmb->packages["PiCalculator"]->Param<Real>("radius");
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
} // namespace PiCalculator
