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
void ProcessProperties(properties_t& properties, ParameterInput *pin) {}

void InitializePhysics(physics_t& physics, ParameterInput *pin) {
  auto package = std::make_shared<StateDescriptor>("FaceFieldExample");

  Params& params = package->AllParams();
  params.Add("px",
	     pin->DoesParameterExist("FaceExample","px") ?
	     pin->GetReal("FaceExample","px") : 1.0);
  params.Add("py",
	     pin->DoesParameterExist("FaceExample","py") ?
	     pin->GetReal("FaceExample","py") : 1.0);
  

  Metada m;
  m = Metadata({Metadata::cell, Metadata::derived,
		Metadata::oneCopy, Metadata::graphics});
  package->AddField("c.c.interpolated_value", m, DerivedOwnership::unique);

  m = Metadata({Metadata::face, Metadata::derived, Metadata::oneCopy});
  package->AddField("f.f.face_averaged_value", m, DerivedOwnership::unique);
  
  physics["FaceFieldExample"] = package;
}

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  // don't do anything here
}

// TODO: should this be a staged driver with one stage?
DriverStatus FaceFieldExample::Execute() {
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
  pmesh->mbcnt = pmesh->nbtotal; // this is how many blocks were processed
  return DriverStatus::complete;
}

TaskList FaceFieldExample::MakeTaskList(MeshBlock *pmb) {
  // make a task list for this mesh block
  TaskList tl;

  TaskID none(0);
  auto get_area = tl.AddTask([](MeshBlock* pmb)->TaskStatus {
			       auto physics = pmb->physics["FaceFieldExample"];
			       auto px = physics->Param<Real>("px");
			       auto py = physics->Param<Real>("py");
			       Container<Real>& rc = pmb->real_container;
			       int is = pmb->is;
			       int js = pmb->js;
			       int ks = pmb->ks;
			       int ie = pmb->ie;
			       int je = pmb->je;
			       int ke = pmb->ke;
			       Coordinates *pcoord = pmb->pcoord.get();
			       Variable<Real>& face = rc.Get("c.c.face_averaged_value");
			       Variable<Real>& cell = rc.Get("f.f.interpolated_value");
			       
			     },
			     none, pmb);
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

