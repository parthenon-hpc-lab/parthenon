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
#include "face_fields_example.hpp"

namespace parthenon {

// can be used to set global properties that all meshblocks want to know about
void ProcessProperties(properties_t& properties, ParameterInput *pin) {}

// Usually you would call "initialize" functions from several physics packages here.
// Since there's only one, we use only one.
void InitializePhysics(physics_t& physics, ParameterInput *pin) {
  auto package = std::make_shared<StateDescriptor>("FaceFieldExample");

  Params& params = package->AllParams();
  params.Add("px",
	     pin->DoesParameterExist("FaceExample","px") ?
	     pin->GetReal("FaceExample","px") : 1.0);
  params.Add("py",
	     pin->DoesParameterExist("FaceExample","py") ?
	     pin->GetReal("FaceExample","py") : 1.0);
  params.Add("pz",
	     pin->DoesParameterExist("FaceExample","pz") ?
	     pin->GetReal("FaceExample","pz") : 0);
  

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
  pmesh->mbcnt = pmesh->nbtotal;
  return DriverStatus::complete;
}

TaskList FaceFieldExample::MakeTaskList(MeshBlock *pmb) {
  // make a task list for this mesh block
  TaskList tl;

  TaskID none(0);
  auto fill_faces = tl.AddTask([](MeshBlock* pmb)->TaskStatus {
      auto physics = pmb->physics["FaceFieldExample"];
      Real px = physics->Param<Real>("px");
      Real py = physics->Param<Real>("py");
      Real pz = physics->Param<Real>("pz");
      Container<Real>& rc = pmb->real_container;
      Coordinates *pcoord = pmb->pcoord.get();
      int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
      int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;
      auto& face = rc.GetFace("c.c.face_averaged_value");
      // fill faces
      for (int k=ks; k<=ke; k++) {
	Real z = pcoord->x3v(k);
	for (int j=js; j<=je; j++) {
	  Real y = pcoord->x2v(j);
	  for (int i=is; i<=ie+1; i++) {
	    Real x = pcoord->x1f(i);
	    face(1,k,j,i) = pow(x,px) + pow(y,py) + pow(z,pz);
	  }
	}
      }
      for (int k=ks; k<=ke; k++) {
	Real z = pcoord->x3v(k);
	for (int j=js; j<=je+1; j++) {
	  Real y = pcoord->x2f(j);
	  for (int i=is; i<=ie; i++) {
	    Real x = pcoord->x1v(i);
	    face(2,k,j,i) = pow(x,px) + pow(y,py) + pow(z,pz);
	  }
	}
      }
      for (int k=ks; k<=ke+1; k++) {
	Real z= pcoord->x3f(k);
	for (int j=js; j<=je; j++) {
	  Real y = pcoord->x2v(j);
	  for (int i=is; i<=ie; i++) {
	    Real x = pcoord->x1v(i);
	    face(3,k,j,i) = pow(x,px) + pow(y,py) + pow(z,pz);
	  }
	}
      }
    },
    none, pmb);

  auto interpolate = tl.AddTask([](MeshBlock* pmb)->TaskStatus {
      Container<Real>& rc = pmb->real_container;
      int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
      int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;
      auto& face = rc.GetFace("c.c.face_averaged_value");
      auto& cell = rc.Get("f.f.interpolated_value");
      // perform interpolation
      for (int k = ks; k <= ke; k++) {
	for (int j = js; j <= je; j++) {
	  for (int i = is; i <= ie; i++) {
	    cell(k,j,i) = (1./6.)*(face(1,k,j,i) + face(1,k,j,i+1)
				   + face(2,k,j,i) + face(2,k,j+1,i)
				   + face(3,k,j,i) + face(3,k+1,j,i));
	  }
	}
      }
    },
    fill_faces, pmb);
  
  return std::move(tl);
}

} // namespace parthenon

