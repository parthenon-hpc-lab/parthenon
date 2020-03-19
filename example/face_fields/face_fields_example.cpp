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
#include <utility>
#include <vector>
#ifdef MPI_PARALLEL
#include <mpi.h>
#endif
#include "face_fields_example.hpp"
#include "parthenon_manager.hpp"

namespace parthenon {

  Packages_t ParthenonManager::ProcessPackages(std::unique_ptr<ParameterInput>& pin) {
    Packages_t packages;
    auto package = std::make_shared<StateDescriptor>("FaceFieldExample");

    Params& params = package->AllParams();
    params.Add("px", pin->GetOrAddReal("FaceExample", "px", 2.0));
    params.Add("py", pin->GetOrAddReal("FaceExample", "py", 2.0));
    params.Add("pz", pin->GetOrAddReal("FaceExample", "pz", 2.0));

    Metadata m;
    std::vector<int> array_size({2});
    m = Metadata({Metadata::cell, Metadata::vector, Metadata::derived,
          Metadata::oneCopy, Metadata::graphics}, array_size);
    package->AddField("c.c.interpolated_value", m, DerivedOwnership::unique);

    m = Metadata({Metadata::cell, Metadata::derived,
          Metadata::oneCopy, Metadata::graphics});
    package->AddField("c.c.interpolated_sum", m, DerivedOwnership::unique);

    m = Metadata({Metadata::face, Metadata::vector,
          Metadata::derived, Metadata::oneCopy}, array_size);
    package->AddField("f.f.face_averaged_value", m, DerivedOwnership::unique);

    packages["FaceFieldExample"] = package;
    return packages;
  }

  void MeshBlock::ProblemGenerator(ParameterInput *pin) {
    // don't do anything here
  }

  DriverStatus FaceFieldExample::Execute() {
    DriverUtils::ConstructAndExecuteBlockTasks<>(this);

    // post-evolution analysis
    Real rank_sum = 0.0;
    MeshBlock* pmb = pmesh->pblock;
    while (pmb != nullptr) {
      int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
      int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;
      auto& summed = pmb->real_container.Get("c.c.interpolated_sum");
      for (int k = ks; k <= ke; k++) {
        for (int j = js; j <= je; j++) {
          for (int i = is; i <= ie; i++) {
            rank_sum += summed(k,j,i);
          }
        }
      }
      pmb = pmb->next;
    }
#ifdef MPI_PARALLEL
    Real global_sum;
    MPI_Reduce(&rank_sum, &global_sum, 1, MPI_ATHENA_REAL, MPI_SUM, 0, MPI_COMM_WORLD);
#else
    Real global_sum = rank_sum;
#endif
    if (parthenon::Globals::my_rank == 0) {
      std::cout << "\n\n"
                << "Sum of all interpolated vars = " << global_sum << "\n"
                << "It should be 0.0\n"
                << std::endl;
    }

    pmesh->mbcnt = pmesh->nbtotal;
    return DriverStatus::complete;
  }

  TaskList FaceFieldExample::MakeTaskList(MeshBlock *pmb) {
    // make a task list for this mesh block
    TaskList tl;
    TaskID none(0);

    auto fill_faces = tl.AddTask<BlockTask>(FaceFields::fill_faces, none, pmb);

    auto interpolate = tl.AddTask<BlockTask>([](MeshBlock* pmb)->TaskStatus {
        Container<Real>& rc = pmb->real_container;
        int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
        int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;
        auto& face = rc.GetFace("f.f.face_averaged_value");
        auto& cell = rc.Get("c.c.interpolated_value");
        // perform interpolation
        for (int e = 0; e < 2; e++) {
          for (int k = ks; k <= ke; k++) {
            for (int j = js; j <= je; j++) {
              for (int i = is; i <= ie; i++) {
                cell(e,k,j,i) = (1./6.)*(face(1,e,k,j,i) + face(1,e,k,j,i+1)
                                         + face(2,e,k,j,i) + face(2,e,k,j+1,i)
                                         + face(3,e,k,j,i) + face(3,e,k+1,j,i));
              }
            }
          }
        }
        return TaskStatus::success;
      },
      fill_faces, pmb);

    auto sum = tl.AddTask<BlockTask>([](MeshBlock* pmb)->TaskStatus {
        Container<Real>& rc = pmb->real_container;
        int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
        int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;
        auto& interped = rc.Get("c.c.interpolated_value");
        auto& summed = rc.Get("c.c.interpolated_sum");
        for (int k = ks; k <= ke; k++) {
          for (int j = js; j <= je; j++) {
            for (int i = is; i <= ie; i++) {
              summed(k,j,i) = interped(0,k,j,i) + interped(1,k,j,i);
            }
          }
        }
        return TaskStatus::success;
      },
      interpolate, pmb);

    return std::move(tl);
  }

} // namespace parthenon

parthenon::TaskStatus FaceFields::fill_faces(parthenon::MeshBlock* pmb) {
  using parthenon::Real;

  auto physics = pmb->physics["FaceFieldExample"];
  Real px = physics->Param<Real>("px");
  Real py = physics->Param<Real>("py");
  Real pz = physics->Param<Real>("pz");
  parthenon::Container<Real>& rc = pmb->real_container;
  parthenon::Coordinates *pcoord = pmb->pcoord.get();
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;
  auto& face = rc.GetFace("f.f.face_averaged_value");
  // fill faces
  for (int e = 0; e < face.Get(1).GetDim4(); e++) {
    int sign = (e == 0) ? -1 : 1;
    for (int k=ks; k<=ke; k++) {
      Real z = pcoord->x3v(k);
      for (int j=js; j<=je; j++) {
        Real y = pcoord->x2v(j);
        for (int i=is; i<=ie+1; i++) {
          Real x = pcoord->x1f(i);
          face(1,e,k,j,i) = sign*(pow(x,px) + pow(y,py) + pow(z,pz));
        }
      }
    }
  }
  for (int e = 0; e < face.Get(2).GetDim4(); e++) {
    int sign = (e == 0) ? -1 : 1;
    for (int k=ks; k<=ke; k++) {
      Real z = pcoord->x3v(k);
      for (int j=js; j<=je+1; j++) {
        Real y = pcoord->x2f(j);
        for (int i=is; i<=ie; i++) {
          Real x = pcoord->x1v(i);
          face(2,e,k,j,i) = sign*(pow(x,px) + pow(y,py) + pow(z,pz));
        }
      }
    }
  }
  for (int e = 0; e < face.Get(3).GetDim4(); e++) {
    int sign = (e == 0) ? -1 : 1;
    for (int k=ks; k<=ke+1; k++) {
      Real z= pcoord->x3f(k);
      for (int j=js; j<=je; j++) {
        Real y = pcoord->x2v(j);
        for (int i=is; i<=ie; i++) {
          Real x = pcoord->x1v(i);
          face(3,e,k,j,i) = sign*(pow(x,px) + pow(y,py) + pow(z,pz));
        }
      }
    }
  }
  return parthenon::TaskStatus::success;
}
