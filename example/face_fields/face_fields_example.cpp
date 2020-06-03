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

#include "face_fields_example.hpp"

#include <iostream>
#include <utility>
#include <vector>

#include "parthenon_mpi.hpp"

#include "parthenon_manager.hpp"

namespace parthenon {

Packages_t ParthenonManager::ProcessPackages(std::unique_ptr<ParameterInput> &pin) {
  Packages_t packages;
  auto package = std::make_shared<StateDescriptor>("FaceFieldExample");

  Params &params = package->AllParams();
  params.Add("px", pin->GetOrAddReal("FaceExample", "px", 2.0));
  params.Add("py", pin->GetOrAddReal("FaceExample", "py", 2.0));
  params.Add("pz", pin->GetOrAddReal("FaceExample", "pz", 2.0));

  Metadata m;
  std::vector<int> array_size({2});
  m = Metadata({Metadata::Cell, Metadata::Vector, Metadata::Derived, Metadata::OneCopy,
                Metadata::Graphics},
               array_size);
  package->AddField("c.c.interpolated_value", m, DerivedOwnership::unique);

  m = Metadata(
      {Metadata::Cell, Metadata::Derived, Metadata::OneCopy, Metadata::Graphics});
  package->AddField("c.c.interpolated_sum", m, DerivedOwnership::unique);

  m = Metadata({Metadata::Face, Metadata::Vector, Metadata::Derived, Metadata::OneCopy},
               array_size);
  package->AddField("f.f.face_averaged_value", m, DerivedOwnership::unique);

  packages["FaceFieldExample"] = package;
  return packages;
}

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  // don't do anything here
}

DriverStatus FaceFieldExample::Execute() {
  PreExecute();
  DriverUtils::ConstructAndExecuteBlockTasks<>(this);

  // post-evolution analysis
  Real rank_sum = 0.0;
  MeshBlock *pmb = pmesh->pblock;
  while (pmb != nullptr) {
    parthenon::IndexDomain interior = parthenon::IndexDomain::interior;
    parthenon::IndexRange ib = pmb->cellbounds.GetBoundsI(interior);
    parthenon::IndexRange jb = pmb->cellbounds.GetBoundsJ(interior);
    parthenon::IndexRange kb = pmb->cellbounds.GetBoundsK(interior);
    Container<Real> &rc = pmb->real_containers.Get();
    auto &summed = rc.Get("c.c.interpolated_sum");
    for (int k = kb.s; k <= kb.e; k++) {
      for (int j = jb.s; j <= jb.e; j++) {
        for (int i = ib.s; i <= ib.e; i++) {
          rank_sum += summed(k, j, i);
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
  PostExecute();
  return DriverStatus::complete;
}

TaskList FaceFieldExample::MakeTaskList(MeshBlock *pmb) {
  // make a task list for this mesh block
  TaskList tl;
  TaskID none(0);

  auto fill_faces = tl.AddTask<BlockTask>(FaceFields::fill_faces, none, pmb);

  auto interpolate = tl.AddTask<BlockTask>(
      [](MeshBlock *pmb) -> TaskStatus {
        Container<Real> &rc = pmb->real_containers.Get();
        parthenon::IndexDomain interior = parthenon::IndexDomain::interior;
        parthenon::IndexRange ib = pmb->cellbounds.GetBoundsI(interior);
        parthenon::IndexRange jb = pmb->cellbounds.GetBoundsJ(interior);
        parthenon::IndexRange kb = pmb->cellbounds.GetBoundsK(interior);
        auto &face = rc.GetFace("f.f.face_averaged_value");
        auto &cell = rc.Get("c.c.interpolated_value");
        // perform interpolation
        for (int e = 0; e < 2; e++) {
          for (int k = kb.s; k <= kb.e; k++) {
            for (int j = jb.s; j <= jb.e; j++) {
              for (int i = ib.s; i <= ib.e; i++) {
                cell(e, k, j, i) =
                    (1. / 6.) * (face(1, e, k, j, i) + face(1, e, k, j, i + 1) +
                                 face(2, e, k, j, i) + face(2, e, k, j + 1, i) +
                                 face(3, e, k, j, i) + face(3, e, k + 1, j, i));
              }
            }
          }
        }
        return TaskStatus::complete;
      },
      fill_faces, pmb);

  auto sum = tl.AddTask<BlockTask>(
      [](MeshBlock *pmb) -> TaskStatus {
        Container<Real> &rc = pmb->real_containers.Get();
        parthenon::IndexDomain interior = parthenon::IndexDomain::interior;
        parthenon::IndexRange ib = pmb->cellbounds.GetBoundsI(interior);
        parthenon::IndexRange jb = pmb->cellbounds.GetBoundsJ(interior);
        parthenon::IndexRange kb = pmb->cellbounds.GetBoundsK(interior);
        auto &interped = rc.Get("c.c.interpolated_value");
        auto &summed = rc.Get("c.c.interpolated_sum");
        for (int k = kb.s; k <= kb.e; k++) {
          for (int j = jb.s; j <= jb.e; j++) {
            for (int i = ib.s; i <= ib.e; i++) {
              summed(k, j, i) = interped(0, k, j, i) + interped(1, k, j, i);
            }
          }
        }
        return TaskStatus::complete;
      },
      interpolate, pmb);

  return tl;
}

} // namespace parthenon

parthenon::TaskStatus FaceFields::fill_faces(parthenon::MeshBlock *pmb) {
  using parthenon::Real;

  auto example = pmb->packages["FaceFieldExample"];
  Real px = example->Param<Real>("px");
  Real py = example->Param<Real>("py");
  Real pz = example->Param<Real>("pz");
  parthenon::Container<Real> &rc = pmb->real_containers.Get();
  auto coords = pmb->coords;
  parthenon::IndexDomain interior = parthenon::IndexDomain::interior;
  parthenon::IndexRange ib = pmb->cellbounds.GetBoundsI(interior);
  parthenon::IndexRange jb = pmb->cellbounds.GetBoundsJ(interior);
  parthenon::IndexRange kb = pmb->cellbounds.GetBoundsK(interior);
  auto &face = rc.GetFace("f.f.face_averaged_value");
  // fill faces
  for (int e = 0; e < face.Get(1).GetDim(4); e++) {
    int sign = (e == 0) ? -1 : 1;
    for (int k = kb.s; k <= kb.e; k++) {
      Real z = coords.x3v(k);
      for (int j = jb.s; j <= jb.e; j++) {
        Real y = coords.x2v(j);
        for (int i = ib.s; i <= ib.e + 1; i++) {
          Real x = coords.x1f(i);
          face(1, e, k, j, i) = sign * (pow(x, px) + pow(y, py) + pow(z, pz));
        }
      }
    }
  }
  for (int e = 0; e < face.Get(2).GetDim(4); e++) {
    int sign = (e == 0) ? -1 : 1;
    for (int k = kb.s; k <= kb.e; k++) {
      Real z = coords.x3v(k);
      for (int j = jb.s; j <= jb.e + 1; j++) {
        Real y = coords.x2f(j);
        for (int i = ib.s; i <= ib.e; i++) {
          Real x = coords.x1v(i);
          face(2, e, k, j, i) = sign * (pow(x, px) + pow(y, py) + pow(z, pz));
        }
      }
    }
  }
  for (int e = 0; e < face.Get(3).GetDim(4); e++) {
    int sign = (e == 0) ? -1 : 1;
    for (int k = kb.s; k <= kb.e + 1; k++) {
      Real z = coords.x3f(k);
      for (int j = jb.s; j <= jb.e; j++) {
        Real y = coords.x2v(j);
        for (int i = ib.s; i <= ib.e; i++) {
          Real x = coords.x1v(i);
          face(3, e, k, j, i) = sign * (pow(x, px) + pow(y, py) + pow(z, pz));
        }
      }
    }
  }
  return parthenon::TaskStatus::complete;
}
