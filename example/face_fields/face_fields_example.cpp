//========================================================================================
// (C) (or copyright) 2020-2021. Triad National Security, LLC. All rights reserved.
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

using parthenon::DriverStatus;
using parthenon::MeshBlock;
using parthenon::Metadata;
using parthenon::Packages_t;
using parthenon::ParameterInput;
using parthenon::Params;
using parthenon::Real;
using parthenon::StateDescriptor;
using parthenon::TaskID;
using parthenon::TaskList;
using parthenon::TaskStatus;

namespace FaceFields {

Packages_t ProcessPackages(std::unique_ptr<ParameterInput> &pin) {
  Packages_t packages;
  auto package = std::make_shared<StateDescriptor>("FaceFieldExample");

  Params &params = package->AllParams();
  params.Add("px", pin->GetOrAddReal("FaceExample", "px", 2.0));
  params.Add("py", pin->GetOrAddReal("FaceExample", "py", 2.0));
  params.Add("pz", pin->GetOrAddReal("FaceExample", "pz", 2.0));

  Metadata m;
  std::vector<int> array_size({2});
  m = Metadata({Metadata::Cell, Metadata::Vector, Metadata::Derived, Metadata::OneCopy},
               array_size);
  package->AddField("c.c.interpolated_value", m);

  m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy});
  package->AddField("c.c.interpolated_sum", m);

  m = Metadata({Metadata::Face, Metadata::Vector, Metadata::Derived, Metadata::OneCopy},
               array_size);
  package->AddField("f.f.face_averaged_value", m);

  packages.Add(package);
  return packages;
}

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  // don't do anything here
}

DriverStatus FaceFieldExample::Execute() {
  Driver::PreExecute();
  parthenon::DriverUtils::ConstructAndExecuteBlockTasks<>(this);

  // post-evolution analysis
  Real rank_sum = 0.0;
  for (auto &pmb : pmesh->block_list) {
    parthenon::IndexDomain const interior = parthenon::IndexDomain::interior;
    parthenon::IndexRange const ib = pmb->cellbounds.GetBoundsI(interior);
    parthenon::IndexRange const jb = pmb->cellbounds.GetBoundsJ(interior);
    parthenon::IndexRange const kb = pmb->cellbounds.GetBoundsK(interior);
    auto &rc = pmb->meshblock_data.Get();
    auto &summed = rc->Get("c.c.interpolated_sum").data;
    for (int k = kb.s; k <= kb.e; k++) {
      for (int j = jb.s; j <= jb.e; j++) {
        for (int i = ib.s; i <= ib.e; i++) {
          rank_sum += summed(k, j, i);
        }
      }
    }
  }
#ifdef MPI_PARALLEL
  Real global_sum;
  PARTHENON_MPI_CHECK(MPI_Reduce(&rank_sum, &global_sum, 1, MPI_PARTHENON_REAL, MPI_SUM,
                                 0, MPI_COMM_WORLD));
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
  Driver::PostExecute(DriverStatus::complete);
  return DriverStatus::complete;
}

TaskList FaceFieldExample::MakeTaskList(MeshBlock *pmb) {
  // make a task list for this mesh block
  TaskList tl;
  TaskID none(0);

  auto fill_faces = tl.AddTask(none, FaceFields::fill_faces, pmb);

  auto interpolate = tl.AddTask(
      fill_faces,
      [](MeshBlock *pmb) -> TaskStatus {
        auto &rc = pmb->meshblock_data.Get();
        parthenon::IndexDomain interior = parthenon::IndexDomain::interior;
        parthenon::IndexRange ib = pmb->cellbounds.GetBoundsI(interior);
        parthenon::IndexRange jb = pmb->cellbounds.GetBoundsJ(interior);
        parthenon::IndexRange kb = pmb->cellbounds.GetBoundsK(interior);
        auto &x1f = rc->GetFace("f.f.face_averaged_value").Get(1);
        auto &x2f = rc->GetFace("f.f.face_averaged_value").Get(2);
        auto &x3f = rc->GetFace("f.f.face_averaged_value").Get(3);
        auto &cell = rc->Get("c.c.interpolated_value").data;
        // perform interpolation
        for (int e = 0; e < 2; e++) {
          for (int k = kb.s; k <= kb.e; k++) {
            for (int j = jb.s; j <= jb.e; j++) {
              for (int i = ib.s; i <= ib.e; i++) {
                cell(e, k, j, i) = (1. / 6.) * (x1f(e, k, j, i) + x1f(e, k, j, i + 1) +
                                                x2f(e, k, j, i) + x2f(e, k, j + 1, i) +
                                                x3f(e, k, j, i) + x3f(e, k + 1, j, i));
              }
            }
          }
        }
        return TaskStatus::complete;
      },
      pmb);

  auto sum = tl.AddTask(
      interpolate,
      [](MeshBlock *pmb) -> TaskStatus {
        auto &rc = pmb->meshblock_data.Get();
        parthenon::IndexDomain interior = parthenon::IndexDomain::interior;
        parthenon::IndexRange ib = pmb->cellbounds.GetBoundsI(interior);
        parthenon::IndexRange jb = pmb->cellbounds.GetBoundsJ(interior);
        parthenon::IndexRange kb = pmb->cellbounds.GetBoundsK(interior);
        auto &interped = rc->Get("c.c.interpolated_value").data;
        auto &summed = rc->Get("c.c.interpolated_sum").data;
        for (int k = kb.s; k <= kb.e; k++) {
          for (int j = jb.s; j <= jb.e; j++) {
            for (int i = ib.s; i <= ib.e; i++) {
              summed(k, j, i) = interped(0, k, j, i) + interped(1, k, j, i);
            }
          }
        }
        return TaskStatus::complete;
      },
      pmb);

  return tl;
}

parthenon::TaskStatus fill_faces(parthenon::MeshBlock *pmb) {
  using parthenon::Real;

  auto example = pmb->packages.Get("FaceFieldExample");
  Real px = example->Param<Real>("px");
  Real py = example->Param<Real>("py");
  Real pz = example->Param<Real>("pz");
  auto &rc = pmb->meshblock_data.Get();
  auto coords = pmb->coords;
  parthenon::IndexDomain interior = parthenon::IndexDomain::interior;
  parthenon::IndexRange ib = pmb->cellbounds.GetBoundsI(interior);
  parthenon::IndexRange jb = pmb->cellbounds.GetBoundsJ(interior);
  parthenon::IndexRange kb = pmb->cellbounds.GetBoundsK(interior);
  auto &x1f = rc->GetFace("f.f.face_averaged_value").Get(1);
  auto &x2f = rc->GetFace("f.f.face_averaged_value").Get(2);
  auto &x3f = rc->GetFace("f.f.face_averaged_value").Get(3);
  // fill faces
  for (int e = 0; e < x1f.GetDim(4); e++) {
    int sign = (e == 0) ? -1 : 1;
    for (int k = kb.s; k <= kb.e; k++) {
      Real z = coords.x3v(k);
      for (int j = jb.s; j <= jb.e; j++) {
        Real y = coords.x2v(j);
        for (int i = ib.s; i <= ib.e + 1; i++) {
          Real x = coords.x1f(i);
          x1f(e, k, j, i) = sign * (pow(x, px) + pow(y, py) + pow(z, pz));
        }
      }
    }
  }
  for (int e = 0; e < x2f.GetDim(4); e++) {
    int sign = (e == 0) ? -1 : 1;
    for (int k = kb.s; k <= kb.e; k++) {
      Real z = coords.x3v(k);
      for (int j = jb.s; j <= jb.e + 1; j++) {
        Real y = coords.x2f(j);
        for (int i = ib.s; i <= ib.e; i++) {
          Real x = coords.x1v(i);
          x2f(e, k, j, i) = sign * (pow(x, px) + pow(y, py) + pow(z, pz));
        }
      }
    }
  }
  for (int e = 0; e < x3f.GetDim(4); e++) {
    int sign = (e == 0) ? -1 : 1;
    for (int k = kb.s; k <= kb.e + 1; k++) {
      Real z = coords.x3f(k);
      for (int j = jb.s; j <= jb.e; j++) {
        Real y = coords.x2v(j);
        for (int i = ib.s; i <= ib.e; i++) {
          Real x = coords.x1v(i);
          x3f(e, k, j, i) = sign * (pow(x, px) + pow(y, py) + pow(z, pz));
        }
      }
    }
  }
  return parthenon::TaskStatus::complete;
}

} // namespace FaceFields
