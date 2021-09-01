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

#include <mpi.h>
#include <sstream>
#include <string>

#include <parthenon/package.hpp>

#include "config.hpp"
#include "defs.hpp"
#include "globals.hpp"
#include "sparse_advection_driver.hpp"
#include "sparse_advection_package.hpp"
#include "utils/error_checking.hpp"

using namespace parthenon::package::prelude;
using namespace parthenon;

// *************************************************//
// redefine some weakly linked parthenon functions *//
// *************************************************//

using sparse_advection_package::NUM_FIELDS;
using sparse_advection_package::RealArr_t;

namespace sparse_advection_example {

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  using parthenon::MetadataFlag;

  auto cellbounds = pmb->cellbounds;
  IndexRange ib = cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = cellbounds.GetBoundsK(IndexDomain::entire);

  auto coords = pmb->coords;
  auto &data = pmb->meshblock_data.Get();
  auto pkg = pmb->packages.Get("sparse_advection_package");
  const auto r = pkg->Param<Real>("init_size");
  const auto size = r * r;

  const auto &x0s = pkg->Param<RealArr_t>("x0");
  const auto &y0s = pkg->Param<RealArr_t>("y0");

  for (int f = 0; f < NUM_FIELDS; ++f) {
    // we initialize sparse id i only on one rank
    if ((f % Globals::nranks) == Globals::my_rank) {
      // allocate the sparse field on the blocks where we get non-zero values
      bool any_nonzero = false;
      const Real x0 = x0s[f];
      const Real y0 = y0s[f];

      for (int k = kb.s; k <= kb.e; k++) {
        for (int j = jb.s; j <= jb.e; j++) {
          for (int i = ib.s; i <= ib.e; i++) {
            auto x = coords.x1v(i) - x0;
            auto y = coords.x2v(j) - y0;
            auto z = coords.x3v(k);
            auto r2 = x * x + y * y + z * z;
            if (r2 < size) {
              any_nonzero = true;
            }
          }
        }
      }

      printf("Block %i: any_nonzero: %s\n", pmb->gid, any_nonzero ? "YES" : "NO");

      if (any_nonzero) {
        auto v = data->AllocSparseID("sparse", f)->data;
        pmb->par_for(
            "SparseAdvection::ProblemGenerator", 0, 0, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
            KOKKOS_LAMBDA(const int n, const int k, const int j, const int i) {
              auto x = coords.x1v(i) - x0;
              auto y = coords.x2v(j) - y0;
              auto z = coords.x3v(k);
              auto r2 = x * x + y * y + z * z;
              v(n, k, j, i) = (r2 < size ? 1.0 : 0.0);
            });
      }
    }
  }
}

//========================================================================================
//! \fn void Mesh::PostStepDiagnosticsInLoop(Mesh *mes, ParameterInput *pin, SimTime &tm)
//  \brief Count the blocks on which sparse ids are allocated
//========================================================================================

void PostStepDiagnosticsInLoop(Mesh *mesh, ParameterInput *pin, const SimTime &tm) {
  auto pkg = mesh->block_list[0]->packages.Get("sparse_advection_package");
  const auto n = NUM_FIELDS;

  std::vector<int> num_allocated(n, 0);

  for (auto &pmb : mesh->block_list) {
    auto rc = pmb->meshblock_data.Get(); // get base container
    for (int i = 0; i < n; ++i) {
      if (rc->IsAllocated("sparse", i)) {
        num_allocated[i] += 1;
      }
    }
  }

#ifdef MPI_PARALLEL
  if (Globals::my_rank == 0) {
    PARTHENON_MPI_CHECK(MPI_Reduce(MPI_IN_PLACE, num_allocated.data(), 4, MPI_INT32_T,
                                   MPI_SUM, 0, MPI_COMM_WORLD));
  } else {
    PARTHENON_MPI_CHECK(MPI_Reduce(num_allocated.data(), num_allocated.data(), 4,
                                   MPI_INT32_T, MPI_SUM, 0, MPI_COMM_WORLD));
  }
#endif

  // only the root process outputs the result
  if (Globals::my_rank == 0) {
    printf("Number of expansions: ");
    for (int i = 0; i < n; ++i) {
      printf("%i: %i%s", i, num_allocated[i], i == n - 1 ? "" : ", ");
    }
    printf("\n");
  }

  return;
}

Packages_t ProcessPackages(std::unique_ptr<ParameterInput> &pin) {
  Packages_t packages;
  auto pkg = sparse_advection_package::Initialize(pin.get());
  packages.Add(pkg);

  auto app = std::make_shared<StateDescriptor>("advection_app");
  packages.Add(app);

  return packages;
}

} // namespace sparse_advection_example
