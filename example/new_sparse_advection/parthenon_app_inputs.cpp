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

#include <sstream>
#include <string>

#include <parthenon/package.hpp>

#include "config.hpp"
#include "defs.hpp"
#include "globals.hpp"
#include "interface/variable.hpp"
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
  IndexRange ib = cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = cellbounds.GetBoundsK(IndexDomain::interior);

  auto coords = pmb->coords;
  auto &data = pmb->meshblock_data.Get();
  auto pkg = pmb->packages.Get("sparse_advection_package");
  const auto r = pkg->Param<Real>("init_size");
  const auto size = r * r;

  const auto &x0s = pkg->Param<RealArr_t>("x0");
  const auto &y0s = pkg->Param<RealArr_t>("y0");
  const bool do_restart_test = pkg->Param<bool>("restart_test");

  // if do_restart_test is true, start at -1 to get one more iteration to take care of the
  // restart test specific variables
  for (int f = do_restart_test ? -1 : 0; f < NUM_FIELDS; ++f) {
    const bool restart_test = (f == -1);
    const auto this_size = restart_test ? 0.5 * size : size;
    // allocate the sparse field on the blocks where we get non-zero values
    bool any_nonzero = false;
    const Real x0 = restart_test ? 0.0 : x0s[f];
    const Real y0 = restart_test ? 0.0 : y0s[f];

    for (int k = kb.s; k <= kb.e; k++) {
      for (int j = jb.s; j <= jb.e; j++) {
        for (int i = ib.s; i <= ib.e; i++) {
          auto x = coords.x1v(i) - x0;
          auto y = coords.x2v(j) - y0;
          auto z = coords.x3v(k);
          auto r2 = x * x + y * y + z * z;
          if (r2 < this_size) {
            any_nonzero = true;
            break;
          }
        }
        if (any_nonzero) break;
      }
      if (any_nonzero) break;
    }

    if (any_nonzero) {
      // Allocate all variables controlled by this variable
      auto sparse = MakeVarLabel("sparse", f);

      auto &var_names = pmb->pmy_mesh->resolved_packages->GetControlledVariables(
          MakeVarLabel("sparse", f));
      for (auto &vname : var_names)
        pmb->AllocateSparse(vname);

      // VariablePack<Real> v;
      /*
      if (restart_test) {
        pmb->AllocSparseID("shape_shift", 1);
        pmb->AllocSparseID("shape_shift", 3);
        pmb->AllocSparseID("shape_shift", 4);

        v = data->PackVariables(
            std::vector<std::string>{"dense_A", "dense_B", "shape_shift"});
      } else {
        pmb->AllocSparseID("sparse", f);
        v = data->PackVariables(std::vector<std::string>{MakeVarLabel("sparse", f)});
      }
      */

      auto tup = parthenon::SparsePack<>::Get(data.get(),
                                              std::vector<std::string>{sparse});
      auto v = std::get<0>(tup);
      auto pack_map = std::get<1>(tup);
      parthenon::PackIdx isp(pack_map[sparse]);
      const int b = 0; // Just one block in the pack

      pmb->par_for(
          "SparseAdvection::ProblemGenerator", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
          KOKKOS_LAMBDA(const int k, const int j, const int i) {
            auto x = coords.x1v(i) - x0;
            auto y = coords.x2v(j) - y0;
            auto z = coords.x3v(k);
            auto r2 = x * x + y * y + z * z;
            v(b, isp, k, j, i) = (r2 < this_size ? 1.0 : 0.0);
          });
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
    PARTHENON_MPI_CHECK(MPI_Reduce(MPI_IN_PLACE, num_allocated.data(), n, MPI_INT,
                                   MPI_SUM, 0, MPI_COMM_WORLD));
  } else {
    PARTHENON_MPI_CHECK(MPI_Reduce(num_allocated.data(), num_allocated.data(), n, MPI_INT,
                                   MPI_SUM, 0, MPI_COMM_WORLD));
  }
#endif

  // only the root process outputs the result
  if (Globals::my_rank == 0) {
    printf("Number of allocations: ");
    for (int i = 0; i < n; ++i) {
      printf("%i: %i%s", i, num_allocated[i], i == n - 1 ? "" : ", ");
    }
    printf("\n");
  }
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
