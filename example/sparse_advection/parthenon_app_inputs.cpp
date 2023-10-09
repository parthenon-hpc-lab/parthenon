// (C) (or copyright) 2023. Triad National Security, LLC. All rights reserved.
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

#include <limits>
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
          auto x = coords.Xc<1>(i) - x0;
          auto y = coords.Xc<2>(j) - y0;
          auto z = coords.Xc<3>(k);
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
      VariablePack<Real> v;

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

      pmb->par_for(
          "SparseAdvection::ProblemGenerator", 0, v.GetDim(4) - 1, kb.s, kb.e, jb.s, jb.e,
          ib.s, ib.e, KOKKOS_LAMBDA(const int n, const int k, const int j, const int i) {
            auto x = coords.Xc<1>(i) - x0;
            auto y = coords.Xc<2>(j) - y0;
            auto z = coords.Xc<3>(k);
            auto r2 = x * x + y * y + z * z;
            v(n, k, j, i) = (r2 < this_size ? 1.0 : 0.0);
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

  std::uint64_t mem_min = std::numeric_limits<std::uint64_t>::max();
  std::uint64_t mem_max = 0;
  std::uint64_t mem_tot = 0;
  std::uint64_t blocks_tot = mesh->block_list.size();
  for (auto pmb : mesh->block_list) {
    auto blk_mem = pmb->ReportMemUsage();
    mem_min = std::min(blk_mem, mem_min);
    mem_max = std::max(blk_mem, mem_max);
    mem_tot += blk_mem;
  }

#ifdef MPI_PARALLEL
  static_assert(sizeof(std::uint64_t) == sizeof(unsigned long long int),
                "MPI_UNSIGNED_LONG_LONG same as uint64_t");
  if (Globals::my_rank == 0) {
    PARTHENON_MPI_CHECK(MPI_Reduce(MPI_IN_PLACE, num_allocated.data(), n, MPI_INT,
                                   MPI_SUM, 0, MPI_COMM_WORLD));
    PARTHENON_MPI_CHECK(MPI_Reduce(MPI_IN_PLACE, &mem_min, 1, MPI_UNSIGNED_LONG_LONG,
                                   MPI_MIN, 0, MPI_COMM_WORLD));
    PARTHENON_MPI_CHECK(MPI_Reduce(MPI_IN_PLACE, &mem_max, 1, MPI_UNSIGNED_LONG_LONG,
                                   MPI_MAX, 0, MPI_COMM_WORLD));
    PARTHENON_MPI_CHECK(MPI_Reduce(MPI_IN_PLACE, &mem_tot, 1, MPI_UNSIGNED_LONG_LONG,
                                   MPI_SUM, 0, MPI_COMM_WORLD));
    PARTHENON_MPI_CHECK(MPI_Reduce(MPI_IN_PLACE, &blocks_tot, 1, MPI_UNSIGNED_LONG_LONG,
                                   MPI_SUM, 0, MPI_COMM_WORLD));
  } else {
    PARTHENON_MPI_CHECK(MPI_Reduce(num_allocated.data(), num_allocated.data(), n, MPI_INT,
                                   MPI_SUM, 0, MPI_COMM_WORLD));
    PARTHENON_MPI_CHECK(MPI_Reduce(&mem_min, &mem_min, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN,
                                   0, MPI_COMM_WORLD));
    PARTHENON_MPI_CHECK(MPI_Reduce(&mem_max, &mem_max, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX,
                                   0, MPI_COMM_WORLD));
    PARTHENON_MPI_CHECK(MPI_Reduce(&mem_tot, &mem_tot, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                                   0, MPI_COMM_WORLD));
    PARTHENON_MPI_CHECK(MPI_Reduce(&blocks_tot, &blocks_tot, 1, MPI_UNSIGNED_LONG_LONG,
                                   MPI_SUM, 0, MPI_COMM_WORLD));
  }
#endif

  // only the root process outputs the result
  if (Globals::my_rank == 0) {
    std::printf("\tNumber of allocations: ");
    for (int i = 0; i < n; ++i) {
      std::printf("%i: %i%s", i, num_allocated[i], i == n - 1 ? "" : ", ");
    }
    std::printf("\n");
    Real mem_avg = static_cast<Real>(mem_tot) / static_cast<Real>(blocks_tot);
    std::printf("\tMem used/block in bytes [min, max, avg] = [%lu, %lu, %.14e]\n",
                mem_min, mem_max, mem_avg);
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
