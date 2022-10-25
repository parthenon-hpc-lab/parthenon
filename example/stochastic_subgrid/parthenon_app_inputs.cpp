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

#include <sstream>
#include <string>

#include <parthenon/package.hpp>

#include "config.hpp"
#include "defs.hpp"
#include "stochastic_subgrid_driver.hpp"
#include "stochastic_subgrid_package.hpp"
#include "utils/error_checking.hpp"

using namespace parthenon::package::prelude;
using namespace parthenon;

// *************************************************//
// redefine some weakly linked parthenon functions *//
// *************************************************//

namespace stochastic_subgrid_example {

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  auto &rc = pmb->meshblock_data.Get();
  auto &q = rc->Get("advected").data;

  auto pkg = pmb->packages.Get("stochastic_subgrid_package");
  const auto &amp = pkg->Param<Real>("amp");
  const auto &vel = pkg->Param<Real>("vel");
  const auto &k_par = pkg->Param<Real>("k_par");
  const auto &cos_a2 = pkg->Param<Real>("cos_a2");
  const auto &cos_a3 = pkg->Param<Real>("cos_a3");
  const auto &sin_a2 = pkg->Param<Real>("sin_a2");
  const auto &sin_a3 = pkg->Param<Real>("sin_a3");
  const auto &profile = pkg->Param<std::string>("profile");

  auto q_h = q.GetHostMirror();

  const auto cellbounds = pmb->cellbounds;
  const IndexRange ib = cellbounds.GetBoundsI(IndexDomain::entire);
  const IndexRange jb = cellbounds.GetBoundsJ(IndexDomain::entire);
  const IndexRange kb = cellbounds.GetBoundsK(IndexDomain::entire);

  const auto &coords = pmb->coords;

  for (int n = 0; n < q_h.GetDim(4); ++n) {
    for (int k = kb.s; k <= kb.e; k++) {
      for (int j = jb.s; j <= jb.e; j++) {
        for (int i = ib.s; i <= ib.e; i++) {
          if (profile == "wave") {
            Real x = cos_a2 * (coords.Xc<1>(i) * cos_a3 + coords.Xc<2>(j) * sin_a3) +
                     coords.Xc<3>(k) * sin_a2;
            Real sn = std::sin(k_par * x);
            q_h(n, k, j, i) = 1.0 + amp * sn * vel;
          } else if (profile == "smooth_gaussian") {
            Real rsq = coords.Xc<1>(i) * coords.Xc<1>(i) + coords.Xc<2>(j) * coords.Xc<2>(j) +
                       coords.Xc<3>(k) * coords.Xc<3>(k);
            q_h(n, k, j, i) = 1. + amp * exp(-100.0 * rsq);
          } else if (profile == "hard_sphere") {
            Real rsq = coords.Xc<1>(i) * coords.Xc<1>(i) + coords.Xc<2>(j) * coords.Xc<2>(j) +
                       coords.Xc<3>(k) * coords.Xc<3>(k);
            q_h(n, k, j, i) = (rsq < 0.15 * 0.15 ? 1.0 : 0.0);
          } else {
            q_h(n, k, j, i) = 0.0;
          }

          q_h(n, k, j, i) *= (n + 1);
        }
      }
    }
  }
  q.DeepCopy(q_h);
}

//========================================================================================
//! \fn void Mesh::UserWorkAfterLoop(ParameterInput *pin, SimTime &tm)
//  \brief Compute L1 error in advection test and output to file, also make histogram of
//  number of iterations
//========================================================================================

void UserWorkAfterLoop(Mesh *mesh, ParameterInput *pin, SimTime &tm) {
  if (pin->GetOrAddBoolean("Advection", "compute_error", false)) {
    // Initialize errors to zero
    Real l1_err = 0.0;
    Real max_err = 0.0;

    for (auto &pmb : mesh->block_list) {
      auto pkg = pmb->packages.Get("stochastic_subgrid_package");

      auto rc = pmb->meshblock_data.Get(); // get base container
      const auto &amp = pkg->Param<Real>("amp");
      const auto &vel = pkg->Param<Real>("vel");
      const auto &k_par = pkg->Param<Real>("k_par");
      const auto &cos_a2 = pkg->Param<Real>("cos_a2");
      const auto &cos_a3 = pkg->Param<Real>("cos_a3");
      const auto &sin_a2 = pkg->Param<Real>("sin_a2");
      const auto &sin_a3 = pkg->Param<Real>("sin_a3");
      const auto &profile = pkg->Param<std::string>("profile");

      const IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
      const IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
      const IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

      // calculate error on host
      auto q = rc->Get("advected").data.GetHostMirrorAndCopy();
      for (int k = kb.s; k <= kb.e; k++) {
        for (int j = jb.s; j <= jb.e; j++) {
          for (int i = ib.s; i <= ib.e; i++) {
            Real ref_val;
            if (profile == "wave") {
              Real x =
                  cos_a2 * (pmb->coords.Xc<1>(i) * cos_a3 + pmb->coords.Xc<2>(j) * sin_a3) +
                  pmb->coords.Xc<3>(k) * sin_a2;
              Real sn = std::sin(k_par * x);
              ref_val = 1.0 + amp * sn * vel;
            } else if (profile == "smooth_gaussian") {
              Real rsq = pmb->coords.Xc<1>(i) * pmb->coords.Xc<1>(i) +
                         pmb->coords.Xc<2>(j) * pmb->coords.Xc<2>(j) +
                         pmb->coords.Xc<3>(k) * pmb->coords.Xc<3>(k);
              ref_val = 1. + amp * exp(-100.0 * rsq);
            } else if (profile == "hard_sphere") {
              Real rsq = pmb->coords.Xc<1>(i) * pmb->coords.Xc<1>(i) +
                         pmb->coords.Xc<2>(j) * pmb->coords.Xc<2>(j) +
                         pmb->coords.Xc<3>(k) * pmb->coords.Xc<3>(k);
              ref_val = (rsq < 0.15 * 0.15 ? 1.0 : 0.0);
            } else {
              ref_val = 1e9; // use an artifically large error
            }

            // Weight l1 error by cell volume
            Real vol = pmb->coords.Volume(k, j, i);

            l1_err += std::abs(ref_val - q(k, j, i)) * vol;
            max_err =
                std::max(static_cast<Real>(std::abs(ref_val - q(k, j, i))), max_err);
          }
        }
      }
    }

    Real max_max_over_l1 = 0.0;

#ifdef MPI_PARALLEL
    if (Globals::my_rank == 0) {
      PARTHENON_MPI_CHECK(MPI_Reduce(MPI_IN_PLACE, &l1_err, 1, MPI_PARTHENON_REAL,
                                     MPI_SUM, 0, MPI_COMM_WORLD));
      PARTHENON_MPI_CHECK(MPI_Reduce(MPI_IN_PLACE, &max_err, 1, MPI_PARTHENON_REAL,
                                     MPI_MAX, 0, MPI_COMM_WORLD));
    } else {
      PARTHENON_MPI_CHECK(MPI_Reduce(&l1_err, &l1_err, 1, MPI_PARTHENON_REAL, MPI_SUM, 0,
                                     MPI_COMM_WORLD));
      PARTHENON_MPI_CHECK(MPI_Reduce(&max_err, &max_err, 1, MPI_PARTHENON_REAL, MPI_MAX,
                                     0, MPI_COMM_WORLD));
    }
#endif

    // only the root process outputs the data
    if (Globals::my_rank == 0) {
      // normalize errors by number of cells
      auto mesh_size = mesh->mesh_size;
      Real vol = (mesh_size.x1max - mesh_size.x1min) *
                 (mesh_size.x2max - mesh_size.x2min) *
                 (mesh_size.x3max - mesh_size.x3min);
      l1_err /= vol;
      // compute rms error
      max_max_over_l1 = std::max(max_max_over_l1, (max_err / l1_err));

      // open output file and write out errors
      std::string fname;
      fname.assign("advection-errors.dat");
      std::stringstream msg;
      FILE *pfile;

      // The file exists -- reopen the file in append mode
      if ((pfile = std::fopen(fname.c_str(), "r")) != nullptr) {
        if ((pfile = std::freopen(fname.c_str(), "a", pfile)) == nullptr) {
          msg << "### FATAL ERROR in function Mesh::UserWorkAfterLoop" << std::endl
              << "Error output file could not be opened" << std::endl;
          PARTHENON_FAIL(msg.str().c_str());
        }

        // The file does not exist -- open the file in write mode and add headers
      } else {
        if ((pfile = std::fopen(fname.c_str(), "w")) == nullptr) {
          msg << "### FATAL ERROR in function Mesh::UserWorkAfterLoop" << std::endl
              << "Error output file could not be opened" << std::endl;
          PARTHENON_FAIL(msg.str().c_str());
        }
        std::fprintf(pfile, "# Nx1  Nx2  Nx3  Ncycle  ");
        std::fprintf(pfile, "L1 max_error/L1  max_error ");
        std::fprintf(pfile, "\n");
      }

      // write errors
      std::fprintf(pfile, "%d  %d", mesh_size.nx1, mesh_size.nx2);
      std::fprintf(pfile, "  %d  %d", mesh_size.nx3, tm.ncycle);
      std::fprintf(pfile, "  %e ", l1_err);
      std::fprintf(pfile, "  %e  %e  ", max_max_over_l1, max_err);
      std::fprintf(pfile, "\n");
      std::fclose(pfile);
    }
  }

  if (pin->GetOrAddBoolean("Random", "compute_histogram", true)) {
    int N_min = pin->GetInteger("Random", "num_iter_min");

    auto pkg = mesh->block_list[0]->packages.Get("stochastic_subgrid_package");
    auto hist_view = pkg->Param<Kokkos::View<int *>>("num_iter_histogram");
    auto hist = Kokkos::create_mirror_view(hist_view);
    Kokkos::deep_copy(hist, hist_view);

#ifdef MPI_PARALLEL
    if (Globals::my_rank == 0) {
      PARTHENON_MPI_CHECK(MPI_Reduce(MPI_IN_PLACE, hist.data(), hist.extent(0),
                                     MPI_INT32_T, MPI_SUM, 0, MPI_COMM_WORLD));
    } else {
      PARTHENON_MPI_CHECK(MPI_Reduce(hist.data(), hist.data(), hist.extent(0),
                                     MPI_INT32_T, MPI_SUM, 0, MPI_COMM_WORLD));
    }
#endif

    if (Globals::my_rank == 0) {
      // print histogram
      printf("\nNum Iter Histogram:\n");
      for (size_t i = 0; i < hist.extent(0); ++i) {
        printf("%8lu  %10i\n", i + size_t(N_min), hist(i));
      }
    }
  }
}

Packages_t ProcessPackages(std::unique_ptr<ParameterInput> &pin) {
  Packages_t packages;
  auto pkg = stochastic_subgrid_package::Initialize(pin.get());
  packages.Add(pkg);

  auto app = std::make_shared<StateDescriptor>("advection_app");
  packages.Add(app);

  return packages;
}

} // namespace stochastic_subgrid_example
