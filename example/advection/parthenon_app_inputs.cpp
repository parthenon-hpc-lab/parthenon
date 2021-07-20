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

#include "advection_driver.hpp"
#include "advection_package.hpp"
#include "config.hpp"
#include "defs.hpp"
#include "utils/error_checking.hpp"

using namespace parthenon::package::prelude;
using namespace parthenon;

// *************************************************//
// redefine some weakly linked parthenon functions *//
// *************************************************//

namespace advection_example {

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  using parthenon::MetadataFlag;

  auto &data = pmb->meshblock_data.Get();

  auto pkg = pmb->packages.Get("advection_package");
  const auto &amp = pkg->Param<Real>("amp");
  const auto &vel = pkg->Param<Real>("vel");
  const auto &k_par = pkg->Param<Real>("k_par");
  const auto &cos_a2 = pkg->Param<Real>("cos_a2");
  const auto &cos_a3 = pkg->Param<Real>("cos_a3");
  const auto &sin_a2 = pkg->Param<Real>("sin_a2");
  const auto &sin_a3 = pkg->Param<Real>("sin_a3");
  const auto &profile = pkg->Param<std::string>("profile");

  auto cellbounds = pmb->cellbounds;
  IndexRange ib = cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = cellbounds.GetBoundsK(IndexDomain::entire);

  auto coords = pmb->coords;
  auto q = data->PackVariables(std::vector<MetadataFlag>{Metadata::Independent});
  const auto num_vars = q.GetDim(4);

  int profile_type;
  if (profile == "wave") profile_type = 0;
  if (profile == "smooth_gaussian") profile_type = 1;
  if (profile == "hard_sphere") profile_type = 2;

  pmb->par_for(
      "Advection::ProblemGenerator", 0, num_vars - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int n, const int k, const int j, const int i) {
        if (profile_type == 0) {
          Real x = cos_a2 * (coords.x1v(i) * cos_a3 + coords.x2v(j) * sin_a3) +
                   coords.x3v(k) * sin_a2;
          Real sn = std::sin(k_par * x);
          q(n, k, j, i) = 1.0 + amp * sn * vel;
        } else if (profile_type == 1) {
          Real rsq = coords.x1v(i) * coords.x1v(i) + coords.x2v(j) * coords.x2v(j) +
                     coords.x3v(k) * coords.x3v(k);
          q(n, k, j, i) = 1. + amp * exp(-100.0 * rsq);
        } else if (profile_type == 2) {
          Real rsq = coords.x1v(i) * coords.x1v(i) + coords.x2v(j) * coords.x2v(j) +
                     coords.x3v(k) * coords.x3v(k);
          q(n, k, j, i) = (rsq < 0.15 * 0.15 ? 1.0 : 0.0);
        } else {
          q(n, k, j, i) = 0.0;
        }
      });
}

//========================================================================================
//! \fn void Mesh::UserWorkAfterLoop(ParameterInput *pin, SimTime &tm)
//  \brief Compute L1 error in advection test and output to file
//========================================================================================

void UserWorkAfterLoop(Mesh *mesh, ParameterInput *pin, SimTime &tm) {
  if (!pin->GetOrAddBoolean("Advection", "compute_error", false)) return;

  // Initialize errors to zero
  Real l1_err = 0.0;
  Real max_err = 0.0;

  for (auto &pmb : mesh->block_list) {
    auto pkg = pmb->packages.Get("advection_package");

    auto rc = pmb->meshblock_data.Get(); // get base container
    const auto &amp = pkg->Param<Real>("amp");
    const auto &vel = pkg->Param<Real>("vel");
    const auto &k_par = pkg->Param<Real>("k_par");
    const auto &cos_a2 = pkg->Param<Real>("cos_a2");
    const auto &cos_a3 = pkg->Param<Real>("cos_a3");
    const auto &sin_a2 = pkg->Param<Real>("sin_a2");
    const auto &sin_a3 = pkg->Param<Real>("sin_a3");
    const auto &profile = pkg->Param<std::string>("profile");

    IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
    IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
    IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

    // calculate error on host
    auto q = rc->Get("advected").data.GetHostMirrorAndCopy();
    for (int k = kb.s; k <= kb.e; k++) {
      for (int j = jb.s; j <= jb.e; j++) {
        for (int i = ib.s; i <= ib.e; i++) {
          Real ref_val;
          if (profile == "wave") {
            Real x =
                cos_a2 * (pmb->coords.x1v(i) * cos_a3 + pmb->coords.x2v(j) * sin_a3) +
                pmb->coords.x3v(k) * sin_a2;
            Real sn = std::sin(k_par * x);
            ref_val = 1.0 + amp * sn * vel;
          } else if (profile == "smooth_gaussian") {
            Real rsq = pmb->coords.x1v(i) * pmb->coords.x1v(i) +
                       pmb->coords.x2v(j) * pmb->coords.x2v(j) +
                       pmb->coords.x3v(k) * pmb->coords.x3v(k);
            ref_val = 1. + amp * exp(-100.0 * rsq);
          } else if (profile == "hard_sphere") {
            Real rsq = pmb->coords.x1v(i) * pmb->coords.x1v(i) +
                       pmb->coords.x2v(j) * pmb->coords.x2v(j) +
                       pmb->coords.x3v(k) * pmb->coords.x3v(k);
            ref_val = (rsq < 0.15 * 0.15 ? 1.0 : 0.0);
          } else {
            ref_val = 1e9; // use an artificially large error
          }

          // Weight l1 error by cell volume
          Real vol = pmb->coords.Volume(k, j, i);

          l1_err += std::abs(ref_val - q(k, j, i)) * vol;
          max_err = std::max(static_cast<Real>(std::abs(ref_val - q(k, j, i))), max_err);
        }
      }
    }
  }

  Real max_max_over_l1 = 0.0;

#ifdef MPI_PARALLEL
  if (Globals::my_rank == 0) {
    PARTHENON_MPI_CHECK(MPI_Reduce(MPI_IN_PLACE, &l1_err, 1, MPI_PARTHENON_REAL, MPI_SUM,
                                   0, MPI_COMM_WORLD));
    PARTHENON_MPI_CHECK(MPI_Reduce(MPI_IN_PLACE, &max_err, 1, MPI_PARTHENON_REAL, MPI_MAX,
                                   0, MPI_COMM_WORLD));
  } else {
    PARTHENON_MPI_CHECK(
        MPI_Reduce(&l1_err, &l1_err, 1, MPI_PARTHENON_REAL, MPI_SUM, 0, MPI_COMM_WORLD));
    PARTHENON_MPI_CHECK(MPI_Reduce(&max_err, &max_err, 1, MPI_PARTHENON_REAL, MPI_MAX, 0,
                                   MPI_COMM_WORLD));
  }
#endif

  // only the root process outputs the data
  if (Globals::my_rank == 0) {
    // normalize errors by number of cells
    auto mesh_size = mesh->mesh_size;
    Real vol = (mesh_size.x1max - mesh_size.x1min) * (mesh_size.x2max - mesh_size.x2min) *
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

  return;
}

Packages_t ProcessPackages(std::unique_ptr<ParameterInput> &pin) {
  Packages_t packages;
  auto pkg = advection_package::Initialize(pin.get());
  packages.Add(pkg);

  auto app = std::make_shared<StateDescriptor>("advection_app");
  app->PreFillDerivedBlock = advection_package::PreFill;
  app->PostFillDerivedBlock = advection_package::PostFill;
  packages.Add(app);

  return packages;
}

} // namespace advection_example
