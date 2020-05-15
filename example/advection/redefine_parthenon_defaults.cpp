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

#include <sstream>

#include <parthenon/package.hpp>
#include <string>

#include "advection_package.hpp"
#include "defs.hpp"
#include "utils/error_checking.hpp"

using namespace parthenon::package::prelude;

// *************************************************//
// redefine some weakly linked parthenon functions *//
// *************************************************//

namespace parthenon {

Packages_t ParthenonManager::ProcessPackages(std::unique_ptr<ParameterInput> &pin) {
  Packages_t packages;
  auto pkg = advection_package::Initialize(pin.get());
  packages[pkg->label()] = pkg;
  return packages;
}

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  Container<Real> &rc = real_containers.Get();
  CellVariable<Real> &q = rc.Get("advected");

  auto pkg = packages["advection_package"];
  const auto &amp = pkg->Param<Real>("amp");
  const auto &vel = pkg->Param<Real>("vel");
  const auto &k_par = pkg->Param<Real>("k_par");
  const auto &cos_a2 = pkg->Param<Real>("cos_a2");
  const auto &cos_a3 = pkg->Param<Real>("cos_a3");
  const auto &sin_a2 = pkg->Param<Real>("sin_a2");
  const auto &sin_a3 = pkg->Param<Real>("sin_a3");
  const auto &profile = pkg->Param<int>("profile");

  for (int k = 0; k < ncells3; k++) {
    for (int j = 0; j < ncells2; j++) {
      for (int i = 0; i < ncells1; i++) {
        if (profile == 0) { // wave
          Real x = cos_a2 * (pcoord->x1v(i) * cos_a3 + pcoord->x2v(j) * sin_a3) +
                   pcoord->x3v(k) * sin_a2;
          Real sn = std::sin(k_par * x);
          q(k, j, i) = 1.0 + amp * sn * vel;
        } else if (profile == 1) { // smooth gaussian
          Real r = std::sqrt(pcoord->x1v(i) * pcoord->x1v(i) +
                             pcoord->x2v(j) * pcoord->x2v(j) +
                             pcoord->x3v(k) * pcoord->x3v(k));
          q(k, j, i) = 1. + exp(-100.0 * r * r);
        } else {
          q(k, j, i) = 0.0;
        }
      }
    }
  }
}

void ParthenonManager::SetFillDerivedFunctions() {
  FillDerivedVariables::SetFillDerivedFunctions(advection_package::PreFill,
                                                advection_package::PostFill);
}

//========================================================================================
//! \fn void Mesh::UserWorkAfterLoop(ParameterInput *pin, SimTime &tm)
//  \brief Compute L1 error in advection test and output to file
//========================================================================================

void Mesh::UserWorkAfterLoop(ParameterInput *pin, SimTime &tm) {
  if (!pin->GetOrAddBoolean("Advection", "compute_error", false)) return;

  // Initialize errors to zero
  Real l1_err = 0.0;
  Real max_err = 0.0;

  MeshBlock *pmb = pblock;
  while (pmb != nullptr) {
    auto pkg = pmb->packages["advection_package"];

    int il = pmb->is, iu = pmb->ie, jl = pmb->js, ju = pmb->je, kl = pmb->ks,
        ku = pmb->ke;

    auto rc = pmb->real_containers.Get(); // get base container
    ParArray3D<Real> q = rc.Get("advected").data.Get<3>();
    const auto &amp = pkg->Param<Real>("amp");
    const auto &vel = pkg->Param<Real>("vel");
    const auto &k_par = pkg->Param<Real>("k_par");
    const auto &cos_a2 = pkg->Param<Real>("cos_a2");
    const auto &cos_a3 = pkg->Param<Real>("cos_a3");
    const auto &sin_a2 = pkg->Param<Real>("sin_a2");
    const auto &sin_a3 = pkg->Param<Real>("sin_a3");
    const auto &profile = pkg->Param<int>("profile");

    // TODO(pgrete) needs to be a reduction when using parallel_for
    for (int k = kl; k <= ku; ++k) {
      for (int j = jl; j <= ju; ++j) {
        for (int i = il; i <= iu; ++i) {
          Real ref_val;
          if (profile == 0) { // wave
            Real x =
                cos_a2 * (pmb->pcoord->x1v(i) * cos_a3 + pmb->pcoord->x2v(j) * sin_a3) +
                pmb->pcoord->x3v(k) * sin_a2;
            Real sn = std::sin(k_par * x);
            ref_val = 1.0 + amp * sn * vel;
          } else if (profile == 1) { // smooth gaussian
            Real r = std::sqrt(pmb->pcoord->x1v(i) * pmb->pcoord->x1v(i) +
                               pmb->pcoord->x2v(j) * pmb->pcoord->x2v(j) +
                               pmb->pcoord->x3v(k) * pmb->pcoord->x3v(k));
            ref_val = 1. + exp(-100.0 * r * r);
          } else {
            ref_val = 1e9; // use an artifically large error
          }

          // Weight l1 error by cell volume
          Real vol = pmb->pcoord->GetCellVolume(k, j, i);

          l1_err += std::abs(ref_val - q(k, j, i)) * vol;
          max_err = std::max(static_cast<Real>(std::abs(ref_val - q(k, j, i))), max_err);
        }
      }
    }
    pmb = pmb->next;
  }

  Real max_max_over_l1 = 0.0;

#ifdef MPI_PARALLEL
  if (Globals::my_rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, &l1_err, 1, MPI_ATHENA_REAL, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &max_err, 1, MPI_ATHENA_REAL, MPI_MAX, 0, MPI_COMM_WORLD);
  } else {
    MPI_Reduce(&l1_err, &l1_err, 1, MPI_ATHENA_REAL, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&max_err, &max_err, 1, MPI_ATHENA_REAL, MPI_MAX, 0, MPI_COMM_WORLD);
  }
#endif

  // only the root process outputs the data
  if (Globals::my_rank == 0) {
    // normalize errors by number of cells
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
        PARTHENON_FAIL(msg.str());
      }

      // The file does not exist -- open the file in write mode and add headers
    } else {
      if ((pfile = std::fopen(fname.c_str(), "w")) == nullptr) {
        msg << "### FATAL ERROR in function Mesh::UserWorkAfterLoop" << std::endl
            << "Error output file could not be opened" << std::endl;
        PARTHENON_FAIL(msg.str());
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

} // namespace parthenon
