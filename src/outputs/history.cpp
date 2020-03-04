//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
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
//! \file history.cpp
//  \brief writes history output data, volume-averaged quantities that are output
//         frequently in time to trace their history.

// C headers

// C++ headers
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

// Athena++ headers
#include "athena.hpp"
#include "athena_arrays.hpp"
#include "coordinates/coordinates.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "outputs.hpp"

// NEW_OUTPUT_TYPES:

// "3" for 1-KE, 2-KE, 3-KE additional columns (come before tot-E)
#define NHISTORY_VARS ((NHYDRO) + (NFIELD) + 3)

namespace parthenon {
//----------------------------------------------------------------------------------------
//! \fn void OutputType::HistoryFile()
//  \brief Writes a history file

void HistoryOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin, bool flag) {
  MeshBlock *pmb = pm->pblock;
  Real real_max = std::numeric_limits<Real>::max();
  Real real_min = std::numeric_limits<Real>::min();
  AthenaArray<Real> vol(pmb->ncells1);
  const int nhistory_output = NHISTORY_VARS + pm->nuser_history_output_;
  std::unique_ptr<Real[]> hst_data(new Real[nhistory_output]);
  // initialize built-in variable sums to 0.0
  for (int n=0; n<NHISTORY_VARS; ++n) hst_data[n] = 0.0;
  // initialize user-defined history outputs depending on the requested operation
  for (int n=0; n<pm->nuser_history_output_; n++) {
    switch (pm->user_history_ops_[n]) {
      case UserHistoryOperation::sum:
        hst_data[NHISTORY_VARS+n] = 0.0;
        break;
      case UserHistoryOperation::max:
        hst_data[NHISTORY_VARS+n] = real_min;
        break;
      case UserHistoryOperation::min:
        hst_data[NHISTORY_VARS+n] = real_max;
        break;
    }
  }

  // Loop over MeshBlocks
  while (pmb != nullptr) {
    // Sum history variables over cells.  Note ghost cells are never included in sums
    for (int k=pmb->ks; k<=pmb->ke; ++k) {
      for (int j=pmb->js; j<=pmb->je; ++j) {
        pmb->pcoord->CellVolume(k, j, pmb->is, pmb->ie, vol);
        for (int i=pmb->is; i<=pmb->ie; ++i) {
          // NEW_OUTPUT_TYPES:

          // Hydro conserved variables:

          hst_data[0] += 0.0;
          hst_data[1] += 0.0;
          hst_data[2] += 0.0;
          hst_data[3] += 0.0;
          // + partitioned KE by coordinate direction:
          hst_data[4] += 0.0;
          hst_data[5] += 0.0;
          hst_data[6] += 0.0;
        }
      }
    }
    for (int n=0; n<pm->nuser_history_output_; n++) { // user-defined history outputs
      if (pm->user_history_func_[n] != nullptr) {
        Real usr_val = pm->user_history_func_[n](pmb, n);
        switch (pm->user_history_ops_[n]) {
          case UserHistoryOperation::sum:
            // TODO(felker): this should automatically volume-weight the sum, like the
            // built-in variables. But existing user-defined .hst fns are currently
            // weighting their returned values.
            hst_data[NHISTORY_VARS+n] += usr_val;
            break;
          case UserHistoryOperation::max:
            hst_data[NHISTORY_VARS+n] = std::max(usr_val, hst_data[NHISTORY_VARS+n]);
            break;
          case UserHistoryOperation::min:
            hst_data[NHISTORY_VARS+n] = std::min(usr_val, hst_data[NHISTORY_VARS+n]);
            break;
        }
      }
    }
    pmb = pmb->next;
  }  // end loop over MeshBlocks

#ifdef MPI_PARALLEL
  // sum built-in/predefined hst_data[] over all ranks
  if (Globals::my_rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, hst_data.get(), NHISTORY_VARS, MPI_ATHENA_REAL, MPI_SUM, 0,
               MPI_COMM_WORLD);
  } else {
    MPI_Reduce(hst_data.get(), hst_data.get(), NHISTORY_VARS, MPI_ATHENA_REAL, MPI_SUM,
               0, MPI_COMM_WORLD);
  }
  // apply separate chosen operations to each user-defined history output
  for (int n=0; n<pm->nuser_history_output_; n++) {
    Real *usr_hst_data = hst_data.get() + NHISTORY_VARS + n;
    MPI_Op usr_op;
    switch (pm->user_history_ops_[n]) {
      case UserHistoryOperation::sum:
        usr_op = MPI_SUM;
        break;
      case UserHistoryOperation::max:
        usr_op = MPI_MAX;
        break;
      case UserHistoryOperation::min:
        usr_op = MPI_MIN;
        break;
    }
    if (Globals::my_rank == 0) {
      MPI_Reduce(MPI_IN_PLACE, usr_hst_data, 1, MPI_ATHENA_REAL, usr_op, 0,
                 MPI_COMM_WORLD);
    } else {
      MPI_Reduce(usr_hst_data, usr_hst_data, 1, MPI_ATHENA_REAL, usr_op, 0,
                 MPI_COMM_WORLD);
    }
  }
#endif

  // only the master rank writes the file
  // create filename: "file_basename" + ".hst".  There is no file number.
  if (Globals::my_rank == 0) {
    std::string fname;
    fname.assign(output_params.file_basename);
    fname.append(".hst");

    // open file for output
    FILE *pfile;
    std::stringstream msg;
    if ((pfile = std::fopen(fname.c_str(),"a")) == nullptr) {
      msg << "### FATAL ERROR in function [OutputType::HistoryFile]" << std::endl
          << "Output file '" << fname << "' could not be opened";
      ATHENA_ERROR(msg);
    }

    // If this is the first output, write header
    if (output_params.file_number == 0) {
      // NEW_OUTPUT_TYPES:

      int iout = 1;
      std::fprintf(pfile,"# Athena++ history data\n"); // descriptor is first line
      std::fprintf(pfile,"# [%d]=time     ", iout++);
      std::fprintf(pfile,"[%d]=dt       ", iout++);
      std::fprintf(pfile,"[%d]=mass     ", iout++);
      std::fprintf(pfile,"[%d]=1-mom    ", iout++);
      std::fprintf(pfile,"[%d]=2-mom    ", iout++);
      std::fprintf(pfile,"[%d]=3-mom    ", iout++);
      std::fprintf(pfile,"[%d]=1-KE     ", iout++);
      std::fprintf(pfile,"[%d]=2-KE     ", iout++);
      std::fprintf(pfile,"[%d]=3-KE     ", iout++);
      for (int n=0; n<pm->nuser_history_output_; n++)
        std::fprintf(pfile,"[%d]=%-8s", iout++,
                     pm->user_history_output_names_[n].c_str());
      std::fprintf(pfile,"\n");                              // terminate line
    }

    // write history variables
    std::fprintf(pfile, output_params.data_format.c_str(), pm->time);
    std::fprintf(pfile, output_params.data_format.c_str(), pm->dt);
    for (int n=0; n<nhistory_output; ++n)
      std::fprintf(pfile, output_params.data_format.c_str(), hst_data[n]);
    std::fprintf(pfile,"\n"); // terminate line
    std::fclose(pfile);
  }

  // increment counters, clean up
  output_params.file_number++;
  output_params.next_time += output_params.dt;
  pin->SetInteger(output_params.block_name, "file_number", output_params.file_number);
  pin->SetReal(output_params.block_name, "next_time", output_params.next_time);
  return;
}
}
