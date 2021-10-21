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

#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "coordinates/coordinates.hpp"
#include "defs.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "outputs/outputs.hpp"
#include "parthenon_arrays.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

//----------------------------------------------------------------------------------------
//! \fn void OutputType::HistoryFile()
//  \brief Writes a history file

void HistoryOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin, SimTime *tm) {
  std::vector<std::string> all_labels = {};
  std::vector<Real> all_results = {};

  // Loop over all packages of the application
  for (const auto &pkg : pm->packages.AllPackages()) {
    // Check if the package has enrolled functions which are stored in the
    // Params under the `hist_param_key` name.
    const auto &params = pkg.second->AllParams();
    if (!params.hasKey(hist_param_key)) {
      continue;
    }
    const auto &hist_vars = params.Get<HstVar_list>(hist_param_key);

    for (const auto &hist_var : hist_vars) {
      // Get "base" MeshData, which always exists but may not be populated yet
      auto &md_base = pm->mesh_data.Get();
      // Populated with all blocks
      if (md_base->NumBlocks() == 0) {
        md_base->Set(pm->block_list, "base");
      } else if (md_base->NumBlocks() != pm->block_list.size()) {
        PARTHENON_WARN(
            "Resetting \"base\" MeshData to contain all blocks. This indicates that "
            "the \"base\" MeshData container has been modified elsewhere. Double check "
            "that the modification was intentional and is compatible with this reset.")
        md_base->Set(pm->block_list, "base");
      }
      auto result = hist_var.hst_fun(md_base.get());
#ifdef MPI_PARALLEL
      // need fence so that the result is ready prior to the MPI call
      Kokkos::fence();
      // apply separate chosen operations to each user-defined history output
      MPI_Op usr_op;
      switch (hist_var.hst_op) {
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
        PARTHENON_MPI_CHECK(MPI_Reduce(MPI_IN_PLACE, &result, 1, MPI_PARTHENON_REAL,
                                       usr_op, 0, MPI_COMM_WORLD));
      } else {
        PARTHENON_MPI_CHECK(MPI_Reduce(&result, &result, 1, MPI_PARTHENON_REAL, usr_op, 0,
                                       MPI_COMM_WORLD));
      }
#endif

      all_results.emplace_back(result);
      all_labels.emplace_back(hist_var.label);
    }
  }

  // only the master rank writes the file
  // create filename: "file_basename" + ".hst".  There is no file number.
  if (Globals::my_rank == 0) {
    std::string fname;
    fname.assign(output_params.file_basename);
    fname.append(".hst");

    // open file for output
    FILE *pfile;
    std::stringstream msg;
    if ((pfile = std::fopen(fname.c_str(), "a")) == nullptr) {
      msg << "### FATAL ERROR in function [OutputType::HistoryFile]" << std::endl
          << "Output file '" << fname << "' could not be opened";
      PARTHENON_FAIL(msg);
    }

    // If this is the first output, write header
    if (output_params.file_number == 0) {
      // NEW_OUTPUT_TYPES:

      int iout = 1;
      std::fprintf(pfile, "#  History data\n"); // descriptor is first line
      std::fprintf(pfile, "# [%d]=time     ", iout++);
      std::fprintf(pfile, "[%d]=dt       ", iout++);
      for (auto &lbl : all_labels) {
        std::fprintf(pfile, "[%d]=%-8s", iout++, lbl.c_str());
      }
      std::fprintf(pfile, "\n"); // terminate line
    }

    // write history variables
    std::fprintf(pfile, output_params.data_format.c_str(), tm->time);
    std::fprintf(pfile, output_params.data_format.c_str(), tm->dt);
    for (auto &val : all_results) {
      std::fprintf(pfile, output_params.data_format.c_str(), val);
    }
    std::fprintf(pfile, "\n"); // terminate line
    std::fclose(pfile);
  }

  // advance output parameters
  output_params.file_number++;
  output_params.next_time += output_params.dt;
  pin->SetInteger(output_params.block_name, "file_number", output_params.file_number);
  pin->SetReal(output_params.block_name, "next_time", output_params.next_time);
}

} // namespace parthenon
