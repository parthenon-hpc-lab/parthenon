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

void HistoryOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin, SimTime *tm,
                                    const SignalHandler::OutputSignal signal) {
  printf("Write history block_name: %s\n", output_params.block_name.c_str());
  // Don't update history log for `output_now` file based outputs.
  if (signal == SignalHandler::OutputSignal::now) {
    return;
  }
  // std::vector<std::string> all_labels = {};
  // std::vector<Real> all_results = {};
  // std::vector<Real> local_results = {};
  // std::vector<int> history_sizes = {}; // TODO(BRR) do we need this or just create
  // labels
  //                                     // when we make pools of everything?
  // std::vector<UserHistoryOperation> history_ops = {};

  std::map<UserHistoryOperation, std::vector<Real>> results;
  std::map<UserHistoryOperation, std::vector<std::string>> labels;
  std::vector<UserHistoryOperation> ops = {
      UserHistoryOperation::sum, UserHistoryOperation::max, UserHistoryOperation::min};
  for (auto &op : ops) {
    results[op] = {};
    labels[op] = {};
  }

  // If "packages" not provided, loop over all packages
  auto &requested_packages = output_params.packages;
  printf("packages size: %i\n", requested_packages.size());
  // bool all_packages = requested_packages.empty();

  // if (requested_packages.empty()) {
  //  for (const auto &pkg : pm->packages.AllPackages()) {
  //  }
  //}

  Dictionary<std::shared_ptr<StateDescriptor>> packages;
  if (requested_packages.empty()) {
    packages = pm->packages.AllPackages();
  } else {
    for (const auto &pkg : pm->packages.AllPackages()) {
      if (std::find(requested_packages.begin(), requested_packages.end(), pkg.first) !=
          requested_packages.end()) {
        packages[pkg.first] = pkg.second;
      }
    }
  }

  // Loop over all packages of the application
  // TODO(BRR) loop over specified packages for this output
  // for (const auto &pkg : pm->packages.AllPackages()) {
  for (const auto &pkg : packages) {

    // if (!all_packages) {
    //  printf("Looking for pkg: %s\n", pkg.first.c_str());
    //  if (std::find(requested_packages.begin(), requested_packages.end(), pkg.first) ==
    //      requested_packages.end()) {
    //    continue;
    //  }
    //}

    // Check if the package has enrolled functions which are stored in the
    // Params under the `hist_param_key` name.
    const auto &params = pkg.second->AllParams();
    if (!params.hasKey(hist_param_key)) {
      continue;
    }
    // TODO(BRR) how to extend this to return std::vector as well?
    const auto &hist_vars = params.Get<HstVar_list>(hist_param_key);

    // Get "base" MeshData, which always exists but may not be populated yet
    auto &md_base = pm->mesh_data.Get();
    // Populated with all blocks
    if (md_base->NumBlocks() == 0) {
      md_base->Set(pm->block_list, pm);
    } else if (md_base->NumBlocks() != pm->block_list.size()) {
      PARTHENON_WARN(
          "Resetting \"base\" MeshData to contain all blocks. This indicates that "
          "the \"base\" MeshData container has been modified elsewhere. Double check "
          "that the modification was intentional and is compatible with this reset.")
      md_base->Set(pm->block_list, pm);
    }

    for (const auto &hist_var : hist_vars) {

      auto result = hist_var.hst_fun(md_base.get());
      results[hist_var.hst_op].push_back(result);
      labels[hist_var.hst_op].push_back(hist_var.label);
      // all_labels.emplace_back(hist_var.label);
    }

    if (!params.hasKey(hist_vec_param_key)) {
      continue;
    }
    const auto &hist_vecs = params.Get<HstVec_list>(hist_vec_param_key);
    for (const auto &hist_vec : hist_vecs) {
      auto result = hist_vec.hst_vec_fun(md_base.get());
      for (int n = 0; n < result.size(); n++) {
        std::string label = hist_vec.label + std::to_string(n);
        results[hist_vec.hst_op].push_back(result[n]);
        labels[hist_vec.hst_op].push_back(label);
      }
    }
  }

  // std::map<UserHistoryOperation, std::vector<Real>> global_results;
  // for (auto &op : ops) {
  //  global_results[op] = std::vector<Real>(results[op].size(), 0.);
  //}

  // Need fence so result is ready prior to MPI call
  Kokkos::fence();

  // const int nresults = local_results.size();
// results.resize(nresults);
#ifdef MPI_PARALLEL
  for (auto &op : ops) {
    if (results[op].size() == 0) {
      continue;
    }

    MPI_Op usr_op;
    if (op == UserHistoryOperation::sum) {
      usr_op = MPI_SUM;
    } else if (op == UserHistoryOperation::max) {
      usr_op = MPI_MAX;
    } else if (op == UserHistoryOperation::min) {
      usr_op = MPI_MIN;
    }

    if (Globals::my_rank == 0) {
      PARTHENON_MPI_CHECK(MPI_Reduce(MPI_IN_PLACE, &(results[op])[0], results[op].size(),
                                     MPI_PARTHENON_REAL, usr_op, 0, MPI_COMM_WORLD));
    } else {
      PARTHENON_MPI_CHECK(MPI_Reduce(&(results[op])[0], &(results[op])[0],
                                     results[op].size(), MPI_PARTHENON_REAL, usr_op, 0,
                                     MPI_COMM_WORLD));
    }

    // results[op] = {};
    // labels[op] = {};
  }
  // MPI_Op usr_op = MPI_SUM; // TODO(BRR) placeholder

  // if (Globals::my_rank == 0) {
  //  PARTHENON_MPI_CHECK(MPI_Reduce(MPI_IN_PLACE, &local_results[0], nresults,
  //                                 MPI_PARTHENON_REAL, usr_op, 0, MPI_COMM_WORLD));
  //} else {
  //  PARTHENON_MPI_CHECK(MPI_Reduce(&local_results[0], &local_results[0], 1,
  //                                 MPI_PARTHENON_REAL, usr_op, 0, MPI_COMM_WORLD));
  //}
#endif // MPI_PARALLEL

  // only the master rank writes the file
  // create filename: "file_basename" + ".hst".  There is no file number.
  if (Globals::my_rank == 0) {
    std::string fname;
    fname.assign(output_params.file_basename);
    fname.append(".out" + std::to_string(output_params.block_number));
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
      // TODO(BRR) optionally overwrite file if this is the first write and not a restart?
      // NEW_OUTPUT_TYPES:

      int iout = 1;
      std::fprintf(pfile, "#  History data\n"); // descriptor is first line
      std::fprintf(pfile, "# [%d]=time     ", iout++);
      std::fprintf(pfile, "[%d]=dt       ", iout++);
      for (auto &op : ops) {
        for (auto &label : labels[op]) {
          std::fprintf(pfile, "[%d]=%-8s", iout++, label.c_str());
        }
      }
      // for (auto &lbl : all_labels) {
      //  std::fprintf(pfile, "[%d]=%-8s", iout++, lbl.c_str());
      //}
      std::fprintf(pfile, "\n"); // terminate line
    }

    // write history variables
    std::fprintf(pfile, output_params.data_format.c_str(), tm->time);
    std::fprintf(pfile, output_params.data_format.c_str(), tm->dt);
    // for (auto &val : all_results) {
    //  std::fprintf(pfile, output_params.data_format.c_str(), val);
    //}
    for (auto &op : ops) {
      for (auto &result : results[op]) {
        std::fprintf(pfile, output_params.data_format.c_str(), result);
      }
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
