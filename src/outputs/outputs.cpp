//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020-2025 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2025. Triad National Security, LLC. All rights reserved.
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
//! \file outputs.cpp
//  \brief implements functions for Parthenon outputs
//
// The number and types of outputs are all controlled by the number and values of
// parameters specified in <output[n]> blocks in the input file.  Each output block must
// be labelled by a unique integer "n".  Following the convention of the parser
// implemented in the ParameterInput class, a second output block with the same integer
// "n" of an earlier block will silently overwrite the values read by the first block. The
// numbering of the output blocks does not need to be consecutive, and blocks may appear
// in any order in the input file.  Moreover, unlike the C version of Athena, the total
// number of <output[n]> blocks does not need to be specified -- in Athena++ a new output
// type will be created for each and every <output[n]> block in the input file.
//
// Each <output[n]> block will result in a new node being created in a linked list of
// OutputType stored in the Outputs class.  During a simulation, outputs are made when
// the simulation time satisfies the criteria implemented in the MakeOutputs() function.
//
// To implement a new output type, write a new OutputType derived class, and construct
// an object of this class in the Outputs constructor at the location indicated by the
// comment text: 'NEW_OUTPUT_TYPES'. Current summary:
// -----------------------------------
// - outputs.cpp
//
// - parthenon_hdf5.cpp, PHDF5Output::WriteOutputFile():
// (cell-centered vs. face-centered data).
//
// - history.cpp: Add the relevant history quantity to your package
// -----------------------------------
//
// HDF5 note: packing gas velocity into the "prim" HDF5 dataset will cause VisIt to treat
// the 3x components as independent scalars instead of a physical vector, unlike how it
// treats .vtk velocity output from Athena++. The workaround is to import the
// vis/visit/*.xml expressions file, which can pack these HDF5 scalars into a vector.
//========================================================================================

#include "outputs/outputs.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "coordinates/coordinates.hpp"
#include "defs.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "outputs/output_parameters.hpp"
#include "pack/swarm_default_names.hpp"
#include "parameter_input.hpp"
#include "parthenon_arrays.hpp"
#include "utils/error_checking.hpp"
#include "utils/utils.hpp"

namespace parthenon {

//----------------------------------------------------------------------------------------
// OutputType constructor

OutputType::OutputType(OutputParameters oparams) : output_params(oparams), num_vars_() {}

//----------------------------------------------------------------------------------------
// Outputs constructor

Outputs::Outputs(Mesh *pm, ParameterInput *pin, SimTime *tm) {
  std::stringstream msg;
  // We should only have at most one each of these output types. Count
  // them so we can raise an error.
  int num_rst_outputs = 0;
  int num_core_outputs = 0;

  // track restart outputs separately. We'll combine at the end.
  std::vector<std::shared_ptr<OutputType>> restart_outputs;

  // loop over "parthenon/output" blocks located in params
  auto pkg = pm->packages.Get("Outputs");
  // loop over input block names.  Find those that start with "parthenon/output", read
  // parameters, and construct a vector of output types.
  // PG: It could be discussed if we should work based on a vector that is initialized in
  // the outpus packages (as before), but I don't think it's bad practice to work on
  // `pinput` again here as we're actually processing (potentially even modifying)
  // `pinput`.
  for (InputBlock *pib = pin->pfirst_block; pib != nullptr; pib = pib->pnext) {
    if (pib->block_name.compare(0, 16, "parthenon/output") != 0) {
      continue;
    }
    std::shared_ptr<OutputType> pnew_type; // the new output we will create
    bool restart = false;                  // we track restart outputs separately so we
                                           // need this temp variable to check
    OutputParameters op;                   // define temporary OutputParameters struct
    op.block_name = pib->block_name;
    const auto outn_str = pib->block_name.substr(16); // 16 because counting starts at 0!
    op.block_number = atoi(outn_str.c_str());
    auto *pfile_number = pkg->MutableParam<int>(outn_str + "/file_number");
    auto *plast_time = pkg->MutableParam<Real>(outn_str + "/last_time");
    auto *plast_n = pkg->MutableParam<int>(outn_str + "/last_n");

    Real dt = 0.0; // default value == 0 means that initial data is written by default
    int dn = -1;
    if (tm != nullptr) {
      dn = pin->GetOrAddInteger(op.block_name, "dn", -1, "output cadence in cycles");

      // If this is a dn controlled output (dn >= 0), soft disable dt based triggering
      // (-> dt = -1), otherwise setting dt to tlim ensures a final output is also
      // written for temporal drivers.
      const auto tlim = dn >= 0 ? -1.0 : tm->tlim;
      dt =
          pin->GetOrAddReal(op.block_name, "dt", tlim, "output cadence in physical time");
    }
    // if this output is "soft-disabled" (negative value) skip processing
    if (dt < 0.0 && dn < 0) {
      continue;
    }

    // JMM: Backwards compatibility hack. Don't allow this unless
    // we're restarting from a legacy file format.
    if (parthenon::Globals::is_restart) {
      bool next_time_exists = pin->DoesParameterExist(op.block_name, "next_time");
      bool next_n_exists = pin->DoesParameterExist(op.block_name, "next_n");
      if (next_time_exists) {
        Real next_time = pin->GetReal(op.block_name, "next_time");
        *plast_time = dt < 0 ? 0.0 : next_time - dt;
        pin->RemoveParameter(op.block_name, "next_time");
      }
      if (next_n_exists) {
        int next_n = pin->GetInteger(op.block_name, "next_n");

        *plast_n = dn < 0 ? 0 : next_n - dn;
        pin->RemoveParameter(op.block_name, "next_n");
      }
      if (next_time_exists || next_n_exists) {
        *pfile_number = pin->GetOrAddInteger(op.block_name, "file_number", 0);
        pin->RemoveParameter(op.block_name, "file_number");
      }
    }

    PARTHENON_REQUIRE_THROWS(!(dt >= 0.0 && dn >= 0),
                             "dt and dn are enabled for the same output block, which "
                             "is not supported. Please set at most one value >= 0.");

    // set time of last output, time between outputs
    op.last_time = *plast_time;
    op.last_n = *plast_n;
    if (tm != nullptr) {
      op.dt = dt;
      op.dn = dn;
      /*
        JMM: Set next time/iteration to output. At startup, the
        process looks something like this:
        1. simulation starts at start_time
        2. Outputs package sets last_output to numeric_limits<>::lowest
        3. In Outputs consturctor, we detect this dummy value and set
        the next output to now
        4. An initial dump is created
        5. When these vars are next updated, next output will be start_time + dt

        Alternatively, we could have set last_output output to
        start_time - dt in the outputs package rather than use a
        signaling number. However, I think the flag is less fraught.
      */
      if (dt >= 0) {
        // TODO(JMM): Should this be a check for Globals::is_restart instead?
        if (op.last_time > std::numeric_limits<Real>::lowest()) {
          op.next_time = op.last_time + dt;
        } else {
          op.next_time = tm->time;
        }
      }
      if (dn >= 0) {
        // TODO(JMM): Should this be a check for Globals::is_restart instead?
        if (op.last_n > std::numeric_limits<int>::lowest()) {
          op.next_n = op.last_n + dn;
        } else {
          op.next_n = tm->ncycle;
        }
      }
    }

    // set file number, basename, id, and format
    op.file_number = std::max(*pfile_number, 0);
    op.file_basename = pin->GetOrAddString("parthenon/job", "problem_id", "parthenon",
                                           "prefix for output files");
    op.file_number_width = pin->GetOrAddInteger(op.block_name, "file_number_width", 5);
    op.file_label_final = pin->GetOrAddBoolean(
        op.block_name, "use_final_label", true,
        "final output will use the word final instead of a number for its index");
    char define_id[10];
    std::snprintf(define_id, sizeof(define_id), "out%d",
                  op.block_number); // default id="outN"
    op.file_id = pin->GetOrAddString(op.block_name, "id", define_id);
    op.file_type = pin->GetString(op.block_name, "file_type", "output type");

    // read ghost cell option
    op.include_ghost_zones = pin->GetOrAddBoolean(
        op.block_name, "ghost_zones", false, "whether or not ghost zones are output");

    // read cartesian mapping option
    op.cartesian_vector = false;

    op.analysis_flag = pin->GetOrAddBoolean(op.block_name, "analysis_output", false);

    // read single precision output option
    const bool is_hdf5_output = (op.file_type == "rst") || (op.file_type == "hdf5") ||
                                (op.file_type == "corehdf5");

    if (is_hdf5_output) {
      op.single_precision_output =
          pin->GetOrAddBoolean(op.block_name, "single_precision_output", false);
      op.sparse_seed_nans =
          pin->GetOrAddBoolean(op.block_name, "sparse_seed_nans", false,
                               "write non-allocated sparse data as NaN");
      op.meshdata_name = pin->GetOrAddString(op.block_name, "meshdata_name", "base",
                                             "which meshdata object to write from");
    } else {
      op.single_precision_output = false;
      op.sparse_seed_nans = false;

      if (pin->DoesParameterExist(op.block_name, "single_precision_output")) {
        std::stringstream warn;
        warn << "Output option single_precision_output only applies to "
                "HDF5 outputs or restarts. Ignoring it for output block '"
             << op.block_name << "'";
        PARTHENON_WARN(warn);
      }
    }

    if (is_hdf5_output) {
      int default_compression_level = 5;
#ifdef PARTHENON_DISABLE_HDF5_COMPRESSION
      default_compression_level = 0;
#endif

      op.hdf5_compression_level = pin->GetOrAddInteger(
          op.block_name, "hdf5_compression_level", default_compression_level);

#ifdef PARTHENON_DISABLE_HDF5_COMPRESSION
      if (op.hdf5_compression_level != 0) {
        std::stringstream err;
        err << "HDF5 compression requested for output block '" << op.block_name
            << "', but HDF5 compression is disabled";
        PARTHENON_THROW(err)
      }
#endif
    } else {
      op.hdf5_compression_level = 0;

      if (pin->DoesParameterExist(op.block_name, "hdf5_compression_level")) {
        std::stringstream warn;
        warn << "Output option hdf5_compression_level only applies to "
                "HDF5 outputs or restarts. Ignoring it for output block '"
             << op.block_name << "'";
        PARTHENON_WARN(warn);
      }
    }

    if (op.file_type == "hst") {
      // Do not use GetOrAddVector because it will pollute the input parameters for
      // restarts
      if (pin->DoesParameterExist(op.block_name, "packages")) {
        op.packages = pin->GetVector<std::string>(op.block_name, "packages");
      } else {
        op.packages = std::vector<std::string>();
      }
    }

    // set output variable and optional data format string used in formatted writes
    if ((op.file_type != "hst") && (op.file_type != "rst") &&
        (op.file_type != "corehdf5") && (op.file_type != "ascent") &&
        (op.file_type != "histogram") && op.file_type != "user") {
      // Do not use GetOrAddVector because it will pollute the input parameters for
      // restarts
      if (pin->DoesParameterExist(op.block_name, "variables")) {
        op.variables = pin->GetVector<std::string>(op.block_name, "variables");
      } else {
        op.variables = std::vector<std::string>();
      }
      // JMM: If the requested var isn't present for a given swarm,
      // it is simply not output.
      op.swarms.clear(); // Not sure this is needed
      if (pin->DoesParameterExist(op.block_name, "swarms")) {
        std::vector<std::string> swarmnames =
            pin->GetVector<std::string>(op.block_name, "swarms", "swarms to output");
        std::size_t nswarms = swarmnames.size();
        if ((pin->DoesParameterExist(op.block_name, "swarm_variables")) &&
            (nswarms > 1)) {
          std::stringstream msg;
          msg << "The swarm_variables field is set in the block '" << op.block_name
              << "' however, there are " << nswarms << " swarms."
              << " All swarms will be assumed to request the vars listed in "
                 "swarm_variables.";
          PARTHENON_WARN(msg);
        }
        for (const auto &swname : swarmnames) {
          if (pin->DoesParameterExist(op.block_name, "swarm_variables")) {
            auto varnames =
                pin->GetVector<std::string>(op.block_name, "swarm_variables",
                                            "swarm variables to output for all swarms");
            op.swarms[swname].insert(varnames.begin(), varnames.end());
          }
          if (pin->DoesParameterExist(op.block_name, swname + "_variables")) {
            auto varnames = pin->GetVector<std::string>(
                op.block_name, swname + "_variables",
                "swarm variables to output for a specific swarm");
            op.swarms[swname].insert(varnames.begin(), varnames.end());
          }
          // Always output id, x, y, and z for swarms so that they work with vis tools.
          // Note, it's fine to add the id by default (even though it might not actually
          // exist) because only variables that do exists are actually being written.
          std::vector<std::string> coords = {
              swarm_position::id::name(), swarm_position::x::name(),
              swarm_position::y::name(), swarm_position::z::name()};
          op.swarms[swname].insert(coords.begin(), coords.end());
        }
      }
    }
    op.data_format = pin->GetOrAddString(op.block_name, "data_format", "%12.5e");
    op.data_format.insert(0, " "); // prepend with blank to separate columns

    // Construct new OutputType according to file format
    // NEW_OUTPUT_TYPES: Add block to construct new types here
    if (op.file_type == "hst") {
      pnew_type = std::make_shared<HistoryOutput>(op);
    } else if (op.file_type == "ascent") {
      pnew_type = std::make_shared<AscentOutput>(op);
    } else if (op.file_type == "histogram") {
#ifdef ENABLE_HDF5
      pnew_type = std::make_shared<HistogramOutput>(op, pin);
#else
      msg << "### FATAL ERROR in Outputs constructor" << std::endl
          << "Executable not configured for HDF5 outputs, but HDF5 file format "
          << "is requested in output/restart block '" << op.block_name << "'. "
          << "You can disable this block without deleting it by setting a dt < 0."
          << std::endl;
      PARTHENON_FAIL(msg);
#endif // ifdef ENABLE_HDF5
    } else if (op.file_type == "user") {
      pnew_type = std::make_shared<UserOutput>(op);
    } else if (is_hdf5_output) {
      restart = (op.file_type == "rst");
      const bool coredump = (op.file_type == "corehdf5");
      if (restart) {
        num_rst_outputs++;
      }
      if (coredump) {
        num_core_outputs++;
      }
#ifdef ENABLE_HDF5
      op.write_xdmf = pin->GetOrAddBoolean(op.block_name, "write_xdmf", true);
      op.write_swarm_xdmf =
          pin->GetOrAddBoolean(op.block_name, "write_swarm_xdmf", false);
      pnew_type = std::make_shared<PHDF5Output>(
          op, restart ? DumpOutputMode::RESTART
                      : (coredump ? DumpOutputMode::CORE : DumpOutputMode::DUMP));
#else
      msg << "### FATAL ERROR in Outputs constructor" << std::endl
          << "Executable not configured for HDF5 outputs, but HDF5 file format "
          << "is requested in output/restart block '" << op.block_name << "'. "
          << "You can disable this block without deleting it by setting a dt < 0."
          << std::endl;
      PARTHENON_FAIL(msg);
#endif // ifdef ENABLE_HDF5
    } else {
      msg << "### FATAL ERROR in Outputs constructor" << std::endl
          << "Unrecognized file format = '" << op.file_type << "' in output block '"
          << op.block_name << "'" << std::endl;
      PARTHENON_FAIL(msg);
    }

    // Append type
    if (restart) {
      restart_outputs.push_back(pnew_type);
    } else {
      output_types_.push_back(pnew_type);
    }
  }
  // check there were no more than one restart file requested
  if (num_rst_outputs > 1) {
    msg << "### FATAL ERROR in Outputs constructor" << std::endl
        << "More than one restart output block detected in input file" << std::endl;
    PARTHENON_FAIL(msg);
  }
  if (num_core_outputs > 1) {
    msg << "### FATAL ERROR in Outputs constructor\n"
        << "More than one corehdf5 output block detected in input file" << std::endl;
    PARTHENON_FAIL(msg);
  }

  // Move restarts to the tail end of the OutputType list, so file counters for other
  // output types are up-to-date in restart file
  output_types_.insert(output_types_.end(), restart_outputs.begin(),
                       restart_outputs.end());
}

//----------------------------------------------------------------------------------------
//! \fn void Outputs::MakeOutputs(Mesh *pm, ParameterInput *pin, bool wtflag)
//  \brief scans through singly linked list of OutputTypes and makes any outputs needed.

void Outputs::MakeOutputs(Mesh *pm, ParameterInput *pin, SimTime *tm,
                          const SignalHandler::OutputSignal signal) {
  PARTHENON_INSTRUMENT
  bool first = true;
  for (auto ptype : output_types_) {
    if ((tm == nullptr) ||
        // output is not soft disabled and
        (((ptype->output_params.dt >= 0.0) || (ptype->output_params.dn >= 0)) &&
         // either dump initial data
         ((tm->ncycle == 0) ||
          //  or by triggering time or cycle based conditions
          ((ptype->output_params.dt >= 0.0) &&
           ((tm->time >= ptype->output_params.next_time) ||
            (tm->tlim > 0.0 && tm->time >= tm->tlim))) ||
          ((ptype->output_params.dn >= 0) &&
           ((tm->ncycle >= ptype->output_params.next_n) ||
            (tm->nlim > 0 && tm->ncycle >= tm->nlim))) ||
          // or by manual triggers
          (signal == SignalHandler::OutputSignal::now) ||
          (signal == SignalHandler::OutputSignal::final) ||
          (signal == SignalHandler::OutputSignal::analysis &&
           ptype->output_params.analysis_flag)))) {
      if (first && ptype->output_params.file_type != "hst") {
        pm->ApplyUserWorkBeforeOutput(pm, pin, *tm);
        for (const auto &pkg : pm->packages.AllPackages()) {
          pkg.second->UserWorkBeforeOutput(pm, pin, *tm);
        }
        first = false;
      }
      if (ptype->output_params.file_type == "rst") {
        pm->ApplyUserWorkBeforeRestartOutput(pm, pin, *tm, &(ptype->output_params));
        for (const auto &pkg : pm->packages.AllPackages()) {
          pkg.second->UserWorkBeforeRestartOutput(pm, pin, *tm, &(ptype->output_params));
        }
      }
      ptype->WriteOutputFile(pm, pin, tm, signal);
    }
  }
}

void OutputType::UpdateNextOutput_(Mesh *pm, SimTime *tm) {
  output_params.file_number++;
  auto pkg = pm->packages.Get("Outputs");
  const auto outn_str = std::to_string(output_params.block_number);
  auto *pfile_number = pkg->MutableParam<int>(outn_str + "/file_number");
  auto *plast_time = pkg->MutableParam<Real>(outn_str + "/last_time");
  auto *plast_n = pkg->MutableParam<int>(outn_str + "/last_n");
  *pfile_number = output_params.file_number;
  if (tm != nullptr) {
    // JMM: Do NOT use the current time to update these, as that can
    // cause drift because timestep is not guaranteed to align with
    // desired output time. Instead set last time to previous next
    // time.
    output_params.last_n = output_params.next_n;
    output_params.last_time = output_params.next_time;
    *plast_n = output_params.last_n;
    *plast_time = output_params.last_time;
    if (output_params.dt > 0.0) {
      output_params.next_time += output_params.dt;
    }
    if (output_params.dn > 0) {
      output_params.next_n += output_params.dn;
    }
  }
}

} // namespace parthenon
