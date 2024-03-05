//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020-2024 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2023. Triad National Security, LLC. All rights reserved.
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
// - outputs.cpp, OutputType:LoadOutputData() (below): conditionally add new OutputData
// node to linked list, depending on the user-input 'variable' string.
//
// - parthenon_hdf5.cpp, PHDF5Output::WriteOutputFile(): need to allocate space for the
// new OutputData node as an HDF5 "variable" inside an existing HDF5 "dataset"
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
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>

#include "coordinates/coordinates.hpp"
#include "defs.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "parameter_input.hpp"
#include "parthenon_arrays.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

//----------------------------------------------------------------------------------------
// OutputType constructor

OutputType::OutputType(OutputParameters oparams)
    : output_params(oparams),
      pnext_type(), // Terminate this node in singly linked list with nullptr
      num_vars_(),
      // nested doubly linked list of OutputData:
      pfirst_data_(), // Initialize head node to nullptr
      plast_data_() { // Initialize tail node to nullptr
}

//----------------------------------------------------------------------------------------
// Outputs constructor

Outputs::Outputs(Mesh *pm, ParameterInput *pin, SimTime *tm) {
  pfirst_type_ = nullptr;
  std::stringstream msg;
  InputBlock *pib = pin->pfirst_block;
  OutputType *pnew_type;
  OutputType *plast = pfirst_type_;
  int num_hst_outputs = 0, num_rst_outputs = 0; // number of history and restart outputs

  // loop over input block names.  Find those that start with "parthenon/output", read
  // parameters, and construct singly linked list of OutputTypes.
  while (pib != nullptr) {
    if (pib->block_name.compare(0, 16, "parthenon/output") == 0) {
      OutputParameters op; // define temporary OutputParameters struct

      // extract integer number of output block.  Save name and number
      std::string outn = pib->block_name.substr(16); // 6 because counting starts at 0!
      op.block_number = atoi(outn.c_str());
      op.block_name.assign(pib->block_name);

      Real dt = 0.0; // default value == 0 means that initial data is written by default
      // for temporal drivers, setting dt to tlim ensures a final output is also written
      if (tm != nullptr) {
        dt = pin->GetOrAddReal(op.block_name, "dt", tm->tlim);
      }
      // if this output is "soft-disabled" (negative value) skip processing
      if (dt < 0.0) {
        pib = pib->pnext; // move to next input block name
        continue;
      }
      // set time of last output, time between outputs
      if (tm != nullptr) {
        op.next_time = pin->GetOrAddReal(op.block_name, "next_time", tm->time);
        op.dt = dt;
      }

      // set file number, basename, id, and format
      op.file_number = pin->GetOrAddInteger(op.block_name, "file_number", 0);
      op.file_basename = pin->GetOrAddString("parthenon/job", "problem_id", "parthenon");
      op.file_number_width = pin->GetOrAddInteger(op.block_name, "file_number_width", 5);
      op.file_label_final = pin->GetOrAddBoolean(op.block_name, "use_final_label", true);
      char define_id[10];
      std::snprintf(define_id, sizeof(define_id), "out%d",
                    op.block_number); // default id="outN"
      op.file_id = pin->GetOrAddString(op.block_name, "id", define_id);
      op.file_type = pin->GetString(op.block_name, "file_type");

      // read ghost cell option
      op.include_ghost_zones = pin->GetOrAddBoolean(op.block_name, "ghost_zones", false);

      // read cartesian mapping option
      op.cartesian_vector = false;

      // read single precision output option
      const bool is_hdf5_output = (op.file_type == "rst") || (op.file_type == "hdf5");

      if (is_hdf5_output) {
        op.single_precision_output =
            pin->GetOrAddBoolean(op.block_name, "single_precision_output", false);
        op.sparse_seed_nans =
            pin->GetOrAddBoolean(op.block_name, "sparse_seed_nans", false);
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
        op.packages = pin->GetOrAddVector<std::string>(pib->block_name, "packages", std::vector<std::string>());
      }

      // set output variable and optional data format string used in formatted writes
      if ((op.file_type != "hst") && (op.file_type != "rst") &&
          (op.file_type != "ascent") && (op.file_type != "histogram")) {
        op.variables = pin->GetOrAddVector<std::string>(pib->block_name, "variables",
                                                        std::vector<std::string>());
        // JMM: If the requested var isn't present for a given swarm,
        // it is simply not output.
        op.swarms.clear(); // Not sure this is needed
        if (pin->DoesParameterExist(pib->block_name, "swarms")) {
          std::vector<std::string> swarmnames =
              pin->GetVector<std::string>(pib->block_name, "swarms");
          std::size_t nswarms = swarmnames.size();
          if ((pin->DoesParameterExist(pib->block_name, "swarm_variables")) &&
              (nswarms > 1)) {
            std::stringstream msg;
            msg << "The swarm_variables field is set in the block '" << pib->block_name
                << "' however, there are " << nswarms << " swarms."
                << " All swarms will be assumed to request the vars listed in "
                   "swarm_variables.";
            PARTHENON_WARN(msg);
          }
          for (const auto &swname : swarmnames) {
            if (pin->DoesParameterExist(pib->block_name, "swarm_variables")) {
              auto varnames =
                  pin->GetVector<std::string>(pib->block_name, "swarm_variables");
              op.swarms[swname].insert(varnames.begin(), varnames.end());
            }
            if (pin->DoesParameterExist(pib->block_name, swname + "_variables")) {
              auto varnames =
                  pin->GetVector<std::string>(pib->block_name, swname + "_variables");
              op.swarms[swname].insert(varnames.begin(), varnames.end());
            }
            // Always output x, y, and z for swarms so that they work with vis tools.
            std::vector<std::string> coords = {"x", "y", "z"};
            op.swarms[swname].insert(coords.begin(), coords.end());
          }
        }
      }
      op.data_format = pin->GetOrAddString(op.block_name, "data_format", "%12.5e");
      op.data_format.insert(0, " "); // prepend with blank to separate columns

      // Construct new OutputType according to file format
      // NEW_OUTPUT_TYPES: Add block to construct new types here
      if (op.file_type == "hst") {
        pnew_type = new HistoryOutput(op);
        num_hst_outputs++;
      } else if (op.file_type == "vtk") {
        pnew_type = new VTKOutput(op);
      } else if (op.file_type == "ascent") {
        pnew_type = new AscentOutput(op);
      } else if (op.file_type == "histogram") {
#ifdef ENABLE_HDF5
        pnew_type = new HistogramOutput(op, pin);
#else
        msg << "### FATAL ERROR in Outputs constructor" << std::endl
            << "Executable not configured for HDF5 outputs, but HDF5 file format "
            << "is requested in output/restart block '" << op.block_name << "'. "
            << "You can disable this block without deleting it by setting a dt < 0."
            << std::endl;
        PARTHENON_FAIL(msg);
#endif // ifdef ENABLE_HDF5
      } else if (is_hdf5_output) {
        const bool restart = (op.file_type == "rst");
        if (restart) {
          num_rst_outputs++;
        }
#ifdef ENABLE_HDF5
        op.write_xdmf = pin->GetOrAddBoolean(op.block_name, "write_xdmf", true);
        pnew_type = new PHDF5Output(op, restart);
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

      // Append type as tail node in singly linked list
      if (pfirst_type_ == nullptr) {
        pfirst_type_ = pnew_type;
      } else {
        plast->pnext_type = pnew_type;
      }
      plast = pnew_type;
    }
    pib = pib->pnext; // move to next input block name
  }

  // check there were no more than one restart file requested
  if (num_rst_outputs > 1) {
    msg << "### FATAL ERROR in Outputs constructor" << std::endl
        << "More than one restart output block detected in input file"
        << std::endl;
    PARTHENON_FAIL(msg);
  }

  // Move restarts to the tail end of the OutputType list, so file counters for other
  // output types are up-to-date in restart file
  int pos = 0, found = 0;
  OutputType *pot = pfirst_type_;
  OutputType *prst = pot;
  while (pot != nullptr) {
    if (pot->output_params.file_type == "rst") {
      prst = pot;
      found = 1;
      if (pot->pnext_type == nullptr) found = 2;
      break;
    }
    pos++;
    pot = pot->pnext_type;
  }
  if (found == 1) {
    // remove the restarting block
    pot = pfirst_type_;
    if (pos == 0) { // head node/first block
      pfirst_type_ = pfirst_type_->pnext_type;
    } else {
      for (int j = 0; j < pos - 1; j++) // seek the list
        pot = pot->pnext_type;
      pot->pnext_type = prst->pnext_type; // remove it
    }
    while (pot->pnext_type != nullptr)
      pot = pot->pnext_type; // find the tail node
    prst->pnext_type = nullptr;
    pot->pnext_type = prst;
  }
  // if found == 2, do nothing; it's already at the tail node/end of the list
} // namespace parthenon

// destructor - iterates through singly linked list of OutputTypes and deletes nodes

Outputs::~Outputs() {
  OutputType *ptype = pfirst_type_;
  while (ptype != nullptr) {
    OutputType *ptype_old = ptype;
    ptype = ptype->pnext_type;
    delete ptype_old;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void OutputType::LoadOutputData(MeshBlock *pmb)
//  \brief Create doubly linked list of OutputData's containing requested variables

void OutputType::LoadOutputData(MeshBlock *pmb) {
  throw std::runtime_error(std::string(__func__) + " is not implemented");
}

//----------------------------------------------------------------------------------------
//! \fn void OutputData::AppendOutputDataNode(OutputData *pod)
//  \brief

void OutputType::AppendOutputDataNode(OutputData *pnew_data) {
  if (pfirst_data_ == nullptr) {
    pfirst_data_ = pnew_data;
  } else {
    pnew_data->pprev = plast_data_;
    plast_data_->pnext = pnew_data;
  }
  // make the input node the new tail node of the doubly linked list
  plast_data_ = pnew_data;
}

//----------------------------------------------------------------------------------------
//! \fn void OutputData::ReplaceOutputDataNode()
//  \brief

void OutputType::ReplaceOutputDataNode(OutputData *pold, OutputData *pnew) {
  if (pold == pfirst_data_) {
    pfirst_data_ = pnew;
    if (pold->pnext != nullptr) { // there is another node in the list
      pnew->pnext = pold->pnext;
      pnew->pnext->pprev = pnew;
    } else { // there is only one node in the list
      plast_data_ = pnew;
    }
  } else if (pold == plast_data_) {
    plast_data_ = pnew;
    pnew->pprev = pold->pprev;
    pnew->pprev->pnext = pnew;
  } else {
    pnew->pnext = pold->pnext;
    pnew->pprev = pold->pprev;
    pnew->pprev->pnext = pnew;
    pnew->pnext->pprev = pnew;
  }
  delete pold;
}

//----------------------------------------------------------------------------------------
//! \fn void OutputData::ClearOutputData()
//  \brief

void OutputType::ClearOutputData() {
  OutputData *pdata = pfirst_data_;
  while (pdata != nullptr) {
    OutputData *pdata_old = pdata;
    pdata = pdata->pnext;
    delete pdata_old;
  }
  // reset pointers to head and tail nodes of doubly linked list:
  pfirst_data_ = nullptr;
  plast_data_ = nullptr;
}

//----------------------------------------------------------------------------------------
//! \fn void Outputs::MakeOutputs(Mesh *pm, ParameterInput *pin, bool wtflag)
//  \brief scans through singly linked list of OutputTypes and makes any outputs needed.

void Outputs::MakeOutputs(Mesh *pm, ParameterInput *pin, SimTime *tm,
                          const SignalHandler::OutputSignal signal) {
  PARTHENON_INSTRUMENT
  bool first = true;
  OutputType *ptype = pfirst_type_;
  while (ptype != nullptr) {
    if ((tm == nullptr) ||
        ((ptype->output_params.dt >= 0.0) &&
         ((tm->ncycle == 0) || (tm->time >= ptype->output_params.next_time) ||
          (tm->time >= tm->tlim) || (signal != SignalHandler::OutputSignal::none)))) {
      if (first && ptype->output_params.file_type != "hst") {
        pm->ApplyUserWorkBeforeOutput(pm, pin, *tm);
        first = false;
      }
      ptype->WriteOutputFile(pm, pin, tm, signal);
    }
    ptype = ptype->pnext_type; // move to next OutputType node in singly linked list
  }
}

} // namespace parthenon
