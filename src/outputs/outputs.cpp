//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2022. Triad National Security, LLC. All rights reserved.
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
//  \brief implements functions for Athena++ outputs
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
// Required parameters that must be specified in an <output[n]> block are:
//   - variable     = cons,prim,D,d,E,e,m,m1,m2,m3,v,v1=vx,v2=vy,v3=vz,p,
//                    bcc,bcc1,bcc2,bcc3,b,b1,b2,b3,phi,uov
//   - file_type    = rst,tab,vtk,hst,hdf5
//   - dt           = problem time between outputs
//
// EXAMPLE of an <output[n]> block for a VTK dump:
//   <output3>
//   file_type   = tab       # Tabular data dump
//   variable    = prim      # variables to be output
//   data_format = %12.5e    # Optional data format string
//   dt          = 0.01      # time increment between outputs
//   x2_slice    = 0.0       # slice in x2
//   x3_slice    = 0.0       # slice in x3
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
// node to linked list, depending on the user-input 'variable' string. Provide direction
// on how to slice a possible 4D source ParArrayND into separate 3D arrays; automatically
// enrolls quantity in vtk.cpp, formatted_table.cpp outputs.

// - parthenon_hdf5.cpp, PHDF5Output::WriteOutputFile(): need to allocate space for the
// new OutputData node as an HDF5 "variable" inside an existing HDF5 "dataset"
// (cell-centered vs. face-centered data).

// - mesh/meshblock.cpp, MeshBlock restart constructor: memcpy quantity (IN THE SAME ORDER
// AS THE VARIABLES ARE WRITTEN IN restart.cpp) from the loaded .rst file to the
// MeshBlock's appropriate physics member object

// - history.cpp, HistoryOutput::WriteOutputFile() (3x places): 1) modify NHISTORY_VARS
// macro so that the size of data_sum[] can accommodate the new physics, when active.
// 2) Compute volume-weighted data_sum[i] for the new quantity + etc. factors
// 3) Provide short string to serve as the column header description of new quantity
// -----------------------------------

// HDF5 note: packing gas velocity into the "prim" HDF5 dataset will cause VisIt to treat
// the 3x components as independent scalars instead of a physical vector, unlike how it
// treats .vtk velocity output from Athena++. The workaround is to import the
// vis/visit/*.xml expressions file, which can pack these HDF5 scalars into a vector.
//========================================================================================

#include "outputs/outputs.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "coordinates/coordinates.hpp"
#include "defs.hpp"
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

      // read slicing options.  Check that slice is within mesh
      if (pin->DoesParameterExist(op.block_name, "x1_slice")) {
        Real x1 = pin->GetReal(op.block_name, "x1_slice");
        if (x1 >= pm->mesh_size.x1min && x1 < pm->mesh_size.x1max) {
          op.x1_slice = x1;
          op.output_slicex1 = true;
        } else {
          msg << "### FATAL ERROR in Outputs constructor" << std::endl
              << "Slice at x1=" << x1 << " in output block '" << op.block_name
              << "' is out of range of Mesh" << std::endl;
          PARTHENON_FAIL(msg);
        }
      }

      if (pin->DoesParameterExist(op.block_name, "x2_slice")) {
        Real x2 = pin->GetReal(op.block_name, "x2_slice");
        if (x2 >= pm->mesh_size.x2min && x2 < pm->mesh_size.x2max) {
          op.x2_slice = x2;
          op.output_slicex2 = true;
        } else {
          msg << "### FATAL ERROR in Outputs constructor" << std::endl
              << "Slice at x2=" << x2 << " in output block '" << op.block_name
              << "' is out of range of Mesh" << std::endl;
          PARTHENON_FAIL(msg);
        }
      }

      if (pin->DoesParameterExist(op.block_name, "x3_slice")) {
        Real x3 = pin->GetReal(op.block_name, "x3_slice");
        if (x3 >= pm->mesh_size.x3min && x3 < pm->mesh_size.x3max) {
          op.x3_slice = x3;
          op.output_slicex3 = true;
        } else {
          msg << "### FATAL ERROR in Outputs constructor" << std::endl
              << "Slice at x3=" << x3 << " in output block '" << op.block_name
              << "' is out of range of Mesh" << std::endl;
          PARTHENON_FAIL(msg);
        }
      }

      // read sum options.  Check for conflicts with slicing.
      op.output_sumx1 = pin->GetOrAddBoolean(op.block_name, "x1_sum", false);
      if ((op.output_slicex1) && (op.output_sumx1)) {
        msg << "### FATAL ERROR in Outputs constructor" << std::endl
            << "Cannot request both slice and sum along x1-direction"
            << " in output block '" << op.block_name << "'" << std::endl;
        PARTHENON_FAIL(msg);
      }
      op.output_sumx2 = pin->GetOrAddBoolean(op.block_name, "x2_sum", false);
      if ((op.output_slicex2) && (op.output_sumx2)) {
        msg << "### FATAL ERROR in Outputs constructor" << std::endl
            << "Cannot request both slice and sum along x2-direction"
            << " in output block '" << op.block_name << "'" << std::endl;
        PARTHENON_FAIL(msg);
      }
      op.output_sumx3 = pin->GetOrAddBoolean(op.block_name, "x3_sum", false);
      if ((op.output_slicex3) && (op.output_sumx3)) {
        msg << "### FATAL ERROR in Outputs constructor" << std::endl
            << "Cannot request both slice and sum along x3-direction"
            << " in output block '" << op.block_name << "'" << std::endl;
        PARTHENON_FAIL(msg);
      }

      // read ghost cell option
      op.include_ghost_zones = pin->GetOrAddBoolean(op.block_name, "ghost_zones", false);

      // read cartesian mapping option
      op.cartesian_vector = false;

      // read single precision output option
      const bool is_hdf5_output =
          (op.file_type == "rst") || (op.file_type == "ath5") || (op.file_type == "hdf5");

      if (is_hdf5_output) {
        op.single_precision_output =
            pin->GetOrAddBoolean(op.block_name, "single_precision_output", false);
      } else {
        op.single_precision_output = false;

        if (pin->DoesParameterExist(op.block_name, "single_precision_output")) {
          std::stringstream warn;
          warn << "### WARNING Output option single_precision_output only applies to "
                  "HDF5 outputs or restarts. Ignoring it for output block '"
               << op.block_name << "'" << std::endl;
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
          err << "### ERROR: HDF5 compression requested for output block '"
              << op.block_name << "', but HDF5 compression is disabled" << std::endl;
          PARTHENON_THROW(err)
        }
#endif
      } else {
        op.hdf5_compression_level = 0;

        if (pin->DoesParameterExist(op.block_name, "hdf5_compression_level")) {
          std::stringstream warn;
          warn << "### WARNING Output option hdf5_compression_level only applies to "
                  "HDF5 outputs or restarts. Ignoring it for output block '"
               << op.block_name << "'" << std::endl;
          PARTHENON_WARN(warn);
        }
      }

      // set output variable and optional data format string used in formatted writes
      if ((op.file_type != "hst") && (op.file_type != "rst")) {
        op.variables = pin->GetVector<std::string>(pib->block_name, "variables");
      }
      op.data_format = pin->GetOrAddString(op.block_name, "data_format", "%12.5e");
      op.data_format.insert(0, " "); // prepend with blank to separate columns

      // Construct new OutputType according to file format
      // NEW_OUTPUT_TYPES: Add block to construct new types here
      if (op.file_type == "hst") {
        pnew_type = new HistoryOutput(op);
        num_hst_outputs++;
      } else if (op.file_type == "tab") {
        pnew_type = new FormattedTableOutput(op);
      } else if (op.file_type == "vtk") {
        pnew_type = new VTKOutput(op);
      } else if (is_hdf5_output) {
        const bool restart = (op.file_type == "rst");
        if (restart) {
          num_rst_outputs++;
        }
#ifdef ENABLE_HDF5
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

  // check there were no more than one history or restart files requested
  if (num_hst_outputs > 1 || num_rst_outputs > 1) {
    msg << "### FATAL ERROR in Outputs constructor" << std::endl
        << "More than one history or restart output block detected in input file"
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
  Kokkos::Profiling::pushRegion("MakeOutputs");
  bool first = true;
  OutputType *ptype = pfirst_type_;
  while (ptype != nullptr) {
    if ((tm == nullptr) ||
        ((ptype->output_params.dt >= 0.0) &&
         ((tm->ncycle == 0) || (tm->time >= ptype->output_params.next_time) ||
          (tm->time >= tm->tlim) || (signal != SignalHandler::OutputSignal::none)))) {
      if (first && ptype->output_params.file_type != "hst") {
        pm->ApplyUserWorkBeforeOutput(pin);
        first = false;
      }
      ptype->WriteOutputFile(pm, pin, tm, signal);
    }
    ptype = ptype->pnext_type; // move to next OutputType node in singly linked list
  }
  Kokkos::Profiling::popRegion(); // MakeOutputs
}

//----------------------------------------------------------------------------------------
//! \fn void OutputType::TransformOutputData(MeshBlock *pmb)
//  \brief Calls sum and slice functions on each direction in turn, in order to allow
//  mulitple operations performed on the same data set

bool OutputType::TransformOutputData(MeshBlock *pmb) {
  bool flag = true;
  if (output_params.output_slicex3) {
    bool ret = SliceOutputData(pmb, 3);
    if (!ret) flag = false;
  }
  if (output_params.output_slicex2) {
    bool ret = SliceOutputData(pmb, 2);
    if (!ret) flag = false;
  }
  if (output_params.output_slicex1) {
    bool ret = SliceOutputData(pmb, 1);
    if (!ret) flag = false;
  }
  if (output_params.output_sumx3) {
    SumOutputData(pmb, 3);
  }
  if (output_params.output_sumx2) {
    SumOutputData(pmb, 2);
  }
  if (output_params.output_sumx1) {
    SumOutputData(pmb, 1);
  }
  return flag;
}

//----------------------------------------------------------------------------------------
//! \fn bool OutputType::SliceOutputData(MeshBlock *pmb, int dim)
//  \brief perform data slicing and update the data list

bool OutputType::SliceOutputData(MeshBlock *pmb, int dim) {
  int islice(0), jslice(0), kslice(0);

  // Compute i,j,k indices of slice; check if in range of data in this block
  const IndexDomain interior = IndexDomain::interior;
  if (dim == 1) {
    if (output_params.x1_slice >= pmb->block_size.x1min &&
        output_params.x1_slice < pmb->block_size.x1max) {
      for (int i = pmb->cellbounds.is(interior) + 1;
           i <= pmb->cellbounds.ie(interior) + 1; ++i) {
        if (pmb->coords.Xf<1, 1>(i) > output_params.x1_slice) {
          islice = i - 1;
          output_params.islice = islice;
          break;
        }
      }
    } else {
      return false;
    }
  } else if (dim == 2) {
    if (output_params.x2_slice >= pmb->block_size.x2min &&
        output_params.x2_slice < pmb->block_size.x2max) {
      for (int j = pmb->cellbounds.js(interior) + 1;
           j <= pmb->cellbounds.je(interior) + 1; ++j) {
        if (pmb->coords.Xf<2, 2>(j) > output_params.x2_slice) {
          jslice = j - 1;
          output_params.jslice = jslice;
          break;
        }
      }
    } else {
      return false;
    }
  } else {
    if (output_params.x3_slice >= pmb->block_size.x3min &&
        output_params.x3_slice < pmb->block_size.x3max) {
      for (int k = pmb->cellbounds.ks(interior) + 1;
           k <= pmb->cellbounds.ke(interior) + 1; ++k) {
        if (pmb->coords.Xf<3, 3>(k) > output_params.x3_slice) {
          kslice = k - 1;
          output_params.kslice = kslice;
          break;
        }
      }
    } else {
      return false;
    }
  }

  // For each node in OutputData doubly linked list, slice arrays containing output data
  OutputData *pdata, *pnew;
  pdata = pfirst_data_;

  while (pdata != nullptr) {
    pnew = new OutputData;
    pnew->type = pdata->type;
    pnew->name = pdata->name;
    int nx4 = pdata->data.GetDim(4);
    int nx3 = pdata->data.GetDim(3);
    int nx2 = pdata->data.GetDim(2);
    int nx1 = pdata->data.GetDim(1);

    // Loop over variables and dimensions, extract slice
    if (dim == 3) {
      pnew->data = ParArrayND<Real>(PARARRAY_TEMP, nx4, 1, nx2, nx1);
      for (int n = 0; n < nx4; ++n) {
        for (int j = out_js; j <= out_je; ++j) {
          for (int i = out_is; i <= out_ie; ++i) {
            pnew->data(n, 0, j, i) = pdata->data(n, kslice, j, i);
          }
        }
      }
    } else if (dim == 2) {
      pnew->data = ParArrayND<Real>(PARARRAY_TEMP, nx4, nx3, 1, nx1);
      for (int n = 0; n < nx4; ++n) {
        for (int k = out_ks; k <= out_ke; ++k) {
          for (int i = out_is; i <= out_ie; ++i) {
            pnew->data(n, k, 0, i) = pdata->data(n, k, jslice, i);
          }
        }
      }
    } else {
      pnew->data = ParArrayND<Real>(PARARRAY_TEMP, nx4, nx3, nx2, 1);
      for (int n = 0; n < nx4; ++n) {
        for (int k = out_ks; k <= out_ke; ++k) {
          for (int j = out_js; j <= out_je; ++j) {
            pnew->data(n, k, j, 0) = pdata->data(n, k, j, islice);
          }
        }
      }
    }

    ReplaceOutputDataNode(pdata, pnew);
    pdata = pnew->pnext;
  }

  // modify array indices
  if (dim == 3) {
    out_ks = 0;
    out_ke = 0;
  } else if (dim == 2) {
    out_js = 0;
    out_je = 0;
  } else {
    out_is = 0;
    out_ie = 0;
  }
  return true;
}

//----------------------------------------------------------------------------------------
//! \fn void OutputType::SumOutputData(OutputData* pod, int dim)
//  \brief perform data summation and update the data list

void OutputType::SumOutputData(MeshBlock *pmb, int dim) {
  // For each node in OutputData doubly linked list, sum arrays containing output data
  OutputData *pdata = pfirst_data_;

  while (pdata != nullptr) {
    OutputData *pnew = new OutputData;
    pnew->type = pdata->type;
    pnew->name = pdata->name;
    int nx4 = pdata->data.GetDim(4);
    int nx3 = pdata->data.GetDim(3);
    int nx2 = pdata->data.GetDim(2);
    int nx1 = pdata->data.GetDim(1);

    // Loop over variables and dimensions, sum over specified dimension
    if (dim == 3) {
      pnew->data = ParArrayND<Real>(PARARRAY_TEMP, nx4, 1, nx2, nx1);
      for (int n = 0; n < nx4; ++n) {
        for (int k = out_ks; k <= out_ke; ++k) {
          for (int j = out_js; j <= out_je; ++j) {
            for (int i = out_is; i <= out_ie; ++i) {
              pnew->data(n, 0, j, i) += pdata->data(n, k, j, i);
            }
          }
        }
      }
    } else if (dim == 2) {
      pnew->data = ParArrayND<Real>(PARARRAY_TEMP, nx4, nx3, 1, nx1);
      for (int n = 0; n < nx4; ++n) {
        for (int k = out_ks; k <= out_ke; ++k) {
          for (int j = out_js; j <= out_je; ++j) {
            for (int i = out_is; i <= out_ie; ++i) {
              pnew->data(n, k, 0, i) += pdata->data(n, k, j, i);
            }
          }
        }
      }
    } else {
      pnew->data = ParArrayND<Real>(PARARRAY_TEMP, nx4, nx3, nx2, 1);
      for (int n = 0; n < nx4; ++n) {
        for (int k = out_ks; k <= out_ke; ++k) {
          for (int j = out_js; j <= out_je; ++j) {
            for (int i = out_is; i <= out_ie; ++i) {
              pnew->data(n, k, j, 0) += pdata->data(n, k, j, i);
            }
          }
        }
      }
    }

    ReplaceOutputDataNode(pdata, pnew);
    pdata = pdata->pnext;
  }

  // modify array indices
  if (dim == 3) {
    out_ks = 0;
    out_ke = 0;
  } else if (dim == 2) {
    out_js = 0;
    out_je = 0;
  } else {
    out_is = 0;
    out_ie = 0;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void OutputType::CalculateCartesianVector(ParArrayND<Real> &src,
//                                ParArrayND<Real> &dst, Coordinates *pco)
//  \brief Convert vectors in curvilinear coordinates into Cartesian

void OutputType::CalculateCartesianVector(ParArrayND<Real> &src, ParArrayND<Real> &dst,
                                          Coordinates_t *pco) {}

} // namespace parthenon
