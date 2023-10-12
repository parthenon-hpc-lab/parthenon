//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2023 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
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
//! \file histogram.cpp
//  \brief 1D and 2D histograms

// options for building
#include "config.hpp"
#include "globals.hpp"
#include "kokkos_abstraction.hpp"
#include "parameter_input.hpp"
#include "utils/error_checking.hpp"
#include <algorithm>
#include <array>
#include <vector>

// Only proceed if HDF5 output enabled
#ifdef ENABLE_HDF5

#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

// Parthenon headers
#include "coordinates/coordinates.hpp"
#include "defs.hpp"
#include "globals.hpp"
#include "interface/variable_state.hpp"
#include "mesh/mesh.hpp"
#include "outputs/output_utils.hpp"
#include "outputs/outputs.hpp"
#include "utils/error_checking.hpp"

// Ascent headers
#ifdef PARTHENON_ENABLE_ASCENT
#include "ascent.hpp"
#include "conduit_blueprint.hpp"
#include "conduit_relay_io.hpp"
#include "conduit_relay_io_blueprint.hpp"
#endif // ifdef PARTHENON_ENABLE_ASCENT

namespace parthenon {

using namespace OutputUtils;

namespace HistUtil {

struct Histogram {
  int ndim;                                 // 1D or 2D histogram
  std::array<std::string, 2> bin_var_names; // variable(s) for bins
  std::array<int, 2> bin_var_components;    // components of bin variables (vector)
  ParArray2D<Real> bin_edges;
  std::string val_var_name; // variable name of variable to be binned
  int val_var_component;    // component of variable to be binned
  ParArray2D<Real> hist;    // resulting histogram

  Histogram(ParameterInput *pin, const std::string & block_name, const std::string & prefix) {
    ndim = pin->GetInteger(block_name, prefix + "ndim");
    PARTHENON_REQUIRE_THROWS(ndim == 1 || ndim == 2, "Histogram dim must be '1' or '2'");

    const auto x_var_name = pin->GetString(block_name, prefix + "x_variable");
    const auto x_var_component =
        pin->GetInteger(block_name, prefix + "x_variable_component");
    const auto x_edges = pin->GetVector<Real>(block_name, prefix + "x_edges");

    // would add additional logic to pick it from a pack...
    PARTHENON_REQUIRE_THROWS(x_var_component >= 0,
                             "Negative component indices are not supported");
    //  required by binning index function
    PARTHENON_REQUIRE_THROWS(std::is_sorted(x_edges.begin(), x_edges.end()),
                             "Bin edges must be in order.");

    // For 1D profile default initalize y variables
    std::string y_var_name = "";
    int y_var_component = -1;
    auto y_edges = std::vector<Real>();
    // and for 2D profile check if they're explicitly set (not default value)
    if (ndim == 2) {
      y_var_name = pin->GetString(block_name, prefix + "y_variable");
      y_var_component = pin->GetInteger(block_name, prefix + "y_variable_component");
      y_edges = pin->GetVector<Real>(block_name, prefix + "y_edges");

      // would add additional logic to pick it from a pack...
      PARTHENON_REQUIRE_THROWS(y_var_component >= 0,
                               "Negative component indices are not supported");
      //  required by binning index function
      PARTHENON_REQUIRE_THROWS(std::is_sorted(y_edges.begin(), y_edges.end()),
                               "Bin edges must be in order.");
    }

    bin_var_names = {x_var_name, y_var_name};
    bin_var_components = {x_var_component, y_var_component};

    bin_edges = ParArray2D<Real>(prefix + "bin_edges", 2); // TODO split these...


    val_var_name = pin->GetString(block_name, prefix + "val_variable");
    val_var_component =
        pin->GetInteger(block_name, prefix + "val_variable_component");
      // would add additional logic to pick it from a pack...
      PARTHENON_REQUIRE_THROWS(val_var_component >= 0,
                               "Negative component indices are not supported");

  }
};

} // namespace HistUtil

//----------------------------------------------------------------------------------------
//! \fn void HistogramOutput:::SetupHistograms(ParameterInput *pin)
//  \brief Process parameter input to setup persistent histograms
HistogramOutput::HistogramOutput(const OutputParameters &op, ParameterInput *pin)
    : OutputType(op) {

  num_histograms_ = pin->GetOrAddInteger(op.block_name, "num_histograms", 0);

  std::vector<HistUtil::Histogram> histograms_; // TODO make private class member

  for (int i = 0; i < num_histograms_; i++) {
    const auto prefix = "hist" + std::to_string(i) + "_";
    histograms_.emplace_back(pin, op.block_name, prefix);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void HistogramOutput:::WriteOutputFile(Mesh *pm)
//  \brief  Calculate histograms
void HistogramOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin, SimTime *tm,
                                      const SignalHandler::OutputSignal signal) {

  // advance output parameters
  output_params.file_number++;
  output_params.next_time += output_params.dt;
  pin->SetInteger(output_params.block_name, "file_number", output_params.file_number);
  pin->SetReal(output_params.block_name, "next_time", output_params.next_time);
}

} // namespace parthenon
#endif // ifndef PARTHENON_ENABLE_ASCENT
