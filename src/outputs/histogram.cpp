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
#include "basic_types.hpp"
#include "config.hpp"
#include "globals.hpp"
#include "kokkos_abstraction.hpp"
#include "parameter_input.hpp"
#include "parthenon_array_generic.hpp"
#include "utils/error_checking.hpp"

// Only proceed if HDF5 output enabled
#ifdef ENABLE_HDF5

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Parthenon headers
#include "coordinates/coordinates.hpp"
#include "defs.hpp"
#include "globals.hpp"
#include "interface/variable_state.hpp"
#include "mesh/mesh.hpp"
#include "outputs/output_utils.hpp"
#include "outputs/outputs.hpp"
#include "outputs/parthenon_hdf5.hpp"
#include "utils/error_checking.hpp"

// ScatterView is not part of Kokkos core interface
#include "Kokkos_ScatterView.hpp"

#include FS_HEADER
namespace fs = FS_NAMESPACE;

namespace parthenon {

using namespace OutputUtils;

namespace HistUtil {

Histogram::Histogram(ParameterInput *pin, const std::string &block_name,
                     const std::string &prefix) {
  ndim = pin->GetInteger(block_name, prefix + "ndim");
  PARTHENON_REQUIRE_THROWS(ndim == 1 || ndim == 2, "Histogram dim must be '1' or '2'");

  x_var_name = pin->GetString(block_name, prefix + "x_variable");

  x_var_component = pin->GetInteger(block_name, prefix + "x_variable_component");
  // would add additional logic to pick it from a pack...
  PARTHENON_REQUIRE_THROWS(x_var_component >= 0,
                           "Negative component indices are not supported");

  const auto x_edges_in = pin->GetVector<Real>(block_name, prefix + "x_edges");
  //  required by binning index function
  PARTHENON_REQUIRE_THROWS(std::is_sorted(x_edges_in.begin(), x_edges_in.end()),
                           "Bin edges must be in order.");
  PARTHENON_REQUIRE_THROWS(x_edges_in.size() >= 2,
                           "Need at least one bin, i.e., two edges.");
  x_edges = ParArray1D<Real>(prefix + "x_edges", x_edges_in.size());
  auto x_edges_h = x_edges.GetHostMirror();
  for (int i = 0; i < x_edges_in.size(); i++) {
    x_edges_h(i) = x_edges_in[i];
  }
  Kokkos::deep_copy(x_edges, x_edges_h);

  // For 1D profile default initalize y variables
  y_var_name = "";
  y_var_component = -1;
  // and for 2D profile check if they're explicitly set (not default value)
  if (ndim == 2) {
    y_var_name = pin->GetString(block_name, prefix + "y_variable");

    y_var_component = pin->GetInteger(block_name, prefix + "y_variable_component");
    // would add additional logic to pick it from a pack...
    PARTHENON_REQUIRE_THROWS(y_var_component >= 0,
                             "Negative component indices are not supported");

    const auto y_edges_in = pin->GetVector<Real>(block_name, prefix + "y_edges");
    //  required by binning index function
    PARTHENON_REQUIRE_THROWS(std::is_sorted(y_edges_in.begin(), y_edges_in.end()),
                             "Bin edges must be in order.");
    PARTHENON_REQUIRE_THROWS(y_edges_in.size() >= 2,
                             "Need at least one bin, i.e., two edges.");
    y_edges = ParArray1D<Real>(prefix + "y_edges", y_edges_in.size());
    auto y_edges_h = y_edges.GetHostMirror();
    for (int i = 0; i < y_edges_in.size(); i++) {
      y_edges_h(i) = y_edges_in[i];
    }
    Kokkos::deep_copy(y_edges, y_edges_h);
  } else {
    y_edges = ParArray1D<Real>(prefix + "y_edges_unused", 0);
  }

  binned_var_name = pin->GetString(block_name, prefix + "binned_variable");
  binned_var_component =
      pin->GetInteger(block_name, prefix + "binned_variable_component");
  // would add additional logic to pick it from a pack...
  PARTHENON_REQUIRE_THROWS(binned_var_component >= 0,
                           "Negative component indices are not supported");

  const auto nxbins = x_edges.extent_int(0) - 1;
  const auto nybins = ndim == 2 ? y_edges.extent_int(0) - 1 : 1;

  result = ParArray2D<Real>(prefix + "result", nybins, nxbins);
  scatter_result = Kokkos::Experimental::ScatterView<Real **>(result.KokkosView());
}

// Returns the upper bound (or the array size if value has not been found)
// Could/Should be replaced with a Kokkos std version once available (currently schedule
// for 4.2 release).
// TODO add unit test
KOKKOS_INLINE_FUNCTION int upper_bound(const ParArray1D<Real> &arr, Real val) {
  int l = 0;
  int r = arr.extent_int(0);
  int m;
  while (l < r) {
    m = l + (r - l) / 2;
    if (val >= arr(m)) {
      l = m + 1;
    } else {
      r = m;
    }
  }
  if (l < arr.extent_int(0) && val >= arr(l)) {
    l++;
  }
  return l;
}

// Computes a 1D or 2D histogram with inclusive lower edges and inclusive rightmost edges.
// Function could in principle be templated on dimension, but it's currently not expected
// to be a performance concern (because it won't be called that often).
void CalcHist(Mesh *pm, const Histogram &hist) {
  Kokkos::Profiling::pushRegion("Calculate single histogram");
  const auto x_var_component = hist.x_var_component;
  const auto y_var_component = hist.y_var_component;
  const auto binned_var_component = hist.binned_var_component;
  const auto x_edges = hist.x_edges;
  const auto y_edges = hist.y_edges;
  const auto hist_ndim = hist.ndim;
  auto result = hist.result;
  auto scatter = hist.scatter_result;

  // Reset ScatterView from previous output
  scatter.reset();
  // Also reset the histogram from previous call.
  // Currently still required for consistent results between host and device backends, see
  // https://github.com/kokkos/kokkos/issues/6363
  Kokkos::deep_copy(result, 0);

  const int num_partitions = pm->DefaultNumPartitions();

  for (int p = 0; p < num_partitions; p++) {
    auto &md = pm->mesh_data.GetOrAdd("base", p);

    const auto x_var = md->PackVariables(std::vector<std::string>{hist.x_var_name});
    const auto y_var = md->PackVariables(std::vector<std::string>{hist.y_var_name});
    const auto binned_var =
        md->PackVariables(std::vector<std::string>{hist.binned_var_name});
    const auto ib = md->GetBoundsI(IndexDomain::interior);
    const auto jb = md->GetBoundsJ(IndexDomain::interior);
    const auto kb = md->GetBoundsK(IndexDomain::interior);

    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "CalcHist", DevExecSpace(), 0, x_var.GetDim(5) - 1, kb.s,
        kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          const auto &x_val = x_var(b, x_var_component, k, j, i);
          if (x_val < x_edges(0) || x_val > x_edges(x_edges.extent_int(0) - 1)) {
            return;
          }
          // No further check for x_bin required as the preceeding if-statement guarantees
          // x_val to fall in one bin.
          const auto x_bin = upper_bound(x_edges, x_val) - 1;

          int y_bin = 0;
          if (hist_ndim == 2) {
            const auto &y_val = y_var(b, y_var_component, k, j, i);
            if (y_val < y_edges(0) || y_val > y_edges(y_edges.extent_int(0) - 1)) {
              return;
            }
            // No further check for y_bin required as the preceeding if-statement
            // guarantees y_val to fall in one bin.
            y_bin = upper_bound(y_edges, y_val) - 1;
          }
          auto res = scatter.access();
          res(y_bin, x_bin) += binned_var(b, binned_var_component, k, j, i);
        });
    // "reduce" results from scatter view to original view. May be a no-op depending on
    // backend.
    Kokkos::Experimental::contribute(result.KokkosView(), scatter);
  }
  // Ensure all (implicit) reductions from contribute are done
  Kokkos::fence(); // May not be required

  // Now reduce over ranks
#ifdef MPI_PARALLEL
  if (Globals::my_rank == 0) {
    PARTHENON_MPI_CHECK(MPI_Reduce(MPI_IN_PLACE, result.data(), result.size(),
                                   MPI_PARTHENON_REAL, MPI_SUM, 0, MPI_COMM_WORLD));
  } else {
    PARTHENON_MPI_CHECK(MPI_Reduce(result.data(), result.data(), result.size(),
                                   MPI_PARTHENON_REAL, MPI_SUM, 0, MPI_COMM_WORLD));
  }
#endif
  Kokkos::Profiling::popRegion(); // Calculate single histogram
}

} // namespace HistUtil

//----------------------------------------------------------------------------------------
//! \fn void HistogramOutput:::SetupHistograms(ParameterInput *pin)
//  \brief Process parameter input to setup persistent histograms
HistogramOutput::HistogramOutput(const OutputParameters &op, ParameterInput *pin)
    : OutputType(op) {

  num_histograms_ = pin->GetOrAddInteger(op.block_name, "num_histograms", 0);

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

  Kokkos::Profiling::pushRegion("Calculate all histograms");
  for (auto &hist : histograms_) {
    CalcHist(pm, hist);
  }
  Kokkos::Profiling::popRegion(); // Calculate all histograms

  Kokkos::Profiling::pushRegion("Dump histograms");
  if (Globals::my_rank == 0) {
    using namespace HDF5;
    // create/open HDF5 file
    const std::string filename = "histogram.hdf";
    H5F file;
    try {
      if (fs::exists(filename)) {
        file = H5F::FromHIDCheck(H5Fopen(filename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT));
      } else {
        file = H5F::FromHIDCheck(
            H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT));
      }
    } catch (std::exception &ex) {
      std::stringstream err;
      err << "### ERROR: Failed to open/create HDF5 output file '" << filename
          << "' with the following error:" << std::endl
          << ex.what() << std::endl;
      PARTHENON_THROW(err)
    }

    std::string out_label;
    if (signal == SignalHandler::OutputSignal::now) {
      out_label = "now";
    } else if (signal == SignalHandler::OutputSignal::final &&
               output_params.file_label_final) {
      out_label = "final";
      // default time based data dump
    } else {
      std::stringstream file_number;
      file_number << std::setw(output_params.file_number_width) << std::setfill('0')
                  << output_params.file_number;
      out_label = file_number.str();
    }

    const H5G all_hist_group = MakeGroup(file, "/" + out_label);
    if (tm != nullptr) {
      HDF5WriteAttribute("NCycle", tm->ncycle, all_hist_group);
      HDF5WriteAttribute("Time", tm->time, all_hist_group);
      HDF5WriteAttribute("dt", tm->dt, all_hist_group);
    }
    HDF5WriteAttribute("num_histograms", num_histograms_, all_hist_group);

    for (int i = 0; i < num_histograms_; i++) {
      auto &hist = histograms_[i];
      const H5G hist_group = MakeGroup(all_hist_group, "/" + std::to_string(i));
      HDF5WriteAttribute("x_var_name", hist.x_var_name, hist_group);
      HDF5WriteAttribute("x_var_component", hist.x_var_component, hist_group);
      HDF5WriteAttribute("y_var_name", hist.y_var_name, hist_group);
      HDF5WriteAttribute("y_var_component", hist.y_var_component, hist_group);
      HDF5WriteAttribute("binned_var_name", hist.binned_var_name, hist_group);
      HDF5WriteAttribute("binned_var_component", hist.binned_var_component, hist_group);

      const auto hist_h = hist.result.GetHostMirrorAndCopy();
      std::cout << "Hist result: ";
      for (int i = 0; i < hist_h.extent_int(1); i++) {
        std::cout << hist_h(0, i) << " ";
      }
      std::cout << "\n";
    }
  }
  Kokkos::Profiling::popRegion(); // Dump histograms

  // advance output parameters
  if (signal == SignalHandler::OutputSignal::none) {
    // After file has been opened with the current number, already advance output
    // parameters so that for restarts the file is not immediatly overwritten again.
    // Only applies to default time-based data dumps, so that writing "now" and "final"
    // outputs does not change the desired output numbering.
    output_params.file_number++;
    output_params.next_time += output_params.dt;
    pin->SetInteger(output_params.block_name, "file_number", output_params.file_number);
    pin->SetReal(output_params.block_name, "next_time", output_params.next_time);
  }
}

} // namespace parthenon
#endif // ifndef PARTHENON_ENABLE_ASCENT
