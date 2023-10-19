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
#include <cmath>
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
#include "utils/sort.hpp" // for upper_bound

// ScatterView is not part of Kokkos core interface
#include "Kokkos_ScatterView.hpp"

namespace parthenon {

using namespace OutputUtils;

namespace HistUtil {

// Parse edges from input parameters. Returns the edges themselves (to be used as list for
// arbitrary bins) as well as min and step sizes (potentially in log space) for direct
// indexing.
std::tuple<ParArray1D<Real>, EdgeType, Real, Real>
GetEdges(ParameterInput *pin, const std::string &block_name, const std::string &prefix) {
  std::vector<Real> edges_in;
  auto edge_type = EdgeType::Undefined;
  auto edge_min = std::numeric_limits<Real>::quiet_NaN();
  auto edge_dbin = std::numeric_limits<Real>::quiet_NaN();

  const auto edge_type_str = pin->GetString(block_name, prefix + "type");
  if (edge_type_str == "lin" || edge_type_str == "log") {
    edge_min = pin->GetReal(block_name, prefix + "min");
    const auto edge_max = pin->GetReal(block_name, prefix + "max");
    PARTHENON_REQUIRE_THROWS(edge_max > edge_min,
                             "Histogram max needs to be larger than min.")

    const auto edge_num_bins = pin->GetReal(block_name, prefix + "num_bins");
    PARTHENON_REQUIRE_THROWS(edge_num_bins >= 1, "Need at least one bin for histogram.");

    if (edge_type_str == "lin") {
      edge_type = EdgeType::Lin;
      edge_dbin = (edge_max - edge_min) / (edge_num_bins);
      for (int i = 0; i < edge_num_bins; i++) {
        edges_in.emplace_back(edge_min + i * edge_dbin);
      }
      edges_in.emplace_back(edge_max);
    } else if (edge_type_str == "log") {
      edge_type = EdgeType::Log;
      PARTHENON_REQUIRE_THROWS(
          edge_min > 0.0 && edge_max > 0.0,
          "Log binning for negative values not implemented. However, you can specify "
          "arbitrary bin edges through the 'list' edge type.")

      // override start with log value for direct indexing in histogram kernel
      edge_min = std::log10(edge_min);
      edge_dbin = (std::log10(edge_max) - edge_min) / (edge_num_bins);
      for (int i = 0; i < edge_num_bins; i++) {
        edges_in.emplace_back(std::pow(10., edge_min + i * edge_dbin));
      }
      edges_in.emplace_back(edge_max);
    } else {
      PARTHENON_FAIL("Not sure how I got here...")
    }

  } else if (edge_type_str == "list") {
    edge_type = EdgeType::List;
    edges_in = pin->GetVector<Real>(block_name, prefix + "list");
    //  required by binning index function
    PARTHENON_REQUIRE_THROWS(std::is_sorted(edges_in.begin(), edges_in.end()),
                             "Bin edges must be in order.");
    PARTHENON_REQUIRE_THROWS(edges_in.size() >= 2,
                             "Need at least one bin, i.e., two edges.");

  } else {
    PARTHENON_THROW(
        "Unknown edge type for histogram. Supported types are lin, log, and list.")
  }
  auto edges = ParArray1D<Real>(prefix, edges_in.size());
  auto edges_h = edges.GetHostMirror();
  for (int i = 0; i < edges_in.size(); i++) {
    edges_h(i) = edges_in[i];
  }
  Kokkos::deep_copy(edges, edges_h);

  PARTHENON_REQUIRE_THROWS(
      edge_type != EdgeType::Undefined,
      "Edge type not set and it's unclear how this code was triggered...");
  return {edges, edge_type, edge_min, edge_dbin};
}

Histogram::Histogram(ParameterInput *pin, const std::string &block_name,
                     const std::string &prefix) {
  ndim = pin->GetInteger(block_name, prefix + "ndim");
  PARTHENON_REQUIRE_THROWS(ndim == 1 || ndim == 2, "Histogram dim must be '1' or '2'");

  x_var_name = pin->GetString(block_name, prefix + "x_variable");
  x_var_component = -1;
  if (x_var_name == "HIST_COORD_X1") {
    x_var_type = VarType::X1;
  } else if (x_var_name == "HIST_COORD_X2") {
    x_var_type = VarType::X2;
  } else if (x_var_name == "HIST_COORD_X3") {
    x_var_type = VarType::X3;
  } else if (x_var_name == "HIST_COORD_R") {
    PARTHENON_REQUIRE_THROWS(
        typeid(Coordinates_t) == typeid(UniformCartesian),
        "Radial coordinate currently only works for uniform Cartesian coordinates.");
    x_var_type = VarType::R;
  } else {
    x_var_type = VarType::Var;
    x_var_component = pin->GetInteger(block_name, prefix + "x_variable_component");
    // would add additional logic to pick it from a pack...
    PARTHENON_REQUIRE_THROWS(x_var_component >= 0,
                             "Negative component indices are not supported");
  }

  std::tie(x_edges, x_edges_type, x_edge_min, x_edge_dbin) =
      GetEdges(pin, block_name, prefix + "x_edges_");

  // For 1D profile default initalize y variables
  y_var_name = "";
  y_var_component = -1;
  y_var_type = VarType::Unused;
  // and for 2D profile check if they're explicitly set (not default value)
  if (ndim == 2) {
    y_var_name = pin->GetString(block_name, prefix + "y_variable");
    if (y_var_name == "HIST_COORD_X1") {
      y_var_type = VarType::X1;
    } else if (y_var_name == "HIST_COORD_X2") {
      y_var_type = VarType::X2;
    } else if (y_var_name == "HIST_COORD_X3") {
      y_var_type = VarType::X3;
    } else if (y_var_name == "HIST_COORD_R") {
      PARTHENON_REQUIRE_THROWS(
          typeid(Coordinates_t) == typeid(UniformCartesian),
          "Radial coordinate currently only works for uniform Cartesian coordinates.");
      y_var_type = VarType::R;
    } else {
      y_var_type = VarType::Var;
      y_var_component = pin->GetInteger(block_name, prefix + "y_variable_component");
      // would add additional logic to pick it from a pack...
      PARTHENON_REQUIRE_THROWS(y_var_component >= 0,
                               "Negative component indices are not supported");
    }

    std::tie(y_edges, y_edges_type, y_edge_min, y_edge_dbin) =
        GetEdges(pin, block_name, prefix + "y_edges_");

  } else {
    y_edges = ParArray1D<Real>(prefix + "y_edges_unused", 0);
  }

  binned_var_name =
      pin->GetOrAddString(block_name, prefix + "binned_variable", "HIST_ONES");
  binned_var_component = -1; // implies that we're not binning a variable but count
  if (binned_var_name != "HIST_ONES") {
    binned_var_component =
        pin->GetInteger(block_name, prefix + "binned_variable_component");
    // would add additional logic to pick it from a pack...
    PARTHENON_REQUIRE_THROWS(binned_var_component >= 0,
                             "Negative component indices are not supported");
  }

  const auto nxbins = x_edges.extent_int(0) - 1;
  const auto nybins = ndim == 2 ? y_edges.extent_int(0) - 1 : 1;

  result = ParArray2D<Real>(prefix + "result", nybins, nxbins);
  scatter_result =
      Kokkos::Experimental::ScatterView<Real **, LayoutWrapper>(result.KokkosView());

  weight_by_vol = pin->GetOrAddBoolean(block_name, prefix + "weight_by_volume", false);

  weight_var_name =
      pin->GetOrAddString(block_name, prefix + "weight_variable", "HIST_ONES");
  weight_var_component = -1; // implies that weighting is not applied
  if (weight_var_name != "HIST_ONES") {
    weight_var_component =
        pin->GetInteger(block_name, prefix + "weight_variable_component");
    // would add additional logic to pick it from a pack...
    PARTHENON_REQUIRE_THROWS(weight_var_component >= 0,
                             "Negative component indices are not supported");
  }
}

// Computes a 1D or 2D histogram with inclusive lower edges and inclusive rightmost edges.
// Function could in principle be templated on dimension, but it's currently not expected
// to be a performance concern (because it won't be called that often).
void CalcHist(Mesh *pm, const Histogram &hist) {
  Kokkos::Profiling::pushRegion("Calculate single histogram");
  const auto x_var_component = hist.x_var_component;
  const auto y_var_component = hist.y_var_component;
  const auto binned_var_component = hist.binned_var_component;
  const auto weight_var_component = hist.weight_var_component;
  const auto x_var_type = hist.x_var_type;
  const auto y_var_type = hist.y_var_type;
  const auto x_edges = hist.x_edges;
  const auto y_edges = hist.y_edges;
  const auto x_edges_type = hist.x_edges_type;
  const auto y_edges_type = hist.y_edges_type;
  const auto x_edge_min = hist.x_edge_min;
  const auto x_edge_dbin = hist.x_edge_dbin;
  const auto y_edge_min = hist.y_edge_min;
  const auto y_edge_dbin = hist.y_edge_dbin;
  const auto hist_ndim = hist.ndim;
  const auto weight_by_vol = hist.weight_by_vol;
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

    const auto x_var_pack_string = x_var_type == VarType::Var
                                       ? std::vector<std::string>{hist.x_var_name}
                                       : std::vector<std::string>{};
    const auto x_var = md->PackVariables(x_var_pack_string);

    const auto y_var_pack_string = y_var_type == VarType::Var
                                       ? std::vector<std::string>{hist.y_var_name}
                                       : std::vector<std::string>{};
    const auto y_var = md->PackVariables(y_var_pack_string);

    const auto binned_var_pack_string =
        binned_var_component == -1 ? std::vector<std::string>{}
                                   : std::vector<std::string>{hist.binned_var_name};
    const auto binned_var = md->PackVariables(binned_var_pack_string);

    const auto weight_var_pack_string =
        weight_var_component == -1 ? std::vector<std::string>{}
                                   : std::vector<std::string>{hist.weight_var_name};
    const auto weight_var = md->PackVariables(weight_var_pack_string);

    const auto ib = md->GetBoundsI(IndexDomain::interior);
    const auto jb = md->GetBoundsJ(IndexDomain::interior);
    const auto kb = md->GetBoundsK(IndexDomain::interior);

    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "CalcHist", DevExecSpace(), 0, md->NumBlocks() - 1, kb.s,
        kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          auto &coords = x_var.GetCoords(b);
          auto x_val = std::numeric_limits<Real>::quiet_NaN();
          if (x_var_type == VarType::X1) {
            x_val = coords.Xc<1>(k, j, i);
          } else if (x_var_type == VarType::X2) {
            x_val = coords.Xc<2>(k, j, i);
          } else if (x_var_type == VarType::X3) {
            x_val = coords.Xc<3>(k, j, i);
          } else if (x_var_type == VarType::R) {
            x_val = Kokkos::sqrt(SQR(coords.Xc<1>(k, j, i)) + SQR(coords.Xc<2>(k, j, i)) +
                                 SQR(coords.Xc<3>(k, j, i)));
          } else {
            x_val = x_var(b, x_var_component, k, j, i);
          }
          if (x_val < x_edges(0) || x_val > x_edges(x_edges.extent_int(0) - 1)) {
            return;
          }

          int x_bin = -1;
          // if we're on the rightmost edge, directly set last bin
          if (x_val == x_edges(x_edges.extent_int(0) - 1)) {
            x_bin = x_edges.extent_int(0) - 2;
          } else {
            // for lin and log directly pick index
            if (x_edges_type == EdgeType::Lin) {
              x_bin = static_cast<int>((x_val - x_edge_min) / x_edge_dbin);
            } else if (x_edges_type == EdgeType::Log) {
              x_bin = static_cast<int>((Kokkos::log10(x_val) - x_edge_min) / x_edge_dbin);
              // otherwise search
            } else {
              x_bin = upper_bound(x_edges, x_val) - 1;
            }
          }
          PARTHENON_DEBUG_REQUIRE(x_bin >= 0, "Bin not found");

          int y_bin = -1;
          if (hist_ndim == 2) {
            auto y_val = std::numeric_limits<Real>::quiet_NaN();
            if (y_var_type == VarType::X1) {
              y_val = coords.Xc<1>(k, j, i);
            } else if (y_var_type == VarType::X2) {
              y_val = coords.Xc<2>(k, j, i);
            } else if (y_var_type == VarType::X3) {
              y_val = coords.Xc<3>(k, j, i);
            } else if (y_var_type == VarType::R) {
              y_val =
                  Kokkos::sqrt(SQR(coords.Xc<1>(k, j, i)) + SQR(coords.Xc<2>(k, j, i)) +
                               SQR(coords.Xc<3>(k, j, i)));
            } else {
              y_val = y_var(b, y_var_component, k, j, i);
            }

            if (y_val < y_edges(0) || y_val > y_edges(y_edges.extent_int(0) - 1)) {
              return;
            }
            // if we're on the rightmost edge, directly set last bin
            if (y_val == y_edges(y_edges.extent_int(0) - 1)) {
              y_bin = y_edges.extent_int(0) - 2;
            } else {
              // for lin and log directly pick index
              if (y_edges_type == EdgeType::Lin) {
                y_bin = static_cast<int>((y_val - y_edge_min) / y_edge_dbin);
              } else if (y_edges_type == EdgeType::Log) {
                y_bin =
                    static_cast<int>((Kokkos::log10(y_val) - y_edge_min) / y_edge_dbin);
                // otherwise search
              } else {
                y_bin = upper_bound(y_edges, y_val) - 1;
              }
            }
            PARTHENON_DEBUG_REQUIRE(y_bin >= 0, "Bin not found");
          }
          auto res = scatter.access();
          const auto val_to_add = binned_var_component == -1
                                      ? 1
                                      : binned_var(b, binned_var_component, k, j, i);
          auto weight = weight_by_vol ? coords.CellVolume(k, j, i) : 1.0;
          weight *= weight_var_component == -1
                        ? 1.0
                        : weight_var(b, weight_var_component, k, j, i);
          res(y_bin, x_bin) += val_to_add * weight;
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

std::string HistogramOutput::GenerateFilename_(ParameterInput *pin, SimTime *tm,
                                               const SignalHandler::OutputSignal signal) {
  using namespace HDF5;

  auto filename = std::string(output_params.file_basename);
  filename.append(".");
  filename.append(output_params.file_id);
  filename.append(".histograms.");
  if (signal == SignalHandler::OutputSignal::now) {
    filename.append("now");
  } else if (signal == SignalHandler::OutputSignal::final &&
             output_params.file_label_final) {
    filename.append("final");
    // default time based data dump
  } else {
    std::stringstream file_number;
    file_number << std::setw(output_params.file_number_width) << std::setfill('0')
                << output_params.file_number;
    filename.append(file_number.str());
  }
  filename.append(".hdf");

  return filename;
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
  // Given the expect size of histograms, we'll use serial HDF
  if (Globals::my_rank == 0) {
    using namespace HDF5;
    H5P const pl_xfer = H5P::FromHIDCheck(H5Pcreate(H5P_DATASET_XFER));

    // As we're reusing the interface from the existing hdf5 output, we have to define
    // everything as 7D arrays.
    // Counts will be set for each histogram individually below.
    const std::array<hsize_t, H5_NDIM> local_offset({0, 0, 0, 0, 0, 0, 0});
    std::array<hsize_t, H5_NDIM> local_count({0, 0, 0, 0, 0, 0, 0});
    std::array<hsize_t, H5_NDIM> global_count({0, 0, 0, 0, 0, 0, 0});

    // create/open HDF5 file
    const std::string filename = GenerateFilename_(pin, tm, signal);

    H5F file;
    try {
      file = H5F::FromHIDCheck(
          H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT));
    } catch (std::exception &ex) {
      std::stringstream err;
      err << "### ERROR: Failed to create HDF5 output file '" << filename
          << "' with the following error:" << std::endl
          << ex.what() << std::endl;
      PARTHENON_THROW(err)
    }

    const H5G info_group = MakeGroup(file, "/Info");
    if (tm != nullptr) {
      HDF5WriteAttribute("NCycle", tm->ncycle, info_group);
      HDF5WriteAttribute("Time", tm->time, info_group);
      HDF5WriteAttribute("dt", tm->dt, info_group);
    }
    HDF5WriteAttribute("num_histograms", num_histograms_, info_group);

    for (int h = 0; h < num_histograms_; h++) {
      auto &hist = histograms_[h];
      const H5G hist_group = MakeGroup(file, "/" + std::to_string(h));
      HDF5WriteAttribute("ndim", hist.ndim, hist_group);
      HDF5WriteAttribute("x_var_name", hist.x_var_name.c_str(), hist_group);
      HDF5WriteAttribute("x_var_component", hist.x_var_component, hist_group);
      HDF5WriteAttribute("binned_var_name", hist.binned_var_name.c_str(), hist_group);
      HDF5WriteAttribute("binned_var_component", hist.binned_var_component, hist_group);

      const auto x_edges_h = hist.x_edges.GetHostMirrorAndCopy();
      local_count[0] = global_count[0] = x_edges_h.extent_int(0);
      HDF5Write1D(hist_group, "x_edges", x_edges_h.data(), local_offset.data(),
                  local_count.data(), global_count.data(), pl_xfer);

      if (hist.ndim == 2) {
        HDF5WriteAttribute("y_var_name", hist.y_var_name.c_str(), hist_group);
        HDF5WriteAttribute("y_var_component", hist.y_var_component, hist_group);

        const auto y_edges_h = hist.y_edges.GetHostMirrorAndCopy();
        local_count[0] = global_count[0] = y_edges_h.extent_int(0);
        HDF5Write1D(hist_group, "y_edges", y_edges_h.data(), local_offset.data(),
                    local_count.data(), global_count.data(), pl_xfer);
      }

      const auto hist_h = hist.result.GetHostMirrorAndCopy();
      // Ensure correct output format (as the data in Parthenon may, in theory, vary by
      // changing the default view layout) so that it matches the numpy output  (row
      // major, x first)
      std::vector<Real> tmp_data(hist_h.size());
      int idx = 0;
      for (int i = 0; i < hist_h.extent_int(1); ++i) {
        for (int j = 0; j < hist_h.extent_int(0); ++j) {
          tmp_data[idx++] = hist_h(j, i);
        }
      }

      local_count[0] = global_count[0] = hist_h.extent_int(1);
      if (hist.ndim == 2) {
        local_count[1] = global_count[1] = hist_h.extent_int(0);

        HDF5Write2D(hist_group, "data", tmp_data.data(), local_offset.data(),
                    local_count.data(), global_count.data(), pl_xfer);
      } else {
        // No y-dim for 1D histogram -- though unnecessary as it's not read anyway
        local_count[1] = global_count[1] = 0;
        HDF5Write1D(hist_group, "data", tmp_data.data(), local_offset.data(),
                    local_count.data(), global_count.data(), pl_xfer);
      }
    }
  }
  Kokkos::Profiling::popRegion(); // Dump histograms

  // advance file ids
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
