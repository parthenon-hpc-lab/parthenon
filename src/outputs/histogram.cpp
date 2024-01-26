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
#include <tuple>
#include <vector>

// Parthenon headers
#include "coordinates/coordinates.hpp"
#include "defs.hpp"
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

// Parse input for x and y vars from input
// std::tuple<std::string, int, VarType>
auto ProcessVarInput(ParameterInput *pin, const std::string &block_name,
                     const std::string &prefix) {
  auto var_name = pin->GetString(block_name, prefix + "variable");
  int var_component = -1;
  VarType var_type;
  if (var_name == "HIST_COORD_X1") {
    var_type = VarType::X1;
  } else if (var_name == "HIST_COORD_X2") {
    var_type = VarType::X2;
  } else if (var_name == "HIST_COORD_X3") {
    var_type = VarType::X3;
  } else if (var_name == "HIST_COORD_R") {
    PARTHENON_REQUIRE_THROWS(
        typeid(Coordinates_t) == typeid(UniformCartesian),
        "Radial coordinate currently only works for uniform Cartesian coordinates.");
    var_type = VarType::R;
  } else {
    var_type = VarType::Var;
    var_component = pin->GetInteger(block_name, prefix + "variable_component");
    // would add additional logic to pick it from a pack...
    PARTHENON_REQUIRE_THROWS(var_component >= 0,
                             "Negative component indices are not supported");
  }

  return std::make_tuple(var_name, var_component, var_type);
}

// Parse edges from input parameters. Returns the edges themselves (to be used as list for
// arbitrary bins) as well as min and step sizes (potentially in log space) for direct
// indexing.
auto GetEdges(ParameterInput *pin, const std::string &block_name,
              const std::string &prefix) {
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

    const auto edge_num_bins = pin->GetInteger(block_name, prefix + "num_bins");
    PARTHENON_REQUIRE_THROWS(edge_num_bins >= 1, "Need at least one bin for histogram.");
    edges_in.reserve(edge_num_bins);

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
    PARTHENON_REQUIRE_THROWS(edges_in.size() >= 2 && edges_in[1] > edges_in[0],
                             "Need at least one bin, i.e., two distinct edges.");

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
  return std::make_tuple(edges, edge_type, edge_min, edge_dbin);
}

Histogram::Histogram(ParameterInput *pin, const std::string &block_name,
                     const std::string &name) {
  name_ = name;
  const auto prefix = name + "_";

  ndim_ = pin->GetInteger(block_name, prefix + "ndim");
  PARTHENON_REQUIRE_THROWS(ndim_ == 1 || ndim_ == 2, "Histogram dim must be '1' or '2'");

  std::tie(x_var_name_, x_var_component_, x_var_type_) =
      ProcessVarInput(pin, block_name, prefix + "x_");

  std::tie(x_edges_, x_edges_type_, x_edge_min_, x_edge_dbin_) =
      GetEdges(pin, block_name, prefix + "x_edges_");

  // For 1D profile default initalize y variables
  y_var_name_ = "";
  y_var_component_ = -1;
  y_var_type_ = VarType::Unused;
  // and for 2D profile check if they're explicitly set (not default value)
  if (ndim_ == 2) {
    std::tie(y_var_name_, y_var_component_, y_var_type_) =
        ProcessVarInput(pin, block_name, prefix + "y_");

    std::tie(y_edges_, y_edges_type_, y_edge_min_, y_edge_dbin_) =
        GetEdges(pin, block_name, prefix + "y_edges_");

  } else {
    y_edges_ = ParArray1D<Real>(prefix + "y_edges_unused", 0);
  }

  binned_var_name_ =
      pin->GetOrAddString(block_name, prefix + "binned_variable", "HIST_ONES");
  binned_var_component_ = -1; // implies that we're not binning a variable but count
  if (binned_var_name_ != "HIST_ONES") {
    binned_var_component_ =
        pin->GetInteger(block_name, prefix + "binned_variable_component");
    // would add additional logic to pick it from a pack...
    PARTHENON_REQUIRE_THROWS(binned_var_component_ >= 0,
                             "Negative component indices are not supported");
  }

  const auto nxbins = x_edges_.extent_int(0) - 1;
  const auto nybins = ndim_ == 2 ? y_edges_.extent_int(0) - 1 : 1;

  result_ = ParArray2D<Real>(prefix + "result", nybins, nxbins);
  scatter_result = Kokkos::Experimental::ScatterView<
      Real **, LayoutWrapper, parthenon::DevExecSpace, Kokkos::Experimental::ScatterSum,
      Kokkos::Experimental::ScatterDuplicated, Kokkos::Experimental::ScatterNonAtomic>(
      result.KokkosView());

  accumulate_ = pin->GetOrAddBoolean(block_name, prefix + "accumulate", false);
  weight_by_vol_ = pin->GetOrAddBoolean(block_name, prefix + "weight_by_volume", false);

  weight_var_name_ =
      pin->GetOrAddString(block_name, prefix + "weight_variable", "HIST_ONES");
  weight_var_component_ = -1; // implies that weighting is not applied
  if (weight_var_name_ != "HIST_ONES") {
    weight_var_component_ =
        pin->GetInteger(block_name, prefix + "weight_variable_component");
    // would add additional logic to pick it from a pack...
    PARTHENON_REQUIRE_THROWS(weight_var_component_ >= 0,
                             "Negative component indices are not supported");
  }
}

// Computes a 1D or 2D histogram with inclusive lower edges and inclusive rightmost edges.
// Function could in principle be templated on dimension, but it's currently not expected
// to be a performance concern (because it won't be called that often).
void Histogram::CalcHist(Mesh *pm) {
  Kokkos::Profiling::pushRegion("Calculate single histogram");
  const auto x_var_component = x_var_component_;
  const auto y_var_component = y_var_component_;
  const auto binned_var_component = binned_var_component_;
  const auto weight_var_component = weight_var_component_;
  const auto x_var_type = x_var_type_;
  const auto y_var_type = y_var_type_;
  const auto x_edges = x_edges_;
  const auto y_edges = y_edges_;
  const auto x_edges_type = x_edges_type_;
  const auto y_edges_type = y_edges_type_;
  const auto x_edge_min = x_edge_min_;
  const auto x_edge_dbin = x_edge_dbin_;
  const auto y_edge_min = y_edge_min_;
  const auto y_edge_dbin = y_edge_dbin_;
  const auto hist_ndim = ndim_;
  const auto weight_by_vol = weight_by_vol_;
  const auto accumulate = accumulate_;
  auto result = result_;
  auto scatter = scatter_result;

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
                                       ? std::vector<std::string>{x_var_name_}
                                       : std::vector<std::string>{};
    const auto x_var = md->PackVariables(x_var_pack_string);

    const auto y_var_pack_string = y_var_type == VarType::Var
                                       ? std::vector<std::string>{y_var_name_}
                                       : std::vector<std::string>{};
    const auto y_var = md->PackVariables(y_var_pack_string);

    const auto binned_var_pack_string = binned_var_component == -1
                                            ? std::vector<std::string>{}
                                            : std::vector<std::string>{binned_var_name_};
    const auto binned_var = md->PackVariables(binned_var_pack_string);

    const auto weight_var_pack_string = weight_var_component == -1
                                            ? std::vector<std::string>{}
                                            : std::vector<std::string>{weight_var_name_};
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

          int x_bin = -1;
          // First handle edge cases explicitly
          if (x_val < x_edges(0)) {
            if (accumulate) {
              x_bin = 0;
            } else {
              return;
            }
          } else if (x_val > x_edges(x_edges.extent_int(0) - 1)) {
            if (accumulate) {
              x_bin = x_edges.extent_int(0) - 2;
            } else {
              return;
            }
            // if we're on the rightmost edge, directly set last bin
          } else if (x_val == x_edges(x_edges.extent_int(0) - 1)) {
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

          // needs to be zero as for the 1D histogram we need 0 as first index of the 2D
          // result array
          int y_bin = 0;
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

            y_bin = -1; // reset to impossible value
            // First handle edge cases explicitly
            if (y_val < y_edges(0)) {
              if (accumulate) {
                y_bin = 0;
              } else {
                return;
              }
            } else if (y_val > y_edges(y_edges.extent_int(0) - 1)) {
              if (accumulate) {
                y_bin = y_edges.extent_int(0) - 2;
              } else {
                return;
              }
              // if we're on the rightmost edge, directly set last bin
            } else if (y_val == y_edges(y_edges.extent_int(0) - 1)) {
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

  hist_names_ = pin->GetVector<std::string>(op.block_name, "hist_names");

  for (auto &hist_name : hist_names_) {
    histograms_.emplace_back(pin, op.block_name, hist_name);
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
    hist.CalcHist(pm);
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
    HDF5WriteAttribute("hist_names", hist_names_, info_group);

    for (auto &hist : histograms_) {
      const H5G hist_group = MakeGroup(file, "/" + hist.name_);
      HDF5WriteAttribute("ndim", hist.ndim_, hist_group);
      HDF5WriteAttribute("x_var_name", hist.x_var_name_.c_str(), hist_group);
      HDF5WriteAttribute("x_var_component", hist.x_var_component_, hist_group);
      HDF5WriteAttribute("binned_var_name", hist.binned_var_name_.c_str(), hist_group);
      HDF5WriteAttribute("binned_var_component", hist.binned_var_component_, hist_group);

      const auto x_edges_h = hist.x_edges_.GetHostMirrorAndCopy();
      local_count[0] = global_count[0] = x_edges_h.extent_int(0);
      HDF5Write1D(hist_group, "x_edges", x_edges_h.data(), local_offset.data(),
                  local_count.data(), global_count.data(), pl_xfer);

      if (hist.ndim_ == 2) {
        HDF5WriteAttribute("y_var_name", hist.y_var_name_.c_str(), hist_group);
        HDF5WriteAttribute("y_var_component", hist.y_var_component_, hist_group);

        const auto y_edges_h = hist.y_edges_.GetHostMirrorAndCopy();
        local_count[0] = global_count[0] = y_edges_h.extent_int(0);
        HDF5Write1D(hist_group, "y_edges", y_edges_h.data(), local_offset.data(),
                    local_count.data(), global_count.data(), pl_xfer);
      }

      const auto hist_h = hist.result_.GetHostMirrorAndCopy();
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
      if (hist.ndim_ == 2) {
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
