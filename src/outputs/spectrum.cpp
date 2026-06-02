//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2023-2025 The Parthenon collaboration
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
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "coordinates/coordinates.hpp"
#include "defs.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "outputs/outputs.hpp"
#include "parthenon_arrays.hpp"
#include "utils/error_checking.hpp"
#include "utils/FFTManager.hpp"

namespace parthenon {

//----------------------------------------------------------------------------------------
//! \fn void OutputType::SpectralOutput()
//  \brief Writes a spectrum output file

void SpectralOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin, SimTime *tm,
                                     const SignalHandler::OutputSignal signal) {
                                        
  const auto var_name = pin->GetString(output_params.block_name, "variable");
  const auto components = pin->GetVector<int>(output_params.block_name, "components");
  const auto output_label = pin->GetOrAddString(output_params.block_name, "output_label", var_name);

  auto &md = pm->mesh_data.Get();

  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  auto vars = md->PackVariables(std::vector<std::string>{var_name});

  // Get Mesh geometry information: 
  auto UniformGridHelper = pm->GetUniformGridHelper();
  const auto Nx = UniformGridHelper->global_mesh_size[0];
  const auto Ny = UniformGridHelper->global_mesh_size[1];
  const auto Nz = UniformGridHelper->global_mesh_size[2];

  // Initialize FFTManager and I/O arrays:
  int n_comp = components.size(); // number of field components to transform 
  auto FFTManager = pm->GetFFTManager(); 
  const auto fft_size_inbox = FFTManager->size_real_space_box();
  parthenon::ParArray1D<Real> input("fft input", n_comp * fft_size_inbox);
  parthenon::ParArray1D<std::complex<Real>> output("fft output",
                                                   n_comp * FFTManager->size_fourier_space_box());
  PARTHENON_REQUIRE_THROWS(pm->DefaultNumPartitions() == 1, 
                           "Only pack_size=-1 currently supported for heffte.") // pack size -1 means 1 pack per rank

  // copy components to device
  parthenon::ParArray1D<int> components_d("components", components.size());
  auto components_h = components_d.GetHostMirror();
  for (int n = 0; n < n_comp; n++) components_h(n) = components[n];
  components_d.DeepCopy(components_h);

  auto helper = UniformGridHelper->GetKernelHelper();

  // Gather block data into flat arrays for FFT input: 
  par_for(
      "Init FFT fields", 0, pm->GetNumMeshBlocksThisRank() - 1, kb.s, kb.e, jb.s, jb.e,
      ib.s, ib.e, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto idx = helper.FlatIndex(b, k, j, i);
        for (int n = 0; n < n_comp; n++) {
          input(n * fft_size_inbox + idx) = vars(b, components_d(n), k, j, i);
        }
      });

  for (int i = 0; i < n_comp; i++) {
    FFTManager->Forward(input.data() + i * FFTManager->size_real_space_box(),
                        output.data() + i * FFTManager->size_fourier_space_box());
  }

  const auto k_max = std::sqrt(SQR(Nx / 2) + SQR(Ny / 2) + SQR(Nz / 2));

  const auto num_bins = static_cast<int>(std::ceil(k_max)) + 1;
  // TODO(pgrete) if these are being reused, then ensure to reset (i.e., init 0 to and
  // call .reset())
  parthenon::ParArray2D<Real> spectra("spectra", num_bins, 3);
  // temp view for reduction for better performance (switches
  // between atomics and data duplication depending on the platform)
  auto scatter_spectra =
      Kokkos::Experimental::ScatterView<Real **, parthenon::LayoutWrapper>(
          spectra.KokkosView());

  auto fb = FFTManager->fourier_space_box();

  // Calculate spectrum: 
  const auto fft_size_outbox = FFTManager->size_fourier_space_box();
  auto kernel_helper = FFTManager->GetKernelHelper();
  parthenon::par_for(
      "CalcSpec", fb.low[2], fb.high[2], fb.low[1], fb.high[1], fb.low[0], fb.high[0],
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        auto k_vec = kernel_helper.Wavevector(k, j, i);

        // for simple binning/indexing
        auto k_mag = std::sqrt(SQR(k_vec[0]) + SQR(k_vec[1]) + SQR(k_vec[2]));
        auto k_mag_int = static_cast<int>(std::floor(k_mag));

        const auto outidx = kernel_helper.FourierFlatIndex(k, j, i);

        auto val = 0.0;
        for (int n = 0; n < n_comp; n++) {
          val += SQR(output[outidx + n * fft_size_outbox].real()) +
                 SQR(output[outidx + n * fft_size_outbox].imag());
        }

        // account for Hermitian symmetry of r2c transform
        const auto fac = ((k_vec[0] > 0) && (2 * k_vec[0] != Nx)) ? 2.0 : 1.0;

        auto spec = scatter_spectra.access();
        // 0: histsum - 1: ksum - 2: histcount
        spec(k_mag_int, 0) += fac * val;
        spec(k_mag_int, 1) += fac * k_mag;
        spec(k_mag_int, 2) += fac * 1.0;
      });
      
  Kokkos::Experimental::contribute(spectra.KokkosView(), scatter_spectra);

  Kokkos::fence(); // May not be required.
#ifdef MPI_PARALLEL
  //  Sum the perturbations over all processors
  if (parthenon::Globals::my_rank == 0) {
    PARTHENON_MPI_CHECK(MPI_Reduce(MPI_IN_PLACE, spectra.data(), spectra.size(),
                                   MPI_PARTHENON_REAL, MPI_SUM, 0, MPI_COMM_WORLD));
  } else {
    PARTHENON_MPI_CHECK(MPI_Reduce(spectra.data(), spectra.data(), spectra.size(),
                                   MPI_PARTHENON_REAL, MPI_SUM, 0, MPI_COMM_WORLD));
  }
#endif // MPI_PARALLEL

  auto spectra_h = spectra.GetHostMirrorAndCopy(); // spectra_h is the Spectral data (Parthenon array) on host

  // Write spectrum to ordinary text file:
  if (parthenon::Globals::my_rank == 0) {
    
    std::string fname;
    fname.assign(output_params.file_basename);
    fname.append("."+output_label+".");
    fname.append(output_params.file_id);
    fname.append(".");
    if (signal == SignalHandler::OutputSignal::now) {
    fname.append("now");
  } else if (signal == SignalHandler::OutputSignal::final &&
             output_params.file_label_final) {
    fname.append("final");
    // default time based data dump
  } else {
    std::stringstream file_number;
    file_number << std::setw(output_params.file_number_width) << std::setfill('0')
                << output_params.file_number;
    fname.append(file_number.str());
  }
    fname.append(".spc");

    std::ofstream fout(fname);
    if (!fout.is_open()) {
        PARTHENON_FAIL("Could not open " + fname + " for writing");
    }

    // Write each bin's results to the file: val_sum, k_sum, count_sum
    fout << "# Bin    val_sum    K_sum    Count\n";
    for (int i = 0; i < num_bins; ++i) {
        fout << i << " "
             << spectra_h(i, 0) << " "
             << spectra_h(i, 1) << " "
             << spectra_h(i, 2) << "\n";
    }
    fout.close();
    }


  // advance output parameters
  UpdateNextOutput_(pm, tm);  

} // void SpectralOutput::WriteOutputFile

} // namespace parthenon