//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2026 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================

// This file was made in part with generative AI.

#include "utils/calc_spectrum.hpp"

#include <cmath>
#include <string>
#include <vector>

#include "defs.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "parthenon_arrays.hpp"
#include "utils/error_checking.hpp"
#include "utils/fft_manager.hpp"
#include "utils/uniform_grid_helper.hpp"

namespace parthenon {
namespace utils {
namespace fft {
parthenon::ParArray2D<SpecReal>
CalcSpectrum(Mesh *pm, const parthenon::ParArray1D<Real> &input, const int n_comp) {
  PARTHENON_REQUIRE_THROWS(pm != nullptr, "CalcSpectrum: mesh pointer must not be null");
  PARTHENON_REQUIRE_THROWS(n_comp > 0, "CalcSpectrum: n_comp must be positive");
  PARTHENON_REQUIRE_THROWS(pm->DefaultNumPartitions() == 1,
                           "Only num_packs=1 currently supported for heffte.")

  auto FFTManager = pm->GetFFTManager();
  const auto fft_size_inbox = FFTManager->size_real_space_box();
  const auto expected_input_size = static_cast<std::size_t>(n_comp) * fft_size_inbox;
  PARTHENON_REQUIRE_THROWS(input.size() == expected_input_size,
                           "CalcSpectrum: input array has size " +
                               std::to_string(input.size()) +
                               ", but expected n_comp * size_real_space_box() = " +
                               std::to_string(expected_input_size));

  auto mesh_size = pm->mesh_size;
  const auto nx = mesh_size.nx(X1DIR);
  const auto ny = mesh_size.nx(X2DIR);
  const auto nz = mesh_size.nx(X3DIR);

  parthenon::ParArray1D<Kokkos::complex<Real>> output(
      "fft output", n_comp * FFTManager->size_fourier_space_box());

  for (int i = 0; i < n_comp; i++) {
    FFTManager->Forward(input.data() + i * FFTManager->size_real_space_box(),
                        output.data() + i * FFTManager->size_fourier_space_box());
  }

  const auto k_max = std::sqrt(SQR(nx / 2) + SQR(ny / 2) + SQR(nz / 2));
  const auto num_bins = static_cast<int>(std::ceil(k_max)) + 1;

  parthenon::ParArray2D<SpecReal> spectra("spectra", num_bins, 3);
  auto scatter_spectra =
      Kokkos::Experimental::ScatterView<SpecReal **, parthenon::LayoutWrapper>(
          spectra.KokkosView());

  auto fb = FFTManager->fourier_space_box();

  const auto fft_size_outbox = FFTManager->size_fourier_space_box();
  auto kernel_helper = FFTManager->GetKernelHelper();
  parthenon::par_for(
      "CalcSpec", fb.low[2], fb.high[2], fb.low[1], fb.high[1], fb.low[0], fb.high[0],
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        auto k_vec = kernel_helper.Wavevector(k, j, i);
        auto k_mag = std::sqrt(SQR(k_vec[0]) + SQR(k_vec[1]) + SQR(k_vec[2]));
        auto k_mag_int = static_cast<int>(std::floor(k_mag));
        const auto outidx = kernel_helper.FourierFlatIndex(k, j, i);
        auto val = 0.0;
        for (int n = 0; n < n_comp; n++) {
          val += SQR(output[outidx + n * fft_size_outbox].real()) +
                 SQR(output[outidx + n * fft_size_outbox].imag());
        }
        const auto fac = ((k_vec[2] > 0) && (2 * k_vec[2] != nx)) ? 2.0 : 1.0;
        auto spec = scatter_spectra.access();
        spec(k_mag_int, 0) += fac * val;
        spec(k_mag_int, 1) += fac * k_mag;
        spec(k_mag_int, 2) += fac * 1.0;
      });

  Kokkos::Experimental::contribute(spectra.KokkosView(), scatter_spectra);
  Kokkos::fence();

#ifdef MPI_PARALLEL
  PARTHENON_REQUIRE_THROWS(sizeof(SpecReal) == sizeof(double),
                           "Need to fix comm data types manually.");
  if (parthenon::Globals::my_rank == 0) {
    PARTHENON_MPI_CHECK(MPI_Reduce(MPI_IN_PLACE, spectra.data(), spectra.size(),
                                   MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD));
  } else {
    PARTHENON_MPI_CHECK(MPI_Reduce(spectra.data(), spectra.data(), spectra.size(),
                                   MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD));
  }
#endif

  return spectra;
}

parthenon::ParArray2D<SpecReal> CalcSpectrum(Mesh *pm, const std::string &var_name,
                                             const std::vector<int> &components) {
  PARTHENON_REQUIRE_THROWS(pm != nullptr, "CalcSpectrum: mesh pointer must not be null");
  PARTHENON_REQUIRE_THROWS(!components.empty(),
                           "CalcSpectrum: at least one component is required");

  auto &md = pm->mesh_data.Get();

  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  auto vars = md->PackVariables(std::vector<std::string>{var_name});

  const int n_comp = components.size();
  auto FFTManager = pm->GetFFTManager();
  const auto fft_size_inbox = FFTManager->size_real_space_box();
  parthenon::ParArray1D<Real> input("fft input", n_comp * fft_size_inbox);

  parthenon::ParArray1D<int> components_d("components", components.size());
  auto components_h = components_d.GetHostMirror();
  for (int n = 0; n < n_comp; n++) {
    PARTHENON_REQUIRE_THROWS(components[n] >= 0 && components[n] < vars.GetDim(4),
                             "CalcSpectrum: component " + std::to_string(components[n]) +
                                 " out of range for variable '" + var_name + "'");
    components_h(n) = components[n];
  }
  components_d.DeepCopy(components_h);

  auto UniformGridHelper = pm->GetUniformGridHelper();
  auto helper = UniformGridHelper->GetKernelHelper();

  par_for(
      "Init FFT fields", 0, md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto idx = helper.FlatIndex(b, k, j, i);
        for (int n = 0; n < n_comp; n++) {
          input(n * fft_size_inbox + idx) = vars(b, components_d(n), k, j, i);
        }
      });

  return CalcSpectrum(pm, input, n_comp);
}
} // namespace fft
} // namespace utils
} // namespace parthenon
