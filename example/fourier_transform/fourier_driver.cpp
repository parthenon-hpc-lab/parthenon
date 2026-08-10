//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2026 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
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

// This file was made in part with generative AI.

#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <parthenon/driver.hpp>

#include "fourier_driver.hpp"
#include "utils/calc_spectrum.hpp"

using namespace parthenon::driver::prelude;
using fourier_transform::FourierDriver;

Packages_t ProcessPackages(std::unique_ptr<ParameterInput> &pin);
void FillTestField(MeshBlock *pmb, ParameterInput *pin);

int main(int argc, char *argv[]) {
  ParthenonManager pman;
  pman.app_input->ProcessPackages = ProcessPackages;
  pman.app_input->ProblemGenerator = FillTestField;

  auto manager_status = pman.ParthenonInitEnv(argc, argv);
  if (manager_status == ParthenonStatus::complete) {
    pman.ParthenonFinalize();
    return 0;
  }
  if (manager_status == ParthenonStatus::error) {
    pman.ParthenonFinalize();
    return 1;
  }

  pman.ParthenonInitPackagesAndMesh();
  {
    FourierDriver driver(pman.pinput.get(), pman.app_input.get(), pman.pmesh.get());
    driver.Execute();
  }
  pman.ParthenonFinalize();
  return 0;
}

// Initialize a simple test field. Note that FFTs only work on a uniform grid, no AMR.
void FillTestField(MeshBlock *pmb, ParameterInput *pin) {
  auto &mbd = pmb->meshblock_data.Get();
  auto field = mbd->Get("test_field").data;
  auto vec_field = mbd->Get("test_vector_field").data;

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  auto &coords = pmb->coords;

  pmb->par_for(
      PARTHENON_AUTO_LABEL, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        // smooth pattern with some "perturbations"
        field(k, j, i) = Kokkos::sin(coords.Xc<1>(i)) * Kokkos::sin(coords.Xc<2>(j)) *
                             Kokkos::sin(coords.Xc<3>(k)) +
                         0.1 * (k + j + i) / (kb.e + jb.e + ib.e);
        vec_field(0, k, j, i) =
            Kokkos::sin(coords.Xc<1>(i)) * Kokkos::sin(coords.Xc<2>(j)) *
                Kokkos::sin(coords.Xc<3>(k)) +
            0.1 * (k + j + i) / (kb.e + jb.e + ib.e) + 1; // added mean component
        vec_field(1, k, j, i) = Kokkos::sin(coords.Xc<1>(i)) *
                                    Kokkos::cos(coords.Xc<2>(j)) *
                                    Kokkos::sin(coords.Xc<3>(k)) +
                                0.1 * (k + j + i) / (kb.e + jb.e + ib.e);
        vec_field(2, k, j, i) = Kokkos::sin(coords.Xc<1>(i)) *
                                    Kokkos::sin(coords.Xc<2>(j)) *
                                    Kokkos::cos(coords.Xc<3>(k)) +
                                0.1 * (k + j + i) / (kb.e + jb.e + ib.e);
      });
}

Packages_t ProcessPackages(std::unique_ptr<ParameterInput> &pin) {
  Packages_t packages;

  auto package = std::make_shared<parthenon::StateDescriptor>("fourier_transform");

  // Register a scalar field for FFT round-trip test
  parthenon::Metadata m({parthenon::Metadata::Cell, parthenon::Metadata::Derived,
                         parthenon::Metadata::OneCopy});
  package->AddField("test_field", m);

  // Register a vector field for FFT round-trip test
  m = parthenon::Metadata({parthenon::Metadata::Cell, parthenon::Metadata::Derived,
                           parthenon::Metadata::OneCopy},
                          std::vector<int>({3}));
  package->AddField("test_vector_field", m);

  packages.Add(package);
  return packages;
}

parthenon::DriverStatus FourierDriver::Execute() {
  PreExecute();

  auto &md = pmesh->mesh_data.Get();
  auto UniformGridHelper =
      pmesh->GetUniformGridHelper(); // Helper class used to map block-local indices to a
                                     // flat mesh index
  auto FFTManager = pmesh->GetFFTManager(); // Class that holds and executes FFT plans

  // define input and output arrays for FFT:
  parthenon::ParArray1D<Real> input("fft input", FFTManager->size_real_space_box());
  parthenon::ParArray1D<Kokkos::complex<Real>> output(
      "fft output", FFTManager->size_fourier_space_box());
  // also pre-allocate array for the recovered field after inverse FFT, to check
  // round-trip accuracy:
  parthenon::ParArray1D<Real> input_recovered("fft input recovered",
                                              FFTManager->size_real_space_box());

  auto test_field = md->PackVariables(std::vector<std::string>{"test_field"});

  // Gather block data into flat array for FFT input:
  UniformGridHelper->GatherField("test_field", 0, input);

  // Perform forward FFT - applies 1/N^3 normalization:
  FFTManager->Forward(input.data(), output.data());
  // Perform inverse FFT - applies no normalization:
  FFTManager->Backward(output.data(), input_recovered.data());

  // Check round-trip accuracy (get max difference across all points):
  Real local_max_error = 0.0;
  Kokkos::parallel_reduce(
      "ComputeError", Kokkos::RangePolicy<>(0, FFTManager->size_real_space_box()),
      KOKKOS_LAMBDA(const int idx, Real &max_err) {
        Real diff = Kokkos::abs(input_recovered(idx) - input(idx));
        if (diff > max_err) max_err = diff;
      },
      Kokkos::Max<Real>(local_max_error));

  // Reduce across MPI ranks to get global maximum
  Real max_error = local_max_error;
#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(MPI_Allreduce(&local_max_error, &max_error, 1, MPI_PARTHENON_REAL,
                                    MPI_MAX, MPI_COMM_WORLD));
#endif

  if (parthenon::Globals::my_rank == 0) {
    std::cout << "Max relative error after FFT round-trip: " << max_error << std::endl;
  }

  // Now test the spectrum machinery
  auto spectrum =
      parthenon::utils::fft::CalcSpectrum(pmesh, "test_vector_field", {0, 1, 2});
  const auto spectrum_h = spectrum.GetHostMirrorAndCopy();

  auto test_vector_field_pack =
      md->PackVariables(std::vector<std::string>{"test_vector_field"});
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  auto mesh_size = pmesh->mesh_size;
  const auto Nx = mesh_size.nx(parthenon::X1DIR);
  const auto Ny = mesh_size.nx(parthenon::X2DIR);
  const auto Nz = mesh_size.nx(parthenon::X3DIR);

  // Sanity checks (compare power in real space to spectral space power)
  using parthenon::utils::fft::SpecReal;
  Kokkos::Array<SpecReal, 4> sums{{0.0, 0.0, 0.0, 0.0}};
  Kokkos::parallel_reduce(
      "fieldsqrd_sum",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
          {0, kb.s, jb.s, ib.s},
          {test_vector_field_pack.GetDim(5), kb.e + 1, jb.e + 1, ib.e + 1},
          {1, 1, 1, ib.e + 1 - ib.s}),
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i,
                    SpecReal &sum_usqr, SpecReal &sum_u1, SpecReal &sum_u2,
                    SpecReal &sum_u3) {
        const auto u1 = static_cast<SpecReal>(test_vector_field_pack(b, 0, k, j, i));
        const auto u2 = static_cast<SpecReal>(test_vector_field_pack(b, 1, k, j, i));
        const auto u3 = static_cast<SpecReal>(test_vector_field_pack(b, 2, k, j, i));
        sum_u1 += u1;
        sum_u2 += u2;
        sum_u3 += u3;
        sum_usqr += SQR(u1) + SQR(u2) + SQR(u3);
      },
      sums[0], sums[1], sums[2], sums[3]);

#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(
      MPI_Allreduce(MPI_IN_PLACE, sums.data(), 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
#endif
  const auto norm =
      static_cast<SpecReal>(Nx) * static_cast<SpecReal>(Ny) * static_cast<SpecReal>(Nz);
  sums[0] /= norm;
  sums[1] /= norm;
  sums[2] /= norm;
  sums[3] /= norm;

  // Sum power in spectrum
  SpecReal spec_sum = 0.0;
  for (int i = 0; i < static_cast<int>(spectrum_h.extent(0)); i++) {
    spec_sum += spectrum_h(i, 0);
  }
  if (parthenon::Globals::my_rank == 0) {
    std::cout << "sum u^2=" << sums[0] << " sum uhat^2=" << spec_sum
              << " <u>^2=" << SQR(sums[1]) + SQR(sums[2]) + SQR(sums[3])
              << " uhat(0)^2=" << spectrum_h(0, 0) << " sum u_1=" << sums[1]
              << " sum u_2=" << sums[2] << " sum u_3=" << sums[3] << "\n";
    std::cout << std::format(
        "Error in spectrum total power: {:.15e}\nError in spectrum mean: {:.15e}\n",
        std::abs(sums[0] / spec_sum - 1.0),
        std::abs((SQR(sums[1]) + SQR(sums[2]) + SQR(sums[3])) / spectrum_h(0, 0) - 1.0));
  }

  Driver::PostExecute(DriverStatus::complete);
  return DriverStatus::complete;
}
