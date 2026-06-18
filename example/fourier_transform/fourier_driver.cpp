//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2026 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================

// This file was made in part with generative AI.

#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "fourier_driver.hpp"
#include <parthenon/driver.hpp>

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
  auto &rc = pmb->meshblock_data.Get();
  auto field = rc->Get("test_field").data;

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  auto &coords = pmb->coords;

  pmb->par_for(
      PARTHENON_AUTO_LABEL, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        field(k, j, i) = Kokkos::sin(coords.Xc<1>(i)) * Kokkos::sin(coords.Xc<2>(j)) *
                         Kokkos::sin(coords.Xc<3>(k));
      });
}

Packages_t ProcessPackages(std::unique_ptr<ParameterInput> &pin) {
  Packages_t packages;

  auto package = std::make_shared<parthenon::StateDescriptor>("fourier_transform");

  // Register a scalar field for FFT round-trip test
  parthenon::Metadata m({parthenon::Metadata::Cell, parthenon::Metadata::Derived,
                         parthenon::Metadata::OneCopy});
  package->AddField("test_field", m);

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

  Driver::PostExecute(DriverStatus::complete);
  return DriverStatus::complete;
}
