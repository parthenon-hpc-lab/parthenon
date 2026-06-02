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

// OpenPMD headers
#include <openPMD/openPMD.hpp>

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
//! \fn void HelicitySliceOutput::WriteOutputFile()
//  \brief Writes a helicity slice output file

void HelicitySliceOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin, SimTime *tm,
                                          const SignalHandler::OutputSignal signal) {

  auto &md = pm->mesh_data.Get();

  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);
  auto cons = md->PackVariables(std::vector<std::string>{"cons"});

  auto UniformGridHelper = pm->GetUniformGridHelper();
  auto &loc_view = UniformGridHelper->loc_view;
  const auto &block_size = UniformGridHelper->block_size;
  const auto &local_mesh_size = UniformGridHelper->local_mesh_size;
  const auto Nx = UniformGridHelper->global_mesh_size[0];
  const auto Ny = UniformGridHelper->global_mesh_size[1];
  const auto Nz = UniformGridHelper->global_mesh_size[2];

  // Read output mode from input file:
  // output_mode = "slice" (default): write midplane z-slice
  // output_mode = "full": write full 3D helicity field
  const auto output_mode = pin->GetOrAddString(output_params.block_name,
                                               "output_mode", "slice");
  const bool full_output = (output_mode == "full");
  PARTHENON_REQUIRE_THROWS(output_mode == "slice" || output_mode == "full",
                           "helicity output_mode must be 'slice' or 'full'.");

  PARTHENON_REQUIRE_THROWS(pm->DefaultNumPartitions() == 1,
                           "Only pack_size=-1 currently supported for helicity slice.");

  auto FFTManager = pm->GetFFTManager();
  const auto fft_size_inbox  = FFTManager->size_real_space_box();
  const auto fft_size_outbox = FFTManager->size_fourier_space_box();

  // ------------------------------------------------------------------
  // 1. Pack B into real-space FFT arrays
  // ------------------------------------------------------------------
  parthenon::ParArray1D<Real> input("hel_fft_input", 3 * fft_size_inbox);

  par_for(
      "HelSlice_PackB", 0, pm->GetNumMeshBlocksThisRank() - 1,
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto idx = UniformGridHelper->FlatIndex(b, k, j, i);
        input(idx)                    = cons(b, 5, k, j, i); // Bx
        input(idx + fft_size_inbox)   = cons(b, 6, k, j, i); // By
        input(idx + 2*fft_size_inbox) = cons(b, 7, k, j, i); // Bz
      });

  // ------------------------------------------------------------------
  // 2. Forward FFT B
  // ------------------------------------------------------------------
  parthenon::ParArray1D<std::complex<Real>> B_hat("hel_B_hat", 3 * fft_size_outbox);

  for (int i = 0; i < 3; i++) {
    FFTManager->Forward(input.data()  + i * fft_size_inbox,
                        B_hat.data()  + i * fft_size_outbox);
  }

  // ------------------------------------------------------------------
  // 3. Compute A_hat = i * (k x B_hat) / k^2  (Coulomb gauge)
  // ------------------------------------------------------------------
  const auto x1min = pin->GetReal("parthenon/mesh", "x1min");
  const auto x1max = pin->GetReal("parthenon/mesh", "x1max");
  const Real Lx = x1max - x1min;
  PARTHENON_REQUIRE_THROWS(Lx > 0.0, "Box size Lx must be positive.");

  parthenon::ParArray1D<Kokkos::complex<Real>> A_hat("hel_A_hat", 3 * fft_size_outbox);

  auto outbox = FFTManager->fourier_space_box();

  ib.s = outbox.low[0];  ib.e = outbox.high[0];
  jb.s = outbox.low[1];  jb.e = outbox.high[1];
  kb.s = outbox.low[2];  kb.e = outbox.high[2];

  PARTHENON_REQUIRE_THROWS(
      (std::int64_t)(ib.e - ib.s + 1) * (jb.e - jb.s + 1) * (kb.e - kb.s + 1)
          == (std::int64_t)fft_size_outbox,
      "Fourier space box size does not match fft_size_outbox.");

  const Kokkos::complex<Real> imag_unit(0.0, 1.0);

  auto B_hat_kk = reinterpret_cast<Kokkos::complex<Real>*>(B_hat.data());
  auto A_hat_kk = reinterpret_cast<Kokkos::complex<Real>*>(A_hat.data());

  parthenon::par_for(
      "HelSlice_ComputeAhat", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int kz_idx, const int ky_idx, const int kx_idx) {
        const auto kz = kz_idx <= Nz/2 ? kz_idx : kz_idx - Nz;
        const auto ky = ky_idx <= Ny/2 ? ky_idx : ky_idx - Ny;
        const auto kx = kx_idx;

        const Real kx_phys = 2.0 * M_PI * kx / Lx;
        const Real ky_phys = 2.0 * M_PI * ky / Lx;
        const Real kz_phys = 2.0 * M_PI * kz / Lx;
        const Real k2 = kx_phys*kx_phys + ky_phys*ky_phys + kz_phys*kz_phys;

        const std::int64_t idx =
            ((std::int64_t)(kz_idx - kb.s) * (jb.e - jb.s + 1) + (ky_idx - jb.s))
            * (ib.e - ib.s + 1) + kx_idx - ib.s;

        PARTHENON_DEBUG_REQUIRE(idx >= 0 && idx < (std::int64_t)fft_size_outbox,
                                "ComputeAhat: idx out of bounds");

        if (kx == 0 && ky == 0 && kz == 0) {
          A_hat_kk[idx]                     = 0.0;
          A_hat_kk[idx + fft_size_outbox]   = 0.0;
          A_hat_kk[idx + 2*fft_size_outbox] = 0.0;
          return;
        }

        const auto Bx_k = B_hat_kk[idx];
        const auto By_k = B_hat_kk[idx + fft_size_outbox];
        const auto Bz_k = B_hat_kk[idx + 2*fft_size_outbox];

        A_hat_kk[idx]                     = imag_unit * (ky_phys*Bz_k - kz_phys*By_k) / k2;
        A_hat_kk[idx + fft_size_outbox]   = imag_unit * (kz_phys*Bx_k - kx_phys*Bz_k) / k2;
        A_hat_kk[idx + 2*fft_size_outbox] = imag_unit * (kx_phys*By_k - ky_phys*Bx_k) / k2;
      });

  Kokkos::fence();

  // ------------------------------------------------------------------
  // 4. Inverse FFT to get A in real space
  // ------------------------------------------------------------------
  parthenon::ParArray1D<Real> A_real("hel_A_real", 3 * fft_size_inbox);

  for (int i = 0; i < 3; i++) {
    FFTManager->Backward(
        reinterpret_cast<std::complex<Real>*>(A_hat.data()) + i * fft_size_outbox,
        A_real.data() + i * fft_size_inbox);
  }

  Kokkos::fence();

  // ------------------------------------------------------------------
  // 5. Compute h = A.B
  // ------------------------------------------------------------------
  auto realbox = FFTManager->real_space_box();
  const int kz_mid_global = Nz / 2;

  const int local_nx = realbox.size[0];
  const int local_ny = realbox.size[1];
  const int local_nz = realbox.size[2];

  auto A_real_h = A_real.GetHostMirrorAndCopy();
  auto input_h  = input.GetHostMirrorAndCopy();

  // ------------------------------------------------------------------
  // 6. Write via OpenPMD
  // ------------------------------------------------------------------
  using openPMD::Access;
  using openPMD::Series;

  std::string fname;
  fname.assign(output_params.file_basename);
  fname.append(".");
  fname.append(output_params.file_id);
  fname.append(".");
  if (signal == SignalHandler::OutputSignal::now) {
    fname.append("now");
  } else if (signal == SignalHandler::OutputSignal::final &&
             output_params.file_label_final) {
    fname.append("final");
  } else {
    std::stringstream file_number;
    file_number << std::setw(output_params.file_number_width)
                << std::setfill('0') << output_params.file_number;
    fname.append(file_number.str());
  }
  fname.append(".bp");

  Series series = Series(fname, Access::CREATE,
#ifdef MPI_PARALLEL
                         MPI_COMM_WORLD,
#endif
                         "{}");

  auto it = series.iterations[output_params.file_number];
  it.open();

  if (tm != nullptr) {
    it.setTime(tm->time);
    it.setDt(tm->dt);
    it.setAttribute("NCycle", tm->ncycle);
  }

  auto mesh_record = it.meshes["helicity_density"];
  mesh_record.setGeometry(openPMD::Mesh::Geometry::cartesian);
  mesh_record.setDataOrder(openPMD::Mesh::DataOrder::C);

  const Real dx = Lx / Nx;
  auto comp = mesh_record[openPMD::MeshRecordComponent::SCALAR];
  comp.setPosition(std::vector<Real>{0.5, 0.5});

  if (full_output) {
    // ------------------------------------------------------------------
    // Full 3D output
    // ------------------------------------------------------------------
    mesh_record.setGridSpacing(std::vector<Real>{dx, dx, dx});
    mesh_record.setAxisLabels({"z", "y", "x"});
    mesh_record.setGridGlobalOffset({x1min, x1min, x1min});

    openPMD::Extent global_extent = {static_cast<uint64_t>(Nz),
                                     static_cast<uint64_t>(Ny),
                                     static_cast<uint64_t>(Nx)};
    auto dataset = openPMD::Dataset(openPMD::determineDatatype<Real>(), global_extent);
    comp.resetDataset(dataset);

    // compute full local helicity volume
    std::vector<Real> local_volume((std::size_t)local_nx * local_ny * local_nz, 0.0);

    for (int kk = 0; kk < local_nz; kk++) {
      for (int jj = 0; jj < local_ny; jj++) {
        for (int ii = 0; ii < local_nx; ii++) {
          const std::int64_t idx = (std::int64_t)kk * local_ny * local_nx
                                 + (std::int64_t)jj * local_nx + ii;
          PARTHENON_REQUIRE_THROWS(idx >= 0 && idx < (std::int64_t)fft_size_inbox,
                                   "Full helicity: idx out of bounds");
          const Real Ax = A_real_h[idx];
          const Real Ay = A_real_h[idx + fft_size_inbox];
          const Real Az = A_real_h[idx + 2*fft_size_inbox];
          const Real Bx = input_h[idx];
          const Real By = input_h[idx + fft_size_inbox];
          const Real Bz = input_h[idx + 2*fft_size_inbox];
          local_volume[(std::size_t)kk * local_ny * local_nx
                     + (std::size_t)jj * local_nx + ii] = Ax*Bx + Ay*By + Az*Bz;
        }
      }
    }

    // each rank writes its local chunk
    PARTHENON_REQUIRE_THROWS(realbox.low[0] >= 0 && realbox.low[1] >= 0 && realbox.low[2] >= 0,
                             "realbox offsets are negative.");
    openPMD::Offset chunk_offset = {static_cast<uint64_t>(realbox.low[2]),
                                    static_cast<uint64_t>(realbox.low[1]),
                                    static_cast<uint64_t>(realbox.low[0])};
    openPMD::Extent chunk_extent = {static_cast<uint64_t>(local_nz),
                                    static_cast<uint64_t>(local_ny),
                                    static_cast<uint64_t>(local_nx)};

    PARTHENON_REQUIRE_THROWS(
        chunk_offset[0] + chunk_extent[0] <= global_extent[0] &&
        chunk_offset[1] + chunk_extent[1] <= global_extent[1] &&
        chunk_offset[2] + chunk_extent[2] <= global_extent[2],
        "3D chunk offset + extent exceeds global extent.");

    comp.storeChunkRaw(local_volume.data(), chunk_offset, chunk_extent);

  } else {
    // ------------------------------------------------------------------
    // 2D midplane slice output
    // ------------------------------------------------------------------
    mesh_record.setGridSpacing(std::vector<Real>{dx, dx});
    mesh_record.setAxisLabels({"y", "x"});
    mesh_record.setGridGlobalOffset({x1min, x1min});
    comp.setPosition(std::vector<Real>{0.5, 0.5});

    openPMD::Extent global_extent = {static_cast<uint64_t>(Ny),
                                     static_cast<uint64_t>(Nx)};
    auto dataset = openPMD::Dataset(openPMD::determineDatatype<Real>(), global_extent);
    comp.resetDataset(dataset);

    const bool rank_owns_midplane = (kz_mid_global >= realbox.low[2] &&
                                     kz_mid_global <= realbox.high[2]);

    PARTHENON_REQUIRE_THROWS(
        (std::int64_t)local_nx * local_ny <= std::numeric_limits<int>::max(),
        "local_nx * local_ny overflows int.");

    std::vector<Real> local_slice((std::size_t)local_nx * local_ny, 0.0);

    if (rank_owns_midplane) {
      const int kz_local = kz_mid_global - realbox.low[2];
      PARTHENON_REQUIRE_THROWS(kz_local >= 0 && kz_local < nx3l,
                               "kz_local out of range for this rank.");

      for (int jj = 0; jj < local_ny; jj++) {
        for (int ii = 0; ii < local_nx; ii++) {
          const std::int64_t idx = (std::int64_t)kz_local * local_ny * local_nx
                                 + (std::int64_t)jj * local_nx + ii;
          PARTHENON_REQUIRE_THROWS(idx >= 0 && idx < (std::int64_t)fft_size_inbox,
                                   "Helicity slice: idx out of bounds");
          const Real Ax = A_real_h[idx];
          const Real Ay = A_real_h[idx + fft_size_inbox];
          const Real Az = A_real_h[idx + 2*fft_size_inbox];
          const Real Bx = input_h[idx];
          const Real By = input_h[idx + fft_size_inbox];
          const Real Bz = input_h[idx + 2*fft_size_inbox];
          local_slice[(std::size_t)jj * local_nx + ii] = Ax*Bx + Ay*By + Az*Bz;
        }
      }
    }

    if (rank_owns_midplane) {
      PARTHENON_REQUIRE_THROWS(realbox.low[0] >= 0 && realbox.low[1] >= 0,
                               "realbox offsets are negative.");
      openPMD::Offset chunk_offset = {static_cast<uint64_t>(realbox.low[1]),
                                      static_cast<uint64_t>(realbox.low[0])};
      openPMD::Extent chunk_extent = {static_cast<uint64_t>(local_ny),
                                      static_cast<uint64_t>(local_nx)};

      PARTHENON_REQUIRE_THROWS(
          chunk_offset[0] + chunk_extent[0] <= global_extent[0] &&
          chunk_offset[1] + chunk_extent[1] <= global_extent[1],
          "Chunk offset + extent exceeds global extent.");

      comp.storeChunkRaw(local_slice.data(), chunk_offset, chunk_extent);
    }
  }

  it.seriesFlush();
  it.close();
  series.close();

  UpdateNextOutput_(pm, tm);

} // void HelicitySliceOutput::WriteOutputFile

} // namespace parthenon