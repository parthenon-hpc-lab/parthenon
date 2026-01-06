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
                                        
  const auto spec_type = pin->GetInteger("parthenon/output4", "spec_type"); // How to make this work so that it doesn't have to be output4?

  auto &md = pm->mesh_data.Get();

  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);
  auto cons = md->PackVariables(std::vector<std::string>{"cons"});

  // Check if we have a contiguous block of data (over all rank-local blocks)
  std::array local_loc_min{
      std::numeric_limits<std::int64_t>::max(),
      std::numeric_limits<std::int64_t>::max(),
      std::numeric_limits<std::int64_t>::max(),
  };
  std::array local_loc_max{
      std::numeric_limits<std::int64_t>::min(),
      std::numeric_limits<std::int64_t>::min(),
      std::numeric_limits<std::int64_t>::min(),
  };

  // Need to store this info in a way this can be used on device later
  parthenon::ParArray2D<std::int64_t> loc_view("logical location of local blocks",
                                               pm->GetNumMeshBlocksThisRank(), 3);
  auto loc_view_h = loc_view.GetHostMirror();

  // Set rank local min and max logical locations.
  // Also check if all blocks are on the same level (we use this check instead of
  // checking for refinement=none because AMR could have been used to dynamically refine
  // a simulation. We just need to ensure that all blocks are on the same level to
  // create an effective uniform grid.)
  const auto level =
      pm->Forest().GetLegacyTreeLocation(pm->block_list[0]->loc).level();
  for (int b = 0; b < pm->GetNumMeshBlocksThisRank(); b++) {
    auto pmb = pm->block_list[b];
    const auto loc = pm->Forest().GetLegacyTreeLocation(pmb->loc);
    for (int i = 0; i <= 2; i++) {
      local_loc_min.at(i) = std::min(loc.l(i), local_loc_min.at(i));
      local_loc_max.at(i) = std::max(loc.l(i), local_loc_max.at(i));
      loc_view_h(b, i) = loc.l(i);
    }
    PARTHENON_REQUIRE_THROWS(loc.level() == level,
                             "Not all blocks are on the same level.");
  }

  // convert global logical locations to rank-local logical locs
  for (int b = 0; b < pm->GetNumMeshBlocksThisRank(); b++) {
    for (int i = 0; i <= 2; i++) {
      loc_view_h(b, i) -= local_loc_min.at(i);
    }
  }
  Kokkos::deep_copy(loc_view, loc_view_h);

  std::array local_nlocs{
      (local_loc_max.at(0) - local_loc_min.at(0)) + 1,
      (local_loc_max.at(1) - local_loc_min.at(1)) + 1,
      (local_loc_max.at(2) - local_loc_min.at(2)) + 1,
  };
  const auto loc_max_vol = local_nlocs.at(0) * local_nlocs.at(1) * local_nlocs.at(2);
  // std::cerr << "[" << parthenon::Globals::my_rank << "] got local vol of: " <<
  // loc_max_vol << "\n";
  PARTHENON_REQUIRE_THROWS(loc_max_vol == pm->GetNumMeshBlocksThisRank(),
                           "Block coverage on rank cannot be matched to a contiguous "
                           "array, which is required for FFTs. Try a different amount of "
                           "ranks (one block per rank will always work).");

  // TODO(pgrete) not nice, make nicer
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  using backend_tag = heffte::backend::default_backend<heffte::tag::gpu>::type;
  PARTHENON_REQUIRE_THROWS(heffte::gpu::device_count() == 1,
                           "To make this work, we need to ensure that Kokkos and heffte "
                           "use the same GPUs. So hard fail for now.");
#else
  using backend_tag = heffte::backend::default_backend<heffte::tag::cpu>::type;
#endif

  if (parthenon::Globals::my_rank == 0)
    std::cerr << "using backend: " << heffte::backend::name<backend_tag>() << "\n";

  // Adjust (logical) grid size at levels other than the root level.
  // This is required for simulation with mesh refinement so that the phases calculated
  // below take the logical grid size into account. For example, the local phases at
  // level 1 should be calculated assuming a grid that is twice as large as the root
  // grid.

  // PARTHENON_REQUIRE_THROWS(!pm->adaptive, "Ask Luke about the logic here.");
  // const auto root_level = pm->GetRootLevel();
  // auto gnx1 =
  // static_cast<int>(pm->mesh_size.nx(X1DIR) * std::pow(2, level - root_level));
  // auto gnx2 =
  // static_cast<int>(pm->mesh_size.nx(X2DIR) * std::pow(2, level - root_level));
  // auto gnx3 =
  // static_cast<int>(pm->mesh_size.nx(X3DIR) * std::pow(2, level - root_level));

  // const auto gnx1 = pm->mesh_size.nx(X1DIR);
  // const auto gnx2 = pm->mesh_size.nx(X2DIR);
  // const auto gnx3 = pm->mesh_size.nx(X3DIR);
  // Determine global box sizes
  auto mesh_size = pm->mesh_size;
  auto Nx = mesh_size.nx(X1DIR);
  auto Ny = mesh_size.nx(X2DIR);
  auto Nz = mesh_size.nx(X3DIR);

  std::int64_t r2c_direction = 0; // the dimension where the data will shrink
  // construct global input/output boxes: 
  heffte::box3d<> real_indexes({0, 0, 0}, {Nx - 1, Ny - 1, Nz - 1});
  heffte::box3d<> complex_indexes({0, 0, 0}, {(Nx)/2, Ny - 1, Nz - 1});

  std::cout << "Defined heffte boxes\n";
  // check if the complex indexes have correct dimension
  assert(real_indexes.r2c(r2c_direction) == complex_indexes);
  std::cout << "Checked heffte boxes\n";
  // report the indexes
  if (parthenon::Globals::my_rank == 0) {
    std::cout << "The global input contains " << real_indexes.count()
              << " real indexes.\n";
    std::cout << "The global output contains " << complex_indexes.count()
              << " complex indexes.\n";
  }

  // Set local real indices based on the local infos
  // Need to use legacy locations from above (which are global) because locations now
  // are local to the tree, which results in inconsistencies for meshes with multiple
  // trees.
  const auto block_size = pm->GetDefaultBlockSize();
  // block sizes
  const auto nx1b = block_size.nx(X1DIR);
  const auto nx2b = block_size.nx(X2DIR);
  const auto nx3b = block_size.nx(X3DIR);
  // all local blocks sizes (based on logical locations)
  const auto nx1l = local_nlocs.at(0) * nx1b;
  const auto nx2l = local_nlocs.at(1) * nx2b;
  const auto nx3l = local_nlocs.at(2) * nx3b;
  const int gis = local_loc_min.at(0) * nx1b;
  const int gjs = local_loc_min.at(1) * nx2b;
  const int gks = local_loc_min.at(2) * nx3b;
  // fft() interface below requires box3d's of int (to we need to cast down)
  const heffte::box3d<> inbox({gis, gjs, gks}, {static_cast<int>(gis + nx1l - 1),
                                                static_cast<int>(gjs + nx2l - 1),
                                                static_cast<int>(gks + nx3l - 1)});

  // but let heffte determine the best complex decomposition
  std::array<int, 3> proc_grid =
      heffte::proc_setup_min_surface(complex_indexes, parthenon::Globals::nranks);
  std::vector<heffte::box3d<>> complex_boxes =
      heffte::split_world(complex_indexes, proc_grid);
  heffte::box3d<> const outbox = complex_boxes[parthenon::Globals::my_rank];

  // define the heffte class and the input and output geometry
  heffte::fft3d_r2c<backend_tag> fft(inbox, outbox, r2c_direction, MPI_COMM_WORLD);

  // TODO(pgrete) Eventually make these persistent
  int n_comp = 3;
  const auto fft_size_inbox = fft.size_inbox();
  parthenon::ParArray1D<Real> input("fft input", n_comp * fft_size_inbox);
  parthenon::ParArray1D<std::complex<Real>> output("fft output",
                                                   n_comp * fft.size_outbox());
  parthenon::ParArray1D<std::complex<Real>> workspace("fft workspace",
                                                      fft.size_workspace());
  PARTHENON_REQUIRE_THROWS(pm->DefaultNumPartitions() == 1,
                           "Only pack_size=-1 currently supported for heffte.")
  // for (int spec_type = 0; spec_type < 3; spec_type++) {
  par_for(
      "Init FFT fields", 0, pm->GetNumMeshBlocksThisRank() - 1, kb.s, kb.e, jb.s, jb.e,
      ib.s, ib.e, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto kk = k - kb.s + loc_view(b, 2) * nx3b;
        const auto jj = j - jb.s + loc_view(b, 1) * nx2b;
        const auto ii = i - ib.s + loc_view(b, 0) * nx1b;
        const std::int64_t idx = (kk * nx2l + jj) * nx1l + ii;
        if (spec_type == 0) {
          const auto rho = cons(b, 0, k, j, i);
          input(idx) = cons(b, 1, k, j, i) / rho;
          input(idx + fft_size_inbox) = cons(b, 2, k, j, i) / rho;
          input(idx + 2 * fft_size_inbox) = cons(b, 3, k, j, i) / rho;
        } else if (spec_type == 1) {
          const auto sqrtrho_inv = 1.0 / Kokkos::sqrt(cons(b, 0, k, j, i));
          input(idx) = sqrtrho_inv * cons(b, 1, k, j, i);
          input(idx + fft_size_inbox) = sqrtrho_inv * cons(b, 2, k, j, i);
          input(idx + 2 * fft_size_inbox) = sqrtrho_inv * cons(b, 3, k, j, i);
        } else if (spec_type == 2) {
          input(idx) = cons(b, 5, k, j, i);
          input(idx + fft_size_inbox) = cons(b, 6, k, j, i);
          input(idx + 2 * fft_size_inbox) = cons(b, 7, k, j, i);
        } else {
          PARTHENON_FAIL("Unknown spec type");
        }
      });

  // Not useing a batched transform here to keep the workspace small.
  for (int i = 0; i < 3; i++) {
    fft.forward(input.data() + i * fft.size_inbox(),
                output.data() + i * fft.size_outbox(), workspace.data(),
                heffte::scale::full);
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

  ib.s = outbox.low[0];
  ib.e = outbox.high[0];
  jb.s = outbox.low[1];
  jb.e = outbox.high[1];
  kb.s = outbox.low[2];
  kb.e = outbox.high[2];
  const auto fft_size_outbox = fft.size_outbox();
  parthenon::par_for(
      "CalcSpec", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        auto k_z = k <= Nz / 2 ? k : -Nz + k;
        auto k_y = j <= Ny / 2 ? j : -Ny + j;
        auto k_x = i; // because we're using r2c transforms

        // for simple binning/indexing
        auto k_mag = std::sqrt(SQR(k_x) + SQR(k_y) + SQR(k_z));
        auto k_mag_int = static_cast<int>(std::floor(k_mag));

        const auto outidx =
            ((k - kb.s) * (jb.e - jb.s + 1) + (j - jb.s)) * (ib.e - ib.s + 1) + i - ib.s;

        auto val = SQR(output[outidx].real()) + SQR(output[outidx].imag()) +
                   SQR(output[outidx + fft_size_outbox].real()) +
                   SQR(output[outidx + fft_size_outbox].imag()) +
                   SQR(output[outidx + 2 * fft_size_outbox].real()) +
                   SQR(output[outidx + 2 * fft_size_outbox].imag());

        // account for Hermitian symmetry of r2c transform
        const auto fac = ((k_x > 0) && (2 * k_x != Nx)) ? 2.0 : 1.0;

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

  // Write spectrum to ordinary text file
  if (parthenon::Globals::my_rank == 0) {
    
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

    // Decide prefix based on spec_type (optional)
    std::string spec_prefix;
    if (spec_type == 0) spec_prefix = "u";
    else if (spec_type == 1) spec_prefix = "rhoU";
    else if (spec_type == 2) spec_prefix = "B";
    else PARTHENON_FAIL("Unknown spec_type");

    // Write each bin's results to the file: en_sum, k_sum, count_sum
    fout << "# Bin    En_sum    K_sum    Count\n";
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