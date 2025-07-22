//========================================================================================
// (C) (or copyright) 2020-2021. Triad National Security, LLC. All rights reserved.
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

// Standard Includes
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include FS_HEADER
namespace fs = FS_NAMESPACE;

// heffte headers
#include "globals.hpp"
#include "heffte.h"

// Parthenon Includes
#include <coordinates/coordinates.hpp>
#include <kokkos_abstraction.hpp>
#include <mesh/domain.hpp>
#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>

// Local Includes
#include "calc_spec.hpp"

using namespace parthenon::package::prelude;
using parthenon::IndexShape;

namespace calculate_pi {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto package = std::make_shared<StateDescriptor>("calculate_pi");
  Params &params = package->AllParams();

  auto out_num = pin->GetInteger("CalcSpec", "output_number");
  package->AddParam("output_number", out_num);
  std::string field_name("prim");
  Metadata m({Metadata::Cell, Metadata::Derived, Metadata::OneCopy},
             std::vector<int>({8}));
  package->AddField(field_name, m);

  return package;
}

void Log(const std::string &msg) {
  std::cerr << "[" << parthenon::Globals::my_rank << "]: " << msg << "\n";
}

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {

  // pmb->gid
  // Log("Reading data for block: " + std::to_string(pmb->gid));
  auto pkg = pmb->packages.Get("calculate_pi");
  const auto out_num = pkg->Param<int>("output_number");
  // TODO(pgrete) currentl assumes dircect block to rank match
  // TODO(pgrete) use C++20 std::format eventually
  char buff[100];
  auto data_path = pin->GetString("CalcSpec", "data_path");
  sprintf(buff, "%s/bin/rank_%08d/Turb.full_mhd_w_bcc.%05d.bin", data_path.c_str(),
          pmb->gid, out_num);
  std::string filename = buff;
  if (fs::exists(filename)) {
    // Log("Loading " + filename);
  } else {
    Log("Cannot find " + filename);
    PARTHENON_FAIL("Reading data failed.");
  }
  // Get size of file to know how much memory to allocate
  std::uintmax_t filesize = fs::file_size(filename);

  // Read file
  std::ifstream data_stream(filename, std::ios::binary);

  std::string line;
  // Athena binary output version=1.1
  std::getline(data_stream, line);
  // size of preheader=5
  std::getline(data_stream, line);
  // time=0
  std::getline(data_stream, line);
  // cycle=0
  std::getline(data_stream, line);
  // size of location=8
  std::getline(data_stream, line);
  // size of variable=4
  std::getline(data_stream, line);
  // number of variables=8
  std::getline(data_stream, line);
  // variables:  dens  velx  vely  velz  eint  bcc1  bcc2  bcc3
  std::getline(data_stream, line);
  // header offset=6281
  std::getline(data_stream, line);
  // #------------------------- PAR_DUMP -------------------------

  std::size_t pos = line.find("=") + 1;
  std::size_t header_offset = std::stoi(line.substr(pos));

  data_stream.seekg(header_offset, std::ios_base::cur);
  // TODO(pgrete): sizes and types should be adjusted to input
  int32_t mb_idx[6];
  int32_t loc[4];
  double geo[6];
  // mb_index.append(np.frombuffer(fp.read(24), dtype=np.int32).astype(np.int64) - nghost)
  // mb_logical.append(np.frombuffer(fp.read(16), dtype=np.int32))
  // mb_geometry.append(np.frombuffer(fp.read(6 * locsizebytes), dtype=dtype_loc))
  // TODO(pgrete) adjust local mb indices for number of ghosts
  data_stream.read(reinterpret_cast<char *>(&mb_idx), 24);
  data_stream.read(reinterpret_cast<char *>(&loc), 16);
  data_stream.read(reinterpret_cast<char *>(&geo), 6 * 8);

  std::stringstream msg;
  msg << "mb_idx: " << mb_idx[0] << "-" << mb_idx[1] << " " << mb_idx[2] << "-"
      << mb_idx[3] << " " << mb_idx[4] << "-" << mb_idx[5]
      << " Logical locations: " << loc[0] << " " << loc[1] << " " << loc[2] << " "
      << loc[3] << " geometry: " << geo[0] << "-" << geo[1] << " " << geo[2] << "-"
      << geo[3] << " " << geo[4] << "-" << geo[5] << "\n";
  // Log(msg.str());

  auto &cellbounds = pmb->cellbounds;
  auto ib = cellbounds.GetBoundsI(IndexDomain::interior);
  auto jb = cellbounds.GetBoundsJ(IndexDomain::interior);
  auto kb = cellbounds.GetBoundsK(IndexDomain::interior);

  // Sanity checks
  auto block_size_disk = (mb_idx[1] - mb_idx[0] + 1) * (mb_idx[3] - mb_idx[2] + 1) *
                         (mb_idx[5] - mb_idx[4] + 1);
  auto block_size_mesh = (ib.e - ib.s + 1) * (jb.e - jb.s + 1) * (kb.e - kb.s + 1);
  PARTHENON_REQUIRE_THROWS(block_size_disk == block_size_mesh, "Mismatch is block size");

  const auto loc_mesh = pmb->pmy_mesh->Forest().GetLegacyTreeLocation(pmb->loc);
  for (int i = 0; i <= 2; i++) {
    PARTHENON_REQUIRE_THROWS(loc_mesh.l(i) == loc[i], "Mismatch is logical loc");
  }

  const size_t num_vars = 8;

  auto bytes_to_read = num_vars * block_size_mesh * sizeof(float);
  std::vector<float> buf(num_vars * block_size_mesh);
  data_stream.read(reinterpret_cast<char *>(buf.data()), bytes_to_read);

  PARTHENON_REQUIRE_THROWS(data_stream.gcount() == bytes_to_read,
                           "Didn't read all bytes.");

  // auto current_pos = data_stream.tellg();
  // data_stream.seekg(0, data_stream.end);
  // auto final_pos = data_stream.tellg();
  // auto data_left = final_pos - current_pos;
  // if (data_left != 0) {
  // PARTHENON_WARN("There's data left of size: " + std::to_string(data_left));
  // }
  // Close file
  data_stream.close();

  auto &data = pmb->meshblock_data.Get();
  auto &prim_dev = data->Get("prim").data;
  auto &coords = pmb->coords;
  // initializing on host
  auto prim = prim_dev.GetHostMirrorAndCopy();
  size_t idx = 0;
  size_t num_nans = 0;
  size_t num_zeros = 0;
  for (int n = 0; n < num_vars; n++) {
    for (int k = kb.s; k <= kb.e; k++) {
      for (int j = jb.s; j <= jb.e; j++) {
        for (int i = ib.s; i <= ib.e; i++) {
          prim(n, k, j, i) = buf[idx];
          PARTHENON_REQUIRE_THROWS(
              !((n == 0 || n == 4) && prim(n, k, j, i) == 0),
              "[" + std::to_string(parthenon::Globals::my_rank) +
                  "] No zeros allowed for densit and energy. Found one at " +
                  std::to_string(n) + " " + std::to_string(k) + " " + std::to_string(j) +
                  " " + std::to_string(i) + " ");

          if (prim(n, k, j, i) == 0.0) {
            num_zeros += 1;
          }
          if (std::isnan(prim(n, k, j, i))) {
            prim(n, k, j, i) = 0.0;
            num_nans += 1;
          }

          idx++;
        }
      }
    }
  }
  if (num_zeros != 0 || num_nans != 0) {
    Log("block " + std::to_string(pmb->gid) + " got " + std::to_string(num_zeros) +
        " zeros and " + std::to_string(num_nans) + " nans");
  }

  PARTHENON_REQUIRE_THROWS(idx == buf.size(),
                           "Mismatch in data being read and processed");
  // Log("idx is " + std::to_string(idx) + " and vec size is " +
  // std::to_string(buf.size()));

  // copy initialized vars to device
  prim_dev.DeepCopy(prim);
}

TaskStatus CalcSpec(std::shared_ptr<MeshData<Real>> &md, ParArrayHost<Real> areas,
                    int spec_type) {

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

  auto *pmesh = md->GetMeshPointer();
  // Need to store this info in a way this can be used on device later
  parthenon::ParArray2D<std::int64_t> loc_view("logical location of local blocks",
                                               pmesh->GetNumMeshBlocksThisRank(), 3);
  auto loc_view_h = loc_view.GetHostMirror();

  // Set rank local min and max logical locations.
  // Also check if all blocks are on the same level (we use this check instead of
  // checking for refinement=none because AMR could have been used to dynamically refine
  // a simulation. We just need to ensure that all blocks are on the same level to
  // create an effective uniform grid.)
  const auto level =
      pmesh->Forest().GetLegacyTreeLocation(pmesh->block_list[0]->loc).level();
  for (int b = 0; b < pmesh->GetNumMeshBlocksThisRank(); b++) {
    auto pmb = pmesh->block_list[b];
    const auto loc = pmesh->Forest().GetLegacyTreeLocation(pmb->loc);
    for (int i = 0; i <= 2; i++) {
      local_loc_min.at(i) = std::min(loc.l(i), local_loc_min.at(i));
      local_loc_max.at(i) = std::max(loc.l(i), local_loc_max.at(i));
      loc_view_h(b, i) = loc.l(i);
    }
    PARTHENON_REQUIRE_THROWS(loc.level() == level,
                             "Not all blocks are on the same level.");
  }

  // convert global logical locations to rank-local logical locs
  for (int b = 0; b < pmesh->GetNumMeshBlocksThisRank(); b++) {
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
  PARTHENON_REQUIRE_THROWS(loc_max_vol == pmesh->GetNumMeshBlocksThisRank(),
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

  // the dimension where the data will shrink
  int r2c_direction = 0;
  // Adjust (logical) grid size at levels other than the root level.
  // This is required for simulation with mesh refinement so that the phases calculated
  // below take the logical grid size into account. For example, the local phases at
  // level 1 should be calculated assuming a grid that is twice as large as the root
  // grid.

  PARTHENON_REQUIRE_THROWS(!pmesh->adaptive, "Ask Luke about the logic here.");
  // const auto root_level = pmesh->GetRootLevel();
  // auto gnx1 =
  // static_cast<int>(pmesh->mesh_size.nx(X1DIR) * std::pow(2, level - root_level));
  // auto gnx2 =
  // static_cast<int>(pmesh->mesh_size.nx(X2DIR) * std::pow(2, level - root_level));
  // auto gnx3 =
  // static_cast<int>(pmesh->mesh_size.nx(X3DIR) * std::pow(2, level - root_level));
  const auto gnx1 = pmesh->mesh_size.nx(X1DIR);
  const auto gnx2 = pmesh->mesh_size.nx(X2DIR);
  const auto gnx3 = pmesh->mesh_size.nx(X3DIR);

  heffte::box3d<> real_indexes({0, 0, 0}, {gnx1 - 1, gnx2 - 1, gnx3 - 1});
  heffte::box3d<> complex_indexes({0, 0, 0}, {
                                                 (gnx1 - 1) / 2 + 1,
                                                 gnx2 - 1,
                                                 gnx3 - 1,

                                             });
  // check if the complex indexes have correct dimension
  assert(real_indexes.r2c(r2c_direction) == complex_indexes);

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
  const auto block_size = pmesh->GetDefaultBlockSize();
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
  parthenon::ParArray1D<Real> inverse("fft inverse", n_comp * fft_size_inbox);
  parthenon::ParArray1D<std::complex<Real>> output("fft output",
                                                   n_comp * fft.size_outbox());
  parthenon::ParArray1D<std::complex<Real>> workspace("fft workspace",
                                                      n_comp * fft.size_workspace());
  PARTHENON_REQUIRE_THROWS(pmesh->DefaultNumPartitions() == 1,
                           "Only pack_size=-1 currently supported for heffte.")
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);
  // TODO(pgrete) check what's wrong with the variable pack (dealloc) -- especially when
  // called within the for loop
  auto prim = md->PackVariables(std::vector<std::string>{"prim"});
  // for (int spec_type = 0; spec_type < 3; spec_type++) {
  par_for(
      "Init FFT fields", 0, pmesh->GetNumMeshBlocksThisRank() - 1, kb.s, kb.e, jb.s, jb.e,
      ib.s, ib.e, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &p = prim(b);
        const auto kk = k - kb.s + loc_view(b, 2) * nx3b;
        const auto jj = j - jb.s + loc_view(b, 1) * nx2b;
        const auto ii = i - ib.s + loc_view(b, 0) * nx1b;
        const std::int64_t idx = (kk * nx2l + jj) * nx1l + ii;
        if (spec_type == 0) {
          input(idx) = p(1, k, j, i);
          input(idx + fft_size_inbox) = p(2, k, j, i);
          input(idx + 2 * fft_size_inbox) = p(3, k, j, i);
        } else if (spec_type == 1) {
          const auto sqrtrho = Kokkos::sqrt(p(0, k, j, i));
          input(idx) = sqrtrho * p(1, k, j, i);
          input(idx + fft_size_inbox) = sqrtrho * p(2, k, j, i);
          input(idx + 2 * fft_size_inbox) = sqrtrho * p(3, k, j, i);
        } else if (spec_type == 2) {
          input(idx) = p(5, k, j, i);
          input(idx + fft_size_inbox) = p(6, k, j, i);
          input(idx + 2 * fft_size_inbox) = p(7, k, j, i);
        } else {
          PARTHENON_FAIL("Unknown spec type");
        }
      });

  fft.forward(n_comp, input.data(), output.data(), workspace.data());

  const auto k_max = std::sqrt(SQR(gnx1 / 2) + SQR(gnx2 / 2) + SQR(gnx3 / 2));

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
        auto k_z = k <= gnx3 / 2 ? k : -gnx3 + k;
        auto k_y = j <= gnx2 / 2 ? j : -gnx2 + j;
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
        const auto fac = ((k_x > 0) && (2 * k_x != gnx1)) ? 2.0 : 1.0;

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

  if (parthenon::Globals::my_rank == 0) {
    auto pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("calculate_pi");
    const auto out_num = pkg->Param<int>("output_number");
    auto spectra_h = spectra.GetHostMirrorAndCopy();
    // and write data
    std::ofstream outfile;
    const std::string fname("spec_" + std::to_string(out_num) + ".csv");
    // On startup, write header
    // if (tm.ncycle == 0) {
    if (spec_type == 0) {
      outfile.open(fname, std::ofstream::out);
      outfile << "# num_bins, pos spec,...\n";
    } else {
      outfile.open(fname, std::ofstream::out | std::ofstream::app);
    }
    // outfile << "# cycle, time, num_bins, pos spec,...\n";
    // } else {
    // outfile.open(fname, std::ofstream::out | std::ofstream::app);
    // }

    // outfile << tm.ncycle << "," << tm.time << "," << num_bins;
    outfile << num_bins;

    for (int j = 0; j < 3; j++) {
      for (int i = 0; i < num_bins; i++) {
        outfile << "," << spectra_h(i, j);
      }
    }
    outfile << std::endl;

    outfile.close();
  }
  // }
  return TaskStatus::complete;
}

} // namespace calculate_pi
