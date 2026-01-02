#include "FFTManager.hpp"
#include "heffte.h"
#include "mesh/mesh.hpp"

namespace parthenon {

FFTManager::FFTManager(Mesh *mesh) : mesh_(mesh) {}

void FFTManager::Initialize() {
  // Create FFT plan to be used by Forward() and Backward()

  if (initialized_) return;

  // Determine global box sizes
  auto mesh_size = mesh_->mesh_size;
  auto Nx = mesh_size.nx(X1DIR);
  auto Ny = mesh_size.nx(X2DIR);
  auto Nz = mesh_size.nx(X3DIR);

  std::int64_t r2c_direction = 0; // the dimension where the data will shrink
  // construct global input/output boxes: 
  heffte::box3d<> real_indexes({0, 0, 0}, {Nx - 1, Ny - 1, Nz - 1});
  heffte::box3d<> complex_indexes({0, 0, 0}, {(Nx)/2, Ny - 1, Nz - 1});

  // check if the complex indexes have correct dimension
  assert(real_indexes.r2c(r2c_direction) == complex_indexes);

  // Need to store this info in a way this can be used on device later
  parthenon::ParArray2D<std::int64_t> loc_view("logical location of local blocks",
                                               mesh_->GetNumMeshBlocksThisRank(), 3);
  auto loc_view_h = loc_view.GetHostMirror();

  // Set rank local min and max logical locations.
  // Also check if all blocks are on the same level (we use this check instead of
  // checking for refinement=none because AMR could have been used to dynamically refine
  // a simulation. We just need to ensure that all blocks are on the same level to
  // create an effective uniform grid.)
  const auto level =
      mesh_->Forest().GetLegacyTreeLocation(mesh_->block_list[0]->loc).level();

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
  
  for (int b = 0; b < mesh_->GetNumMeshBlocksThisRank(); b++) {
    auto pmb = mesh_->block_list[b];
    const auto loc = mesh_->Forest().GetLegacyTreeLocation(pmb->loc);
    for (int i = 0; i <= 2; i++) {
      local_loc_min.at(i) = std::min(loc.l(i), local_loc_min.at(i));
      local_loc_max.at(i) = std::max(loc.l(i), local_loc_max.at(i));
      loc_view_h(b, i) = loc.l(i);
    }
    PARTHENON_REQUIRE_THROWS(loc.level() == level,
                             "Not all blocks are on the same level.");
  }

  // convert global logical locations to rank-local logical locs
  for (int b = 0; b < mesh_->GetNumMeshBlocksThisRank(); b++) {
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
  PARTHENON_REQUIRE_THROWS(loc_max_vol == mesh_->GetNumMeshBlocksThisRank(),
                           "Block coverage on rank cannot be matched to a contiguous "
                           "array, which is required for FFTs. Try a different amount of "
                           "ranks (one block per rank will always work).");

  // TODO(pgrete) not nice, make nicer
  //#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  //using backend_tag = heffte::backend::default_backend<heffte::tag::gpu>::type;
  //PARTHENON_REQUIRE_THROWS(heffte::gpu::device_count() == 1,
  //                         "To make this work, we need to ensure that Kokkos and heffte "
  //                         "use the same GPUs. So hard fail for now.");
  //#else
  //using backend_tag = heffte::backend::default_backend<heffte::tag::cpu>::type;
  //#endif
  
  // for now, always use CPU backend. Need to change input/output types when using GPU backend. 
  // Since this is only executed once at the beginning of the simulation, this is acceptable for now.
  using backend_tag = heffte::backend::default_backend<heffte::tag::cpu>::type;

  const auto block_size = mesh_->GetDefaultBlockSize();
  // block sizes
  const int nx1b = block_size.nx(parthenon::X1DIR);
  const int nx2b = block_size.nx(parthenon::X2DIR);
  const int nx3b = block_size.nx(parthenon::X3DIR);
  // all local blocks sizes (based on logical locations)
  const std::int64_t nx1l = local_nlocs.at(0) * nx1b;
  const std::int64_t nx2l = local_nlocs.at(1) * nx2b;
  const std::int64_t nx3l = local_nlocs.at(2) * nx3b;
  const int gis = local_loc_min.at(0) * nx1b;
  const int gjs = local_loc_min.at(1) * nx2b;
  const int gks = local_loc_min.at(2) * nx3b;
  // fft() interface below requires box3d's of int (to we need to cast down)
  const heffte::box3d<> real_space_box({gis, gjs, gks}, {static_cast<int>(gis + nx1l - 1),
                                                static_cast<int>(gjs + nx2l - 1),
                                                static_cast<int>(gks + nx3l - 1)});

  // for the fourier space box, we let heffte decide the best decomposition: 
  std::array<int, 3> proc_grid = heffte::proc_setup_min_surface(complex_indexes, parthenon::Globals::nranks);
  std::vector<heffte::box3d<>> complex_boxes = heffte::split_world(complex_indexes, proc_grid);
  heffte::box3d<> const fourier_space_box = complex_boxes[parthenon::Globals::my_rank];

  fft_plan_ = std::make_unique<heffte::fft3d_r2c<backend_tag>>(real_space_box, fourier_space_box, r2c_direction, MPI_COMM_WORLD);

  initialized_ = true;
}

} // namespace parthenon


