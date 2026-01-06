#include "FFTManager.hpp"
#include "heffte.h"
#include "mesh/mesh.hpp"

namespace parthenon {

FFTManager::FFTManager(Mesh *mesh) : mesh_(mesh) {}

void FFTManager::Initialize() {
  // Create FFT plan to be used by Forward() and Backward()
  if (initialized_) return;

  auto UniformGridHelper = mesh_->GetUniformGridHelper();

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

  auto &mesh_start_idx = UniformGridHelper->mesh_start_idx;
  auto &mesh_end_idx = UniformGridHelper->mesh_end_idx;

  const heffte::box3d<> real_space_box({mesh_start_idx[0], mesh_start_idx[1], mesh_start_idx[2]}, {static_cast<int>(mesh_end_idx[0]),
                                                static_cast<int>(mesh_end_idx[1]),
                                                static_cast<int>(mesh_end_idx[2])});

  // for the fourier space box, we let heffte decide the best decomposition: 
  std::array<int, 3> proc_grid = heffte::proc_setup_min_surface(complex_indexes, parthenon::Globals::nranks);
  std::vector<heffte::box3d<>> complex_boxes = heffte::split_world(complex_indexes, proc_grid);
  heffte::box3d<> const fourier_space_box = complex_boxes[parthenon::Globals::my_rank];

  fft_plan_ = std::make_unique<heffte::fft3d_r2c<backend_tag>>(real_space_box, fourier_space_box, r2c_direction, MPI_COMM_WORLD);

  initialized_ = true;
}

auto FFTManager::Forward(int field) {
    Initialize();
    // field: 0 = rho, 1 = vx, 2 = vy, 3 = vz, 4 = energy, 5 = Bx, 6 = By, 7 = Bz
    // Create input array
    std::vector<double> input(fft_plan_->size_inbox());
    // Fill input array from meshblocks
    // TODO(pgrete) implement
    PARTHENON_FAIL("Not implemented yet");
    return fft_plan_->forward(input, heffte::scale::full);
}

} // namespace parthenon


