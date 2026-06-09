//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2026 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================

#include "fft_manager.hpp"
#include "heffte.h"
#include "mesh/mesh.hpp"

namespace parthenon {

struct FFTManager::Impl {
// @pgrete: Can the backend selection be made nicer?
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  using BackendTag = heffte::backend::default_backend<heffte::tag::gpu>::type;
#else
  using BackendTag = heffte::backend::default_backend<heffte::tag::cpu>::type;
#endif

  heffte::fft3d_r2c<BackendTag> fft_plan;
  ParArray1D<std::complex<double>> workspace_;

  Impl(const heffte::box3d<> &real_space_box, const heffte::box3d<> &fourier_space_box,
       int r2c_direction, MPI_Comm comm)
      : fft_plan(real_space_box, fourier_space_box, r2c_direction, comm),
        workspace_("fft workspace", fft_plan.size_workspace()) {}
};

FFTManager::FFTManager(Mesh *mesh) : mesh_(mesh) {
  auto UniformGridHelper = mesh_->GetUniformGridHelper();

  auto mesh_size = mesh_->mesh_size;
  nx_ = mesh_size.nx(X1DIR);
  ny_ = mesh_size.nx(X2DIR);
  nz_ = mesh_size.nx(X3DIR);

  std::int64_t r2c_direction = 0;

  heffte::box3d<> real_indexes({0, 0, 0}, {nx_ - 1, ny_ - 1, nz_ - 1});
  heffte::box3d<> complex_indexes({0, 0, 0}, {nx_ / 2, ny_ - 1, nz_ - 1});

  assert(real_indexes.r2c(r2c_direction) == complex_indexes);

  auto &mesh_start_idx = UniformGridHelper->local_mesh_box.low;
  auto &mesh_end_idx = UniformGridHelper->local_mesh_box.high;

  const heffte::box3d<> real_space_box(
      {mesh_start_idx[0], mesh_start_idx[1], mesh_start_idx[2]},
      {static_cast<int>(mesh_end_idx[0]), static_cast<int>(mesh_end_idx[1]),
       static_cast<int>(mesh_end_idx[2])});

  std::array<int, 3> proc_grid =
      heffte::proc_setup_min_surface(complex_indexes, parthenon::Globals::nranks);

  std::vector<heffte::box3d<>> complex_boxes =
      heffte::split_world(complex_indexes, proc_grid);

  heffte::box3d<> const fourier_space_box = complex_boxes[parthenon::Globals::my_rank];

  impl_ = std::make_unique<Impl>(real_space_box, fourier_space_box, r2c_direction,
                                 MPI_COMM_WORLD);
} // FFTManager::FFTManager

// -----------------------------
// Forward / Backward
// -----------------------------
void FFTManager::Forward(const double *input, std::complex<double> *output) {
  impl_->fft_plan.forward(
      input, output, impl_->workspace_.data(),
      heffte::scale::full); // 1/N^3 normalization for forward transform
}

void FFTManager::Backward(const std::complex<double> *input, double *output) {
  impl_->fft_plan.backward(
      input, output,
      heffte::scale::none); // no normalization for backward transform, so that forward
                            // followed by backward gives back the original field
}

// -----------------------------
// Boxes and sizes
// -----------------------------
parthenon::Box3D FFTManager::fourier_space_box() const {
  parthenon::Box3D box;
  const auto &b = impl_->fft_plan.outbox();
  for (int i = 0; i < 3; i++) {
    box.low[i] = b.low[i];
    box.high[i] = b.high[i];
    box.size[i] = b.high[i] - b.low[i] + 1;
  }
  return box;
}

parthenon::Box3D FFTManager::real_space_box() const {
  parthenon::Box3D box;
  const auto &b = impl_->fft_plan.inbox();
  for (int i = 0; i < 3; i++) {
    box.low[i] = b.low[i];
    box.high[i] = b.high[i];
    box.size[i] = b.high[i] - b.low[i] + 1;
  }
  return box;
}

std::size_t FFTManager::size_fourier_space_box() const {
  const auto box = fourier_space_box();
  return static_cast<std::size_t>(box.size[0]) * static_cast<std::size_t>(box.size[1]) *
         static_cast<std::size_t>(box.size[2]);
}

std::size_t FFTManager::size_real_space_box() const {
  const auto box = real_space_box();
  return static_cast<std::size_t>(box.size[0]) * static_cast<std::size_t>(box.size[1]) *
         static_cast<std::size_t>(box.size[2]);
}

FFTManager::~FFTManager() = default;

} // namespace parthenon
