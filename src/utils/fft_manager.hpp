//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2026 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================

#ifndef UTILS_FFT_MANAGER_HPP_
#define UTILS_FFT_MANAGER_HPP_

#include "parthenon_arrays.hpp"
#include "mesh/uniform_grid_helper.hpp"
#include <complex>
#include <memory>

namespace parthenon {

class Mesh;

class FFTManager {
  friend class Mesh;

 public:
  explicit FFTManager(Mesh *mesh);
  ~FFTManager();

  void Forward(const double *input, std::complex<double> *output);

  void Backward(const std::complex<double> *input, double *output);

  // -----------------------------
  // Box info
  // -----------------------------

  parthenon::Box3D fourier_space_box() const;
  parthenon::Box3D real_space_box() const;

  std::size_t size_fourier_space_box() const; // total number of points
  std::size_t size_real_space_box() const;

  // -----------------------------
  // Device-copyable kernel helper
  // -----------------------------
  struct KernelHelper {
    parthenon::Box3D fourier_box;
    parthenon::Box3D real_box;
    int Nx, Ny, Nz;

    // Flat index into the local Fourier-space array
    KOKKOS_INLINE_FUNCTION
    std::int64_t FourierFlatIndex(const int k, const int j, const int i) const {
      return ((std::int64_t)(k - fourier_box.low[2]) * fourier_box.size[1] +
              (j - fourier_box.low[1])) *
                 fourier_box.size[0] +
             i - fourier_box.low[0];
    }

    // Flat index into the local real-space array
    KOKKOS_INLINE_FUNCTION
    std::int64_t RealFlatIndex(const int k, const int j, const int i) const {
      return ((std::int64_t)(k - real_box.low[2]) * real_box.size[1] +
              (j - real_box.low[1])) *
                 real_box.size[0] +
             i - real_box.low[0];
    }

    // Integer wavevector components (handles negative frequencies)
    // For r2c transforms, kx >= 0 always
    KOKKOS_INLINE_FUNCTION
    std::array<int, 3> Wavevector(const int k, const int j, const int i) const {
      return {i, j <= Ny / 2 ? j : j - Ny, k <= Nz / 2 ? k : k - Nz};
    }
  };

  // Returns a device-copyable helper for use inside Kokkos kernels.
  // Capture by value in KOKKOS_LAMBDA:
  //   auto helper = fftManager->GetKernelHelper();
  //   par_for(..., KOKKOS_LAMBDA(...) { helper.FourierFlatIndex(...); });
  KernelHelper GetKernelHelper() const {
    return {fourier_space_box(), real_space_box(), Nx_, Ny_, Nz_};
  }

 private:
  struct Impl;                 // opaque implementation
  std::unique_ptr<Impl> impl_; // owns backend-specific data

  Mesh *mesh_;

  // Global mesh dimensions, stored during Initialize()
  int Nx_ = 0, Ny_ = 0, Nz_ = 0;
};

} // namespace parthenon

#endif // UTILS_FFT_MANAGER_HPP_
