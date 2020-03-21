//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001
// for Los Alamos National Laboratory (LANL), which is operated by Triad
// National Security, LLC for the U.S. Department of Energy/National Nuclear
// Security Administration. All rights in the program are reserved by Triad
// National Security, LLC, and the U.S. Department of Energy/National Nuclear
// Security Administration. The Government is granted for itself and others
// acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license
// in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do
// so.
//========================================================================================

#include "kokkos_abstraction.hpp"

#include <iostream>
#include <random>
#include <vector>

#include <catch2/catch.hpp>

using parthenon::DevSpace;
using parthenon::ParArray1D;
using parthenon::ParArray2D;
using parthenon::ParArray3D;
using parthenon::ParArray4D;
using Real = double;

template <class T> bool test_wrapper_1d(T loop_pattern, DevSpace exec_space) {
  // https://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution
  std::random_device
      rd; // Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<Real> dis(-1.0, 1.0);

  const int N = 32;
  ParArray1D<Real> arr_dev("device", N);
  auto arr_host_orig = Kokkos::create_mirror(arr_dev);
  auto arr_host_mod = Kokkos::create_mirror(arr_dev);

  // initialize random data on the host not using any wrapper
  for (int i = 0; i < N; i++)
    arr_host_orig(i) = dis(gen);

  // Copy host array content to device
  Kokkos::deep_copy(arr_dev, arr_host_orig);

  // increment data on the device using prescribed wrapper
  parthenon::par_for(
      loop_pattern, "unit test 1D", exec_space, 0, N - 1,
      KOKKOS_LAMBDA(const int i) {
        arr_dev(i) += 1.0;
        });

  // Copy array back from device to host
  Kokkos::deep_copy(arr_host_mod, arr_dev);

  bool all_same = true;

  // compare data on the host
  for (int i = 0; i < N; i++)
    if (arr_host_orig(i) + 1.0 != arr_host_mod(i)) {
      all_same = false;
    }

  return all_same;
}

template <class T> bool test_wrapper_2d(T loop_pattern, DevSpace exec_space) {
  // https://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution
  std::random_device
      rd; // Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<Real> dis(-1.0, 1.0);

  const int N = 32;
  ParArray2D<Real> arr_dev("device", N, N);
  auto arr_host_orig = Kokkos::create_mirror(arr_dev);
  auto arr_host_mod = Kokkos::create_mirror(arr_dev);

  // initialize random data on the host not using any wrapper
  for (int j = 0; j < N; j++)
    for (int i = 0; i < N; i++)
      arr_host_orig(j, i) = dis(gen);

  // Copy host array content to device
  Kokkos::deep_copy(arr_dev, arr_host_orig);

  // increment data on the device using prescribed wrapper
  parthenon::par_for(
      loop_pattern, "unit test 2D", exec_space, 0, N - 1, 0, N - 1,
      KOKKOS_LAMBDA(const int j, const int i) {
        arr_dev(j, i) += 1.0;
        });

  // Copy array back from device to host
  Kokkos::deep_copy(arr_host_mod, arr_dev);

  bool all_same = true;

  // compare data on the host
  for (int j = 0; j < N; j++)
    for (int i = 0; i < N; i++)
      if (arr_host_orig(j, i) + 1.0 != arr_host_mod(j, i)) {
        all_same = false;
      }

  return all_same;
}

template <class T> bool test_wrapper_3d(T loop_pattern, DevSpace exec_space) {
  // https://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution
  std::random_device
      rd; // Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<Real> dis(-1.0, 1.0);

  const int N = 32;
  ParArray3D<Real> arr_dev("device", N, N, N);
  auto arr_host_orig = Kokkos::create_mirror(arr_dev);
  auto arr_host_mod = Kokkos::create_mirror(arr_dev);

  // initialize random data on the host not using any wrapper
  for (int k = 0; k < N; k++)
    for (int j = 0; j < N; j++)
      for (int i = 0; i < N; i++)
        arr_host_orig(k, j, i) = dis(gen);

  // Copy host array content to device
  Kokkos::deep_copy(arr_dev, arr_host_orig);

  // increment data on the device using prescribed wrapper
  parthenon::par_for(
      loop_pattern, "unit test 3D", exec_space, 0, N - 1, 0, N - 1, 0, N - 1,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        arr_dev(k, j, i) += 1.0;
      });

  // Copy array back from device to host
  Kokkos::deep_copy(arr_host_mod, arr_dev);

  bool all_same = true;

  // compare data on the host
  for (int k = 0; k < N; k++)
    for (int j = 0; j < N; j++)
      for (int i = 0; i < N; i++)
        if (arr_host_orig(k, j, i) + 1.0 != arr_host_mod(k, j, i)) {
          all_same = false;
        }

  return all_same;
}

template <class T> bool test_wrapper_4d(T loop_pattern, DevSpace exec_space) {
  // https://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution
  std::random_device
      rd; // Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<Real> dis(-1.0, 1.0);

  const int N = 32;
  ParArray4D<Real> arr_dev("device", N, N, N, N);
  auto arr_host_orig = Kokkos::create_mirror(arr_dev);
  auto arr_host_mod = Kokkos::create_mirror(arr_dev);

  // initialize random data on the host not using any wrapper
  for (int n = 0; n < N; n++)
    for (int k = 0; k < N; k++)
      for (int j = 0; j < N; j++)
        for (int i = 0; i < N; i++)
          arr_host_orig(n, k, j, i) = dis(gen);

  // Copy host array content to device
  Kokkos::deep_copy(arr_dev, arr_host_orig);

  // increment data on the device using prescribed wrapper
  parthenon::par_for(
      loop_pattern, "unit test 4D", exec_space, 0, N - 1, 0, N - 1, 0, N - 1, 0,
      N - 1, KOKKOS_LAMBDA(const int n, const int k, const int j, const int i) {
        arr_dev(n, k, j, i) += 1.0;
      });

  // Copy array back from device to host
  Kokkos::deep_copy(arr_host_mod, arr_dev);

  bool all_same = true;

  // compare data on the host
  for (int n = 0; n < N; n++)
    for (int k = 0; k < N; k++)
      for (int j = 0; j < N; j++)
        for (int i = 0; i < N; i++)
          if (arr_host_orig(n, k, j, i) + 1.0 != arr_host_mod(n, k, j, i)) {
            all_same = false;
          }

  return all_same;
}

TEST_CASE("par_for loops", "[wrapper]") {
  auto default_exec_space = DevSpace();

  SECTION("1D loops") {
    REQUIRE(test_wrapper_1d(parthenon::loop_pattern_mdrange_tag,
                            default_exec_space) == true);
  }

  SECTION("2D loops") {
    REQUIRE(test_wrapper_2d(parthenon::loop_pattern_mdrange_tag,
                            default_exec_space) == true);
  }

  SECTION("3D loops") {
    REQUIRE(test_wrapper_3d(parthenon::loop_pattern_range_tag,
                            default_exec_space) == true);

    REQUIRE(test_wrapper_3d(parthenon::loop_pattern_mdrange_tag,
                            default_exec_space) == true);

    REQUIRE(test_wrapper_3d(parthenon::loop_pattern_tpttrtvr_tag,
                            default_exec_space) == true);

    REQUIRE(test_wrapper_3d(parthenon::loop_pattern_tpttr_tag,
                            default_exec_space) == true);

#ifndef KOKKOS_ENABLE_CUDA
    REQUIRE(test_wrapper_3d(parthenon::loop_pattern_tptvr_tag,
                            default_exec_space) == true);

    REQUIRE(test_wrapper_3d(parthenon::loop_pattern_simdfor_tag,
                            default_exec_space) == true);
#endif
  }

  SECTION("4D loops") {
    REQUIRE(test_wrapper_4d(parthenon::loop_pattern_range_tag,
                            default_exec_space) == true);

    REQUIRE(test_wrapper_4d(parthenon::loop_pattern_mdrange_tag,
                            default_exec_space) == true);

    REQUIRE(test_wrapper_4d(parthenon::loop_pattern_tpttrtvr_tag,
                            default_exec_space) == true);

    REQUIRE(test_wrapper_4d(parthenon::loop_pattern_tpttr_tag,
                            default_exec_space) == true);

#ifndef KOKKOS_ENABLE_CUDA
    REQUIRE(test_wrapper_4d(parthenon::loop_pattern_tptvr_tag,
                            default_exec_space) == true);

    REQUIRE(test_wrapper_4d(parthenon::loop_pattern_simdfor_tag,
                            default_exec_space) == true);
#endif
  }
}

struct BufferPack {
  int nghost;
  int ncells; // number of cells in the linear dimension - very simplistic
              // approach
  ParArray4D<Real> arr_in;
  // buffer in six direction, i plus, i minus, ...
  ParArray1D<Real> buf_ip, buf_im, buf_jp, buf_jm, buf_kp, buf_km;
  BufferPack(const int nghost_, const int ncells_,
             const ParArray4D<Real> arr_in_, ParArray1D<Real> buf_ip_,
             ParArray1D<Real> buf_im_, ParArray1D<Real> buf_jp_,
             ParArray1D<Real> buf_jm_, ParArray1D<Real> buf_kp_,
             ParArray1D<Real> buf_km_)
      : nghost(nghost_), ncells(ncells_), arr_in(arr_in_),
        buf_ip(buf_ip_), buf_im(buf_im_), buf_jp(buf_jp_), buf_jm(buf_jm_),
        buf_kp(buf_kp_), buf_km(buf_km_) {}
  KOKKOS_INLINE_FUNCTION
  void operator()(const int n, const int dir) const {
    int idx = 0;
    const int offset = n * (nghost * (ncells - 2 * nghost) * (ncells - 2 * nghost));

    if (dir == 0) {
      for (auto k = nghost; k < ncells - nghost; k++)
        for (auto j = nghost; j < ncells - nghost; j++)
          for (auto i = nghost; i < 2 * nghost; i++)
            buf_im(offset + idx++) = arr_in(n, k, j, i);
    } else if (dir == 1) {
      for (auto k = nghost; k < ncells - nghost; k++)
        for (auto j = nghost; j < ncells - nghost; j++)
          for (auto i = ncells - nghost; i < ncells; i++)
            buf_ip(offset + idx++) = arr_in(n, k, j, i);
    } else if (dir == 2) {
      for (auto k = nghost; k < ncells - nghost; k++)
        for (auto j = nghost; j < 2 * nghost; j++)
          for (auto i = nghost; i < ncells - nghost; i++)
            buf_jm(offset + idx++) = arr_in(n, k, j, i);
    } else if (dir == 3) {
      for (auto k = nghost; k < ncells - nghost; k++)
        for (auto j = ncells - nghost; j < ncells; j++)
          for (auto i = nghost; i < ncells - nghost; i++)
            buf_jp(offset + idx++) = arr_in(n, k, j, i);
    } else if (dir == 4) {
      for (auto k = nghost; k < 2 * nghost; k++)
        for (auto j = nghost; j < ncells - nghost; j++)
          for (auto i = nghost; i < ncells - nghost; i++)
            buf_km(offset + idx++) = arr_in(n, k, j, i);
    } else if (dir == 5) {
      for (auto k = ncells - nghost; k < ncells; k++)
        for (auto j = nghost; j < ncells - nghost; j++)
          for (auto i = nghost; i < ncells - nghost; i++)
            buf_kp(offset + idx++) = arr_in(n, k, j, i);
    }
  }
};

TEST_CASE("Overlapping SpaceInstances", "[wrapper]") {
  auto default_exec_space = DevSpace();

  const int N = 32;       // ~meshblock size
  const int M = 5;        // ~nhydro
  const int nstreams = 8; // number of streams
  const int nghost = 2;   // number of ghost zones
  const int nbuffers =
      6; // number of buffers, here up, down, left, right, back, front
  const int buf_size = M * nghost * (N - 2 * nghost) * (N - 2 * nghost);
  std::vector<BufferPack> functs;
  std::vector<DevSpace> exec_spaces;

  for (auto n = 0; n < nstreams; n++) {
    functs.push_back(BufferPack(
        nghost, N, ParArray4D<Real>("SpaceInstance in", M, N, N, N),
        ParArray1D<Real>("buf_ip", buf_size),
        ParArray1D<Real>("buf_im", buf_size),
        ParArray1D<Real>("buf_jp", buf_size),
        ParArray1D<Real>("buf_jm", buf_size),
        ParArray1D<Real>("buf_kp", buf_size),
        ParArray1D<Real>("buf_kp", buf_size)));
    exec_spaces.push_back(parthenon::SpaceInstance<DevSpace>::create());
  }

  // warmup
  for (auto it = 0; it < 10; it++) {
    for (auto n = 0; n < nstreams; n++) {
      parthenon::par_for(parthenon::loop_pattern_mdrange_tag, "space",
                         exec_spaces[n], 0, M-1, 0, nbuffers-1, functs[n]);
    }
  }
  Kokkos::fence();

  Kokkos::Timer timer;

  // meausre time using two execution space simultaneously
  // race condition in access to arr_dev doesn't matter for this test
  for (auto n = 0; n < nstreams; n++) {
    parthenon::par_for(parthenon::loop_pattern_mdrange_tag, "space",
                       exec_spaces[n], 0, M-1, 0, nbuffers-1, functs[n]);
  }

  Kokkos::fence();
  auto time_spaces = timer.seconds();

  timer.reset();

  // measure runtime using the default execution space
  for (auto n = 0; n < nstreams; n++) {
    parthenon::par_for(parthenon::loop_pattern_mdrange_tag, "default space",
                       default_exec_space, 0, M-1, 0, nbuffers-1, functs[n]);
  }

  default_exec_space.fence(); // making sure the kernel is done
  auto time_default = timer.seconds();

  std::cout << "time default: " << time_default << std::endl;
  std::cout << "time spaces: " << time_spaces << std::endl;

  // make sure this test is reasonable IIF streams actually overlap, which is
  // not the case for the OpenMP backend at this point
  if (parthenon::SpaceInstance<DevSpace>::overlap()) {
    // make sure the per kernel runtime didn't increase by more than a factor of 2
    REQUIRE(time_default > (static_cast<Real>(nstreams) / 2.0 * time_spaces));
  }
}
