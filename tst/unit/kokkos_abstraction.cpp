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

#include <iostream>
#include <random>

#include "../../src/kokkos_abstraction.hpp"
#include <catch2/catch.hpp>
#include "Kokkos_Macros.hpp"

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
      KOKKOS_LAMBDA(const int i) { arr_dev(i) += 1.0; });

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
      KOKKOS_LAMBDA(const int j, const int i) { arr_dev(j, i) += 1.0; });

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

// reused from kokoks/core/perf_test/PerfTest_ExecSpacePartitioning.cpp
// commit a0d011fb30022362c61b3bb000ae3de6906cb6a7
struct FunctorRange {
  int M, R;
  ParArray1D<Real> arr_dev;
  FunctorRange(int M_, int R_, ParArray1D<Real> arr_dev_)
      : M(M_), R(R_), arr_dev(arr_dev_) {}
  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const {
    for (int r = 0; r < R; r++)
      for (int j = 0; j < M; j++) {
        arr_dev(i) += 1.0;
      }
  }
};

TEST_CASE("Overlapping SpaceInstances", "[wrapper]") {
  auto default_exec_space = DevSpace();
  auto space1 = parthenon::SpaceInstance<DevSpace>::create();
  auto space2 = parthenon::SpaceInstance<DevSpace>::create();

  const int64_t N = 5 * 16 * 16 * 16;

  ParArray1D<Real> arr_dev("SpaceInstance test array", N);
  FunctorRange f(10000, 10, arr_dev);
  // warmup
  for (auto it = 0; it < 10; it++) {
    parthenon::par_for(parthenon::loop_pattern_mdrange_tag, "default space",
                       default_exec_space, 0, N - 1, f);
    parthenon::par_for(parthenon::loop_pattern_mdrange_tag, "space1", space1, 0,
                       N - 1, f);
    parthenon::par_for(parthenon::loop_pattern_mdrange_tag, "space2", space2, 0,
                       N - 1, f);
  }
  Kokkos::fence();

  Kokkos::Timer timer;

  // meausre time using two execution space simultaneously
  // race condition in access to arr_dev doesn't matter for this test
  parthenon::par_for(parthenon::loop_pattern_mdrange_tag, "space1", space1, 0,
                     N - 1, f);
  parthenon::par_for(parthenon::loop_pattern_mdrange_tag, "space2", space2, 0,
                     N - 1, f);
  space1.fence(); // making sure the kernels are done
  space2.fence(); // making sure the kernels are done
  auto time_spaces = timer.seconds();

  timer.reset();

  // measure runtime using the default execution space
  parthenon::par_for(parthenon::loop_pattern_mdrange_tag, "default space",
                     default_exec_space, 0, N - 1, f);
  parthenon::par_for(parthenon::loop_pattern_mdrange_tag, "default space",
                     default_exec_space, 0, N - 1, f);
  default_exec_space.fence(); // making sure the kernel is done
  auto time_default = timer.seconds();

  std::cout << "time default: " << time_default << std::endl;
  std::cout << "time spaces: " << time_spaces << std::endl;

  REQUIRE(time_default > 1.5 * time_spaces);
}