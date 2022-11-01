//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020-2022 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2022. Triad National Security, LLC. All rights reserved.
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
#include <vector>

#include <catch2/catch.hpp>

#include "kokkos_abstraction.hpp"

using parthenon::DevExecSpace;
using parthenon::ParArray1D;
using parthenon::ParArray2D;
using parthenon::ParArray3D;
using parthenon::ParArray4D;
using Real = double;

template <class T>
bool test_wrapper_1d(T loop_pattern, DevExecSpace exec_space) {
  // https://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution
  std::random_device rd;  // Will be used to obtain a seed for the random number engine
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
      KOKKOS_LAMBDA(const int i) { arr_dev(i) += static_cast<Real>(i); });

  // Copy array back from device to host
  Kokkos::deep_copy(arr_host_mod, arr_dev);

  bool all_same = true;

  // compare data on the host
  for (int i = 0; i < N; i++)
    if (arr_host_orig(i) + static_cast<Real>(i) != arr_host_mod(i)) {
      all_same = false;
    }

  return all_same;
}

template <class T>
bool test_wrapper_2d(T loop_pattern, DevExecSpace exec_space) {
  // https://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution
  std::random_device rd;  // Will be used to obtain a seed for the random number engine
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
        arr_dev(j, i) += static_cast<Real>(i + N * j);
      });

  // Copy array back from device to host
  Kokkos::deep_copy(arr_host_mod, arr_dev);

  bool all_same = true;

  // compare data on the host
  for (int j = 0; j < N; j++)
    for (int i = 0; i < N; i++)
      if (arr_host_orig(j, i) + static_cast<Real>(i + N * j) != arr_host_mod(j, i)) {
        all_same = false;
      }

  return all_same;
}

template <class T>
bool test_wrapper_3d(T loop_pattern, DevExecSpace exec_space) {
  // https://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution
  std::random_device rd;  // Will be used to obtain a seed for the random number engine
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
        arr_dev(k, j, i) += static_cast<Real>(i + N * (j + N * k));
      });

  // Copy array back from device to host
  Kokkos::deep_copy(arr_host_mod, arr_dev);

  bool all_same = true;

  // compare data on the host
  for (int k = 0; k < N; k++)
    for (int j = 0; j < N; j++)
      for (int i = 0; i < N; i++)
        if (arr_host_orig(k, j, i) + static_cast<Real>(i + N * (j + N * k)) !=
            arr_host_mod(k, j, i)) {
          all_same = false;
        }

  return all_same;
}

template <class T>
bool test_wrapper_4d(T loop_pattern, DevExecSpace exec_space) {
  // https://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution
  std::random_device rd;  // Will be used to obtain a seed for the random number engine
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
      loop_pattern, "unit test 4D", exec_space, 0, N - 1, 0, N - 1, 0, N - 1, 0, N - 1,
      KOKKOS_LAMBDA(const int n, const int k, const int j, const int i) {
        arr_dev(n, k, j, i) += static_cast<Real>(i + N * (j + N * (k + n)));
      });

  // Copy array back from device to host
  Kokkos::deep_copy(arr_host_mod, arr_dev);

  bool all_same = true;

  // compare data on the host
  for (int n = 0; n < N; n++)
    for (int k = 0; k < N; k++)
      for (int j = 0; j < N; j++)
        for (int i = 0; i < N; i++)
          if (arr_host_orig(n, k, j, i) + static_cast<Real>(i + N * (j + N * (k + n))) !=
              arr_host_mod(n, k, j, i)) {
            all_same = false;
          }

  return all_same;
}

TEST_CASE("par_for loops", "[wrapper]") {
  auto default_exec_space = DevExecSpace();

  SECTION("1D loops") {
    REQUIRE(test_wrapper_1d(parthenon::loop_pattern_flatrange_tag, default_exec_space) ==
            true);
  }

  SECTION("2D loops") {
    REQUIRE(test_wrapper_2d(parthenon::loop_pattern_mdrange_tag, default_exec_space) ==
            true);
  }

  SECTION("3D loops") {
    REQUIRE(test_wrapper_3d(parthenon::loop_pattern_flatrange_tag, default_exec_space) ==
            true);

    REQUIRE(test_wrapper_3d(parthenon::loop_pattern_mdrange_tag, default_exec_space) ==
            true);

    REQUIRE(test_wrapper_3d(parthenon::loop_pattern_tpttrtvr_tag, default_exec_space) ==
            true);

    REQUIRE(test_wrapper_3d(parthenon::loop_pattern_tpttr_tag, default_exec_space) ==
            true);

    if constexpr (std::is_same<Kokkos::DefaultExecutionSpace,
                               Kokkos::DefaultHostExecutionSpace>::value) {
      REQUIRE(test_wrapper_3d(parthenon::loop_pattern_tptvr_tag, default_exec_space) ==
              true);

      REQUIRE(test_wrapper_3d(parthenon::loop_pattern_simdfor_tag, default_exec_space) ==
              true);
    }
  }

  SECTION("4D loops") {
    REQUIRE(test_wrapper_4d(parthenon::loop_pattern_flatrange_tag, default_exec_space) ==
            true);

    REQUIRE(test_wrapper_4d(parthenon::loop_pattern_mdrange_tag, default_exec_space) ==
            true);

    REQUIRE(test_wrapper_4d(parthenon::loop_pattern_tpttrtvr_tag, default_exec_space) ==
            true);

    REQUIRE(test_wrapper_4d(parthenon::loop_pattern_tpttr_tag, default_exec_space) ==
            true);

    if constexpr (std::is_same<Kokkos::DefaultExecutionSpace,
                               Kokkos::DefaultHostExecutionSpace>::value) {
      REQUIRE(test_wrapper_4d(parthenon::loop_pattern_tptvr_tag, default_exec_space) ==
              true);

      REQUIRE(test_wrapper_4d(parthenon::loop_pattern_simdfor_tag, default_exec_space) ==
              true);
    }
  }
}

template <class OuterLoopPattern, class InnerLoopPattern>
bool test_wrapper_nested_3d(OuterLoopPattern outer_loop_pattern,
                            InnerLoopPattern inner_loop_pattern,
                            DevExecSpace exec_space) {
  // Compute the 2nd order centered derivative in x of i+1^2 * j+1^2 * k+1^2

  const int N = 32;
  ParArray3D<Real> dev_u("device u", N, N, N);
  ParArray3D<Real> dev_du("device du", N, N, N - 2);
  auto host_u = Kokkos::create_mirror(dev_u);
  auto host_du = Kokkos::create_mirror(dev_du);

  // initialize with i^2 * j^2 * k^2
  for (int n = 0; n < N; n++)
    for (int k = 0; k < N; k++)
      for (int j = 0; j < N; j++)
        for (int i = 0; i < N; i++)
          host_u(k, j, i) = pow((i + 1) * (j + 2) * (k + 3), 2.0);

  // Copy host array content to device
  Kokkos::deep_copy(dev_u, host_u);

  // Compute the scratch memory needs
  const int scratch_level = 0;
  size_t scratch_size_in_bytes = parthenon::ScratchPad1D<Real>::shmem_size(N);

  // Compute the 2nd order centered derivative in x
  parthenon::par_for_outer(
      outer_loop_pattern, "unit test Nested 3D", exec_space, scratch_size_in_bytes,
      scratch_level, 0, N - 1, 0, N - 1,

      KOKKOS_LAMBDA(parthenon::team_mbr_t team_member, const int k, const int j) {
        // Load a pencil in x to minimize DRAM accesses (and test scratch pad)
        parthenon::ScratchPad1D<Real> scratch_u(team_member.team_scratch(scratch_level),
                                                N);
        parthenon::par_for_inner(inner_loop_pattern, team_member, 0, N - 1,
                                 [&](const int i) { scratch_u(i) = dev_u(k, j, i); });
        // Sync all threads in the team so that scratch memory is consistent
        team_member.team_barrier();

        // Compute the derivative from scratch memory
        parthenon::par_for_inner(
            inner_loop_pattern, team_member, 1, N - 2, [&](const int i) {
              dev_du(k, j, i - 1) = (scratch_u(i + 1) - scratch_u(i - 1)) / 2.;
            });
      });

  // Copy array back from device to host
  Kokkos::deep_copy(host_du, dev_du);

  Real max_rel_err = -1;
  const Real rel_tol = std::numeric_limits<Real>::epsilon();

  // compare data on the host
  for (int k = 0; k < N; k++) {
    for (int j = 0; j < N; j++) {
      for (int i = 1; i < N - 1; i++) {
        const Real analytic = 2.0 * (i + 1) * pow((j + 2) * (k + 3), 2.0);
        const Real err = host_du(k, j, i - 1) - analytic;

        max_rel_err = fmax(fabs(err / analytic), max_rel_err);
      }
    }
  }

  return max_rel_err < rel_tol;
}

template <class OuterLoopPattern, class InnerLoopPattern>
bool test_wrapper_nested_4d(OuterLoopPattern outer_loop_pattern,
                            InnerLoopPattern inner_loop_pattern,
                            DevExecSpace exec_space) {
  // Compute the 2nd order centered derivative in x of i+1^2 * j+1^2 * k+1^2 * n+1^2

  const int N = 32;
  ParArray4D<Real> dev_u("device u", N, N, N, N);
  ParArray4D<Real> dev_du("device du", N, N, N, N - 2);
  auto host_u = Kokkos::create_mirror(dev_u);
  auto host_du = Kokkos::create_mirror(dev_du);

  // initialize with i^2 * j^2 * k^2
  for (int n = 0; n < N; n++)
    for (int k = 0; k < N; k++)
      for (int j = 0; j < N; j++)
        for (int i = 0; i < N; i++)
          host_u(n, k, j, i) = pow((i + 1) * (j + 2) * (k + 3) * (n + 4), 2.0);

  // Copy host array content to device
  Kokkos::deep_copy(dev_u, host_u);

  // Compute the scratch memory needs
  const int scratch_level = 0;
  size_t scratch_size_in_bytes = parthenon::ScratchPad1D<Real>::shmem_size(N);

  // Compute the 2nd order centered derivative in x
  parthenon::par_for_outer(
      outer_loop_pattern, "unit test Nested 4D", exec_space, scratch_size_in_bytes,
      scratch_level, 0, N - 1, 0, N - 1, 0, N - 1,

      KOKKOS_LAMBDA(parthenon::team_mbr_t team_member, const int n, const int k,
                    const int j) {
        // Load a pencil in x to minimize DRAM accesses (and test scratch pad)
        parthenon::ScratchPad1D<Real> scratch_u(team_member.team_scratch(scratch_level),
                                                N);
        parthenon::par_for_inner(inner_loop_pattern, team_member, 0, N - 1,
                                 [&](const int i) { scratch_u(i) = dev_u(n, k, j, i); });
        // Sync all threads in the team so that scratch memory is consistent
        team_member.team_barrier();

        // Compute the derivative from scratch memory
        parthenon::par_for_inner(
            inner_loop_pattern, team_member, 1, N - 2, [&](const int i) {
              dev_du(n, k, j, i - 1) = (scratch_u(i + 1) - scratch_u(i - 1)) / 2.;
            });
      });

  // Copy array back from device to host
  Kokkos::deep_copy(host_du, dev_du);

  Real max_rel_err = -1;
  const Real rel_tol = std::numeric_limits<Real>::epsilon();

  // compare data on the host
  for (int n = 0; n < N; n++) {
    for (int k = 0; k < N; k++) {
      for (int j = 0; j < N; j++) {
        for (int i = 1; i < N - 1; i++) {
          const Real analytic = 2.0 * (i + 1) * pow((j + 2) * (k + 3) * (n + 4), 2.0);
          const Real err = host_du(n, k, j, i - 1) - analytic;

          max_rel_err = fmax(fabs(err / analytic), max_rel_err);
        }
      }
    }
  }

  return max_rel_err < rel_tol;
}

TEST_CASE("nested par_for loops", "[wrapper]") {
  auto default_exec_space = DevExecSpace();

  SECTION("3D nested loops") {
    REQUIRE(test_wrapper_nested_3d(parthenon::outer_loop_pattern_teams_tag,
                                   parthenon::inner_loop_pattern_tvr_tag,
                                   default_exec_space) == true);

    if constexpr (std::is_same<Kokkos::DefaultExecutionSpace,
                               Kokkos::DefaultHostExecutionSpace>::value) {
      REQUIRE(test_wrapper_nested_3d(parthenon::outer_loop_pattern_teams_tag,
                                     parthenon::inner_loop_pattern_simdfor_tag,
                                     default_exec_space) == true);
    }
  }

  SECTION("4D nested loops") {
    REQUIRE(test_wrapper_nested_4d(parthenon::outer_loop_pattern_teams_tag,
                                   parthenon::inner_loop_pattern_tvr_tag,
                                   default_exec_space) == true);

    if constexpr (std::is_same<Kokkos::DefaultExecutionSpace,
                               Kokkos::DefaultHostExecutionSpace>::value) {
      REQUIRE(test_wrapper_nested_4d(parthenon::outer_loop_pattern_teams_tag,
                                     parthenon::inner_loop_pattern_simdfor_tag,
                                     default_exec_space) == true);
    }
  }
}

template <class T>
bool test_wrapper_scan_1d(T loop_pattern, DevExecSpace exec_space) {
  const int N = 10;
  parthenon::ParArray1D<int> buffer("Testing buffer", N);
  // Initialize data
  parthenon::par_for(
      loop_pattern, "Initialize parallel scan array", exec_space, 0, N - 1,
      KOKKOS_LAMBDA(const int i) { buffer(i) = i; });

  parthenon::ParArray1D<int> scanned("Result of scan", N);
  int result;
  parthenon::par_scan(
      loop_pattern, "Parallel scan", exec_space, 0, N - 1,
      KOKKOS_LAMBDA(const int i, int &partial_sum, bool is_final) {
        if (is_final) {
          scanned(i) = partial_sum;
        }
        partial_sum += buffer(i);
      },
      result);

  // compare data on the host
  bool all_same = true;
  auto scanned_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), scanned);
  for (int i = 0; i < N; i++) {
    int ans = 0;
    for (int j = 0; j < i; j++) {
      ans += j;
    }
    if (scanned_h(i) != ans) {
      all_same = false;
    }
  }

  return all_same;
}

TEST_CASE("Parallel scan", "[par_scan]") {
  auto default_exec_space = DevExecSpace();

  SECTION("1D loops") {
    REQUIRE(test_wrapper_scan_1d(parthenon::loop_pattern_flatrange_tag,
                                 default_exec_space) == true);
  }
}

struct MyTestStruct {
  int i;
};

constexpr int test_int = 2;

class MyTestBaseClass {
  KOKKOS_INLINE_FUNCTION
  virtual int GetInt() = 0;
};

struct MyTestDerivedClass : public MyTestBaseClass {
  KOKKOS_INLINE_FUNCTION
  int GetInt() { return test_int; }
};

TEST_CASE("Device Object Allocation", "[wrapper]") {
  parthenon::ParArray1D<int> buffer("Testing buffer", 1);

  GIVEN("A struct") {
    THEN("We can create a unique_ptr to this on device") {
      { auto ptr = parthenon::DeviceAllocate<MyTestStruct>(); }
    }
  }

  GIVEN("An initialized host struct") {
    MyTestStruct s;
    s.i = 5;
    THEN("We can create a unique_ptr to a copy on device") {
      auto ptr = parthenon::DeviceCopy<MyTestStruct>(s);
      auto devptr = ptr.get();

      Kokkos::parallel_for(
          Kokkos::RangePolicy<DevExecSpace>(0, 1),
          KOKKOS_LAMBDA(const int i) { buffer(i) = devptr->i; });

      auto buffer_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), buffer);
      REQUIRE(buffer_h[0] == s.i);
    }
  }

  GIVEN("A derived class") {
    THEN("We can create a unique_ptr to this on device") {
      auto ptr = parthenon::DeviceAllocate<MyTestDerivedClass>();
      auto devptr = ptr.get();

      Kokkos::parallel_for(
          Kokkos::RangePolicy<DevExecSpace>(0, 1),
          KOKKOS_LAMBDA(const int i) { buffer(i) = devptr->GetInt(); });

      auto buffer_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), buffer);
      REQUIRE(buffer_h[0] == test_int);
    }
  }
}
