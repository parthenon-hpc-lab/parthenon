//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020-2022 The Parthenon collaboration
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

#if !(defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL))
    REQUIRE(test_wrapper_3d(parthenon::loop_pattern_tptvr_tag, default_exec_space) ==
            true);

    REQUIRE(test_wrapper_3d(parthenon::loop_pattern_simdfor_tag, default_exec_space) ==
            true);
#endif
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

#if !(defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL))
    REQUIRE(test_wrapper_4d(parthenon::loop_pattern_tptvr_tag, default_exec_space) ==
            true);

    REQUIRE(test_wrapper_4d(parthenon::loop_pattern_simdfor_tag, default_exec_space) ==
            true);
#endif
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

#if !(defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL))
    REQUIRE(test_wrapper_nested_3d(parthenon::outer_loop_pattern_teams_tag,
                                   parthenon::inner_loop_pattern_simdfor_tag,
                                   default_exec_space) == true);
#endif
  }

  SECTION("4D nested loops") {
    REQUIRE(test_wrapper_nested_4d(parthenon::outer_loop_pattern_teams_tag,
                                   parthenon::inner_loop_pattern_tvr_tag,
                                   default_exec_space) == true);

#if !(defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL))
    REQUIRE(test_wrapper_nested_4d(parthenon::outer_loop_pattern_teams_tag,
                                   parthenon::inner_loop_pattern_simdfor_tag,
                                   default_exec_space) == true);
#endif
  }
}

struct LargeNShortTBufferPack {
  int nghost;
  int ncells; // number of cells in the linear dimension - very simplistic
              // approach
  ParArray4D<Real> arr_in;
  // buffer in six direction, i plus, i minus, ...
  ParArray1D<Real> buf_ip, buf_im, buf_jp, buf_jm, buf_kp, buf_km;
  LargeNShortTBufferPack(const int nghost_, const int ncells_,
                         const ParArray4D<Real> arr_in_, ParArray1D<Real> buf_ip_,
                         ParArray1D<Real> buf_im_, ParArray1D<Real> buf_jp_,
                         ParArray1D<Real> buf_jm_, ParArray1D<Real> buf_kp_,
                         ParArray1D<Real> buf_km_)
      : nghost(nghost_), ncells(ncells_), arr_in(arr_in_), buf_ip(buf_ip_),
        buf_im(buf_im_), buf_jp(buf_jp_), buf_jm(buf_jm_), buf_kp(buf_kp_),
        buf_km(buf_km_) {}
  KOKKOS_INLINE_FUNCTION

  void operator()(const int dir, const int n, const int it) const {
    const int offset = n * (nghost * (ncells - 2 * nghost) * (ncells - 2 * nghost));

    if (dir == 0) {
      // it loops over [k,j,i]=[ [nghost,ncells-nghost),
      //                         [nghost,ncells-nghost),
      //                         [nghost,2*nghost) ]
      // const int it_nk = ncells-nghost*2;
      const int it_nj = ncells - nghost * 2;
      const int it_ni = nghost;
      const int it_k = it / (it_ni * it_nj);
      const int it_j = (it - it_k * it_ni * it_nj) / it_ni;
      const int it_i = it - it_k * it_ni * it_nj - it_j * it_ni;
      const int k = it_k + nghost;
      const int j = it_j + nghost;
      const int i = it_i + nghost;
      const int idx = it_i + it_ni * (it_j + it_nj * it_k);
      buf_im(offset + idx) = arr_in(n, k, j, i);
    } else if (dir == 1) {
      // it loops over [k,j,i]=[ [nghost,ncells-nghost),
      //                         [nghost,ncells-nghost),
      //                         [ncells-2*nghost,ncells-nghost) ]
      // const int it_nk = ncells-nghost*2;
      const int it_nj = ncells - nghost * 2;
      const int it_ni = nghost;
      const int it_k = it / (it_ni * it_nj);
      const int it_j = (it - it_k * it_ni * it_nj) / it_ni;
      const int it_i = it - it_k * it_ni * it_nj - it_j * it_ni;
      const int k = it_k + nghost;
      const int j = it_j + nghost;
      const int i = it_i + ncells - 2 * nghost;
      const int idx = it_i + it_ni * (it_j + it_nj * it_k);
      buf_ip(offset + idx) = arr_in(n, k, j, i);
    } else if (dir == 2) {
      // it loops over [k,j,i]=[ [nghost,ncells-nghost),
      //                         [nghost,2*nghost),
      //                         [nghost,ncells-nghost) ]
      // const int it_nk = ncells-nghost*2;
      const int it_nj = nghost;
      const int it_ni = ncells - nghost * 2;
      const int it_k = it / (it_ni * it_nj);
      const int it_j = (it - it_k * it_ni * it_nj) / it_ni;
      const int it_i = it - it_k * it_ni * it_nj - it_j * it_ni;
      const int k = it_k + nghost;
      const int j = it_j + nghost;
      const int i = it_i + nghost;
      const int idx = it_i + it_ni * (it_j + it_nj * it_k);
      buf_jm(offset + idx) = arr_in(n, k, j, i);
    } else if (dir == 3) {
      // it loops over [k,j,i]=[ [nghost,ncells-nghost),
      //                        [ncells-2*nghost,ncells),
      //                        [nghost,ncells-nghost) ]
      // const int it_nk = ncells-nghost*2;
      const int it_nj = nghost;
      const int it_ni = ncells - nghost * 2;
      const int it_k = it / (it_ni * it_nj);
      const int it_j = (it - it_k * it_ni * it_nj) / it_ni;
      const int it_i = it - it_k * it_ni * it_nj - it_j * it_ni;
      const int k = it_k + nghost;
      const int j = it_j + ncells - 2 * nghost;
      const int i = it_i + nghost;
      const int idx = it_i + it_ni * (it_j + it_nj * it_k);
      buf_jp(offset + idx) = arr_in(n, k, j, i);
    } else if (dir == 4) {
      // it loops over [k,j,i]=[ [nghost,2*nghost),
      //                         [nghost,ncells-nghost),
      //                         [nghost,ncells-nghost) ]
      // const int it_nk = nghost;
      const int it_nj = ncells - nghost * 2;
      const int it_ni = ncells - nghost * 2;
      const int it_k = it / (it_ni * it_nj);
      const int it_j = (it - it_k * it_ni * it_nj) / it_ni;
      const int it_i = it - it_k * it_ni * it_nj - it_j * it_ni;
      const int k = it_k + nghost;
      const int j = it_j + nghost;
      const int i = it_i + nghost;
      const int idx = it_i + it_ni * (it_j + it_nj * it_k);
      buf_km(offset + idx) = arr_in(n, k, j, i);
    } else if (dir == 5) {
      // it loops over [k,j,i]=[ [ncells-2*nghost,ncells),
      //                         [nghost,ncells-nghost),
      //                         [nghost,ncells-nghost) ]
      // const int it_nk = nghost;
      const int it_nj = ncells - nghost * 2;
      const int it_ni = ncells - nghost * 2;
      const int it_k = it / (it_ni * it_nj);
      const int it_j = (it - it_k * it_ni * it_nj) / it_ni;
      const int it_i = it - it_k * it_ni * it_nj - it_j * it_ni;
      const int k = it_k + ncells - 2 * nghost;
      const int j = it_j + nghost;
      const int i = it_i + nghost;
      const int idx = it_i + it_ni * (it_j + it_nj * it_k);
      buf_kp(offset + idx) = arr_in(n, k, j, i);
    }
  }

  void run(DevExecSpace exec_space) {
    const int nbuffers = 6; // number of buffers, here up, down, left, right, back, front
    auto M = arr_in.extent(0);
    auto slab_size = buf_im.extent(0) / M;
    parthenon::par_for(parthenon::loop_pattern_mdrange_tag, "LargeNShortTBufferPack",
                       exec_space, 0, nbuffers - 1, 0, M - 1, 0, slab_size - 1, *this);
  }

  template <typename TimeType>
  static void test_time(const TimeType time_default, const TimeType time_spaces) {
    // Test that streams are not introducing a performance penalty (within 10%
    // uncertainty). The efficiency here depends on the available HW.
    REQUIRE(time_spaces < 1.10 * time_default);
  }
};

struct SmallNLongTBufferPack {
  int nghost;
  int ncells; // number of cells in the linear dimension - very simplistic
              // approach
  ParArray4D<Real> arr_in;
  // buffer in six direction, i plus, i minus, ...
  ParArray1D<Real> buf_ip, buf_im, buf_jp, buf_jm, buf_kp, buf_km;
  SmallNLongTBufferPack(const int nghost_, const int ncells_,
                        const ParArray4D<Real> arr_in_, ParArray1D<Real> buf_ip_,
                        ParArray1D<Real> buf_im_, ParArray1D<Real> buf_jp_,
                        ParArray1D<Real> buf_jm_, ParArray1D<Real> buf_kp_,
                        ParArray1D<Real> buf_km_)
      : nghost(nghost_), ncells(ncells_), arr_in(arr_in_), buf_ip(buf_ip_),
        buf_im(buf_im_), buf_jp(buf_jp_), buf_jm(buf_jm_), buf_kp(buf_kp_),
        buf_km(buf_km_) {}
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

  void run(DevExecSpace exec_space) {
    auto M = arr_in.extent(0);
    const int nbuffers = 6; // number of buffers, here up, down, left, right, back, front
    parthenon::par_for(parthenon::loop_pattern_mdrange_tag, "SmallNLongTBufferPack",
                       exec_space, 0, M - 1, 0, nbuffers - 1, *this);
  }

  template <typename TimeType>
  static void test_time(const TimeType time_default, const TimeType time_spaces) {
    // Test that streams are not introducing a performance penalty (within 10%
    // uncertainty). The efficiency here depends on the available HW.
    REQUIRE(time_spaces < 1.10 * time_default);
  }
};

template <class BufferPack>
void test_wrapper_buffer_pack_overlapping_space_instances(const std::string &test_name) {
  auto default_exec_space = DevExecSpace();

  const int N = 24;      // ~meshblock size
  const int M = 5;       // ~nhydro
  const int nspaces = 2; // number of streams
  const int nghost = 2;  // number of ghost zones
  const int buf_size = M * nghost * (N - 2 * nghost) * (N - 2 * nghost);

  std::vector<BufferPack> functs;
  std::vector<DevExecSpace> exec_spaces;

  for (auto n = 0; n < nspaces; n++) {
    functs.push_back(BufferPack(
        nghost, N, ParArray4D<Real>("SpaceInstance in", M, N, N, N),
        ParArray1D<Real>("buf_ip", buf_size), ParArray1D<Real>("buf_im", buf_size),
        ParArray1D<Real>("buf_jp", buf_size), ParArray1D<Real>("buf_jm", buf_size),
        ParArray1D<Real>("buf_kp", buf_size), ParArray1D<Real>("buf_kp", buf_size)));
    exec_spaces.push_back(parthenon::SpaceInstance<DevExecSpace>::create());
  }

  // warmup
  for (auto it = 0; it < 10; it++) {
    for (auto n = 0; n < nspaces; n++) {
      functs[n].run(exec_spaces[n]);
    }
  }
  Kokkos::fence();

  Kokkos::Timer timer;

  // meausre time using two execution space simultaneously
  // race condition in access to arr_dev doesn't matter for this test
  for (auto n = 0; n < nspaces; n++) {
    functs[n].run(exec_spaces[n]);
  }

  Kokkos::fence();
  auto time_spaces = timer.seconds();

  timer.reset();

  // measure runtime using the default execution space
  for (auto n = 0; n < nspaces; n++) {
    functs[n].run(default_exec_space);
  }

  default_exec_space.fence(); // making sure the kernel is done
  auto time_default = timer.seconds();

  std::cout << test_name << std::endl;
  std::cout << "time default: " << time_default << std::endl;
  std::cout << "time spaces: " << time_spaces << std::endl;

  // make sure this test is reasonable IIF streams actually overlap, which is
  // not the case for the OpenMP backend at this point
  if (parthenon::SpaceInstance<DevExecSpace>::overlap()) {
    BufferPack::test_time(time_default, time_spaces);
  }
}
TEST_CASE("Overlapping SpaceInstances", "[wrapper][performance]") {
  SECTION("Many Threads Short Kernel") {
    test_wrapper_buffer_pack_overlapping_space_instances<LargeNShortTBufferPack>(
        "Many Threads Short Kernel");
  }
  SECTION("Few Threads Long Kernel") {
    test_wrapper_buffer_pack_overlapping_space_instances<SmallNLongTBufferPack>(
        "Few Threads Long Kernel");
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

//  GIVEN("A derived class") {
//    THEN("We can create a unique_ptr to this on device") {
//      auto ptr = parthenon::DeviceAllocate<MyTestDerivedClass>();
//      auto devptr = ptr.get();
//
//      Kokkos::parallel_for(
//          Kokkos::RangePolicy<DevExecSpace>(0, 1),
//          KOKKOS_LAMBDA(const int i) { buffer(i) = devptr->GetInt(); });
//
//      auto buffer_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), buffer);
//      REQUIRE(buffer_h[0] == test_int);
//    }
//  }
}
