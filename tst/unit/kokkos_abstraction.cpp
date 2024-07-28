//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020-2022 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2024. Triad National Security, LLC. All rights reserved.
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
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

#include "Kokkos_Core.hpp"
#include "Kokkos_Macros.hpp"
#include "basic_types.hpp"
#include "kokkos_abstraction.hpp"
#include "parthenon_array_generic.hpp"

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

template <size_t ND, typename T, typename... Args>
auto ParArrayND(Args &&...args) {
  static_assert(ND <= 8, "ParArrayND supoorted up to ND=8");
  if constexpr (ND == 0) {
    return parthenon::ParArray0D<T>(std::forward<Args>(args)...);
  } else if constexpr (ND == 1) {
    return parthenon::ParArray1D<T>(std::forward<Args>(args)...);
  } else if constexpr (ND == 2) {
    return parthenon::ParArray2D<T>(std::forward<Args>(args)...);
  } else if constexpr (ND == 3) {
    return parthenon::ParArray3D<T>(std::forward<Args>(args)...);
  } else if constexpr (ND == 4) {
    return parthenon::ParArray4D<T>(std::forward<Args>(args)...);
  } else if constexpr (ND == 5) {
    return parthenon::ParArray5D<T>(std::forward<Args>(args)...);
  } else if constexpr (ND == 6) {
    return parthenon::ParArray6D<T>(std::forward<Args>(args)...);
  } else if constexpr (ND == 7) {
    return parthenon::ParArray7D<T>(std::forward<Args>(args)...);
  } else if constexpr (ND == 8) {
    return parthenon::ParArray8D<T>(std::forward<Args>(args)...);
  }
}
template <size_t ND, typename T, typename... Args>
auto HostArrayND(Args &&...args) {
  static_assert(ND <= 7, "HostArrayND supoorted up to ND=7");
  if constexpr (ND == 0) {
    return parthenon::HostArray0D<T>(std::forward<Args>(args)...);
  } else if constexpr (ND == 1) {
    return parthenon::HostArray1D<T>(std::forward<Args>(args)...);
  } else if constexpr (ND == 2) {
    return parthenon::HostArray2D<T>(std::forward<Args>(args)...);
  } else if constexpr (ND == 3) {
    return parthenon::HostArray3D<T>(std::forward<Args>(args)...);
  } else if constexpr (ND == 4) {
    return parthenon::HostArray4D<T>(std::forward<Args>(args)...);
  } else if constexpr (ND == 5) {
    return parthenon::HostArray5D<T>(std::forward<Args>(args)...);
  } else if constexpr (ND == 6) {
    return parthenon::HostArray6D<T>(std::forward<Args>(args)...);
  } else if constexpr (ND == 7) {
    return parthenon::HostArray7D<T>(std::forward<Args>(args)...);
  }
}

enum class lbounds { integer, indexrange };

template <size_t Rank>
struct test_wrapper_nd_impl {
  template <size_t N>
  using Sequence = std::make_index_sequence<N>;
  int N, indices[Rank - 1], int_bounds[2 * Rank];
  parthenon::IndexRange bounds[Rank];
  decltype(ParArrayND<Rank, Real>()) arr_dev;
  decltype(HostArrayND<Rank, Real>()) arr_host_orig, arr_host_mod;

  test_wrapper_nd_impl(const int _N = 32) : N(_N) {
    arr_dev = GetArray(typename parthenon::meta::sequence_of_ones<Rank>::value());
    arr_host_orig = Kokkos::create_mirror(arr_dev);
    arr_host_mod = Kokkos::create_mirror(arr_dev);
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<Real> dis(-1.0, 1.0);
    par_for_init<Rank>(std::make_index_sequence<Rank - 1>(), gen, dis);
  }

  template <size_t... Is>
  auto GetArray(std::index_sequence<Is...>) {
    static_assert(sizeof...(Is) == Rank);
    return ParArrayND<Rank, Real>("device", N * Is...);
  }

  template <size_t LoopsLeft, size_t... Is>
  void par_for_init(std::index_sequence<Is...>, std::mt19937 &gen,
                    std::uniform_real_distribution<Real> &dis) {
    constexpr size_t id = Rank - LoopsLeft;
    bounds[id].s = 0;
    bounds[id].e = N - 1;
    int_bounds[2 * id] = 0;
    int_bounds[2 * id + 1] = N - 1;
    if constexpr (LoopsLeft == 1) {
      for (int i = 0; i < N; i++) {
        arr_host_orig(indices[Is]..., i) = 0.; // dis(gen);
      }
    } else {
      for (int j = 0; j < N; j++) {
        indices[Rank - LoopsLeft] = j;
        par_for_init<LoopsLeft - 1>(Sequence<Rank - 1>(), gen, dis);
      }
    }
  }

  template <typename... KJI>
  KOKKOS_INLINE_FUNCTION Real increment_data(KJI... kji) {
    static_assert(Rank == sizeof...(KJI), "number of indices matches Rank");
    int inc = 0;
    int inds[sizeof...(KJI)]{kji...};
    for (int i = 0; i < Rank; i++) {
      inc += N * inds[i];
    }
    return static_cast<Real>(inc);
  }

  template <size_t LoopsLeft, size_t... Is>
  bool par_for_comp(std::index_sequence<Is...>) {
    bool all_same = true;
    if constexpr (LoopsLeft == 1) {
      for (int i = 0; i < N; i++) {
        if (arr_host_orig(indices[Is]..., i) + increment_data(indices[Is]..., i) !=
            arr_host_mod(indices[Is]..., i)) {
          all_same = false;
        }
      }
    } else {
      for (int j = 0; j < N; j++) {
        indices[Rank - LoopsLeft] = j;
        all_same = par_for_comp<LoopsLeft - 1>(Sequence<Rank - 1>());
      }
    }
    return all_same;
  }

  template <lbounds bound_type, typename T, size_t... Ids, typename... Ts>
  bool dispatch(parthenon::meta::TypeList<Ts...>, std::index_sequence<Ids...>,
                T loop_pattern, DevExecSpace exec_space) {
    Kokkos::deep_copy(arr_dev, arr_host_orig);
    if constexpr (bound_type == lbounds::integer) {
      parthenon::par_for(
          loop_pattern, "unit test ND integer bounds", exec_space, int_bounds[Ids]...,
          KOKKOS_LAMBDA(Ts... args) {
            arr_dev(std::forward<decltype(args)>(args)...) +=
                increment_data(std::forward<decltype(args)>(args)...);
          });
    } else {
      parthenon::par_for(
          loop_pattern, "unit test ND IndexRange bounds", exec_space, bounds[Ids]...,
          KOKKOS_LAMBDA(Ts... args) {
            arr_dev(std::forward<decltype(args)>(args)...) +=
                increment_data(std::forward<decltype(args)>(args)...);
          });
    }
    Kokkos::deep_copy(arr_host_mod, arr_dev);
    return par_for_comp<Rank>(Sequence<Rank - 1>());
  }
  template <typename T>
  void test(T loop_pattern, DevExecSpace exec_space) {
    /* REQUIRE(dispatch<lbounds::integer>( */
    /*             typename parthenon::meta::ListOfType<Rank, const int>::value(), */
    /*             Sequence<2 * Rank>(), loop_pattern, exec_space) == true); */
    REQUIRE(dispatch<lbounds::indexrange>(
                typename parthenon::meta::ListOfType<Rank, const int>::value(),
                Sequence<Rank>(), loop_pattern, exec_space) == true);
  }

  template <typename T>
  bool par_for_dev(T loop_pattern, DevExecSpace exec_space, lbounds bound_type) {
    return dispatch(Sequence(), loop_pattern, exec_space, bound_type);
  }
};

template <size_t Rank>
void test_wrapper_nd(DevExecSpace exec_space) {
  auto wrappernd = test_wrapper_nd_impl<Rank>(2);
  wrappernd.test(parthenon::loop_pattern_flatrange_tag, exec_space);
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
    test_wrapper_nd<4>(default_exec_space);
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
  parthenon::IndexRange rng{0, N - 1};

  // Compute the 2nd order centered derivative in x
  parthenon::par_for_outer(
      outer_loop_pattern, "unit test Nested 4D", exec_space, scratch_size_in_bytes,
      scratch_level, 0, N - 1, 0, N - 1, 0, N - 1,

      KOKKOS_LAMBDA(parthenon::team_mbr_t team_member, const int n, const int k,
                    const int j) {
        // Load a pencil in x to minimize DRAM accesses (and test scratch pad)
        parthenon::ScratchPad1D<Real> scratch_u(team_member.team_scratch(scratch_level),
                                                N);
        parthenon::par_for_inner(inner_loop_pattern, team_member, rng,
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

template <class T>
bool test_wrapper_reduce_1d(T loop_pattern, DevExecSpace exec_space) {
  constexpr int N = 10;
  parthenon::IndexRange r{0, N - 1};
  parthenon::ParArray1D<int> buffer("Testing buffer", N);
  // Initialize data
  parthenon::par_for(
      loop_pattern, "Initialize parallel reduce array", exec_space, r,
      KOKKOS_LAMBDA(const int i) { buffer(i) = i; });
  int total = 0;
  for (int i = 0; i < N; ++i) {
    total += i;
  }
  int test_tot = 0;
  parthenon::par_reduce(
      loop_pattern, "Sum via par reduce", exec_space, r,
      KOKKOS_LAMBDA(const int i, int &t) { t += i; }, Kokkos::Sum<int>(test_tot));
  return total == test_tot;
}

template <class T>
bool test_wrapper_reduce_3d(T loop_pattern, DevExecSpace exec_space) {
  constexpr int N = 10;
  parthenon::ParArray3D<int> buffer("Testing buffer", N, N, N);
  // Initialize data
  parthenon::par_for(
      loop_pattern, "Initialize parallel reduce array", exec_space, 0, N - 1, 0, N - 1, 0,
      N - 1, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        buffer(k, j, i) = i + j + k;
      });
  int tot = 0;
  for (int k = 0; k < N; ++k) {
    for (int j = 0; j < N; ++j) {
      for (int i = 0; i < N; ++i) {
        tot += i + j + k;
      }
    }
  }
  int test_tot = 0;
  parthenon::par_reduce(
      loop_pattern, "Sum via par reduce", exec_space, 0, N - 1, 0, N - 1, 0, N - 1,
      KOKKOS_LAMBDA(const int k, const int j, const int i, int &t) { t += i + j + k; },
      Kokkos::Sum<int>(test_tot));
  return tot == test_tot;
}

template <class T>
bool test_wrapper_reduce_4d(T loop_pattern, DevExecSpace exec_space) {
  constexpr int N = 10;
  parthenon::ParArray4D<int> buffer("Testing buffer", N, N, N, N);
  // Initialize data
  parthenon::par_for(
      loop_pattern, "Initialize parallel reduce array", exec_space, 0, N - 1, 0, N - 1, 0,
      N - 1, 0, N - 1, KOKKOS_LAMBDA(const int n, const int k, const int j, const int i) {
        buffer(n, k, j, i) = i + j + k + n;
      });
  int tot = 0;
  for (int n = 0; n < N; ++n) {
    for (int k = 0; k < N; ++k) {
      for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
          tot += i + j + k + n;
        }
      }
    }
  }
  int test_tot = 0;
  parthenon::par_reduce(
      loop_pattern, "Sum via par reduce", exec_space, 0, N - 1, 0, N - 1, 0, N - 1, 0,
      N - 1,
      KOKKOS_LAMBDA(const int n, const int k, const int j, const int i, int &t) {
        t += i + j + k + n;
      },
      Kokkos::Sum<int>(test_tot));
  return tot == test_tot;
}

TEST_CASE("Parallel reduce", "[par_reduce]") {
  auto default_exec_space = DevExecSpace();
  SECTION("1D loops") {
    REQUIRE(test_wrapper_reduce_1d(parthenon::loop_pattern_flatrange_tag,
                                   default_exec_space) == true);
    if constexpr (std::is_same<DevExecSpace, Kokkos::Serial>::value) {
      REQUIRE(test_wrapper_reduce_1d(parthenon::loop_pattern_simdfor_tag,
                                     default_exec_space) == true);
    }
  }

  SECTION("3D loops") {
    REQUIRE(test_wrapper_reduce_3d(parthenon::loop_pattern_flatrange_tag,
                                   default_exec_space) == true);
    REQUIRE(test_wrapper_reduce_3d(parthenon::loop_pattern_mdrange_tag,
                                   default_exec_space) == true);
  }

  SECTION("4D loops") {
    REQUIRE(test_wrapper_reduce_4d(parthenon::loop_pattern_flatrange_tag,
                                   default_exec_space) == true);
    REQUIRE(test_wrapper_reduce_4d(parthenon::loop_pattern_mdrange_tag,
                                   default_exec_space) == true);
  }
}
