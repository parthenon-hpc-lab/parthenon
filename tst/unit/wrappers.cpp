#include "../../src/athena.hpp"
#include "Kokkos_Macros.hpp"
#include <catch2/catch.hpp>
#include <iostream>
#include <random>

using parthenon::AthenaArray3D;
using parthenon::AthenaArray4D;

template <class T> bool test_wrapper_3d(T loop_pattern) {

  // https://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution
  std::random_device
      rd; // Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<parthenon::Real> dis(-1.0, 1.0);

  const int N = 32;
  AthenaArray3D<> arr_dev("device", N, N, N);
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
      loop_pattern, "unit test 3D", 0, N - 1, 0, N - 1, 0, N - 1,
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

template <class T> bool test_wrapper_4d(T loop_pattern) {

  // https://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution
  std::random_device
      rd; // Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<parthenon::Real> dis(-1.0, 1.0);

  const int N = 32;
  AthenaArray4D<> arr_dev("device", N, N, N, N);
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
      loop_pattern, "unit test 4D", 0, N - 1, 0, N - 1, 0, N - 1, 0, N - 1,
      KOKKOS_LAMBDA(const int n, const int k, const int j, const int i) {
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

TEST_CASE("1D Range 3D", "[wrapper]") {
  REQUIRE(test_wrapper_3d(parthenon::loop_pattern_range_tag) == true);
}

TEST_CASE("MDRange 3D", "[wrapper]") {
  REQUIRE(test_wrapper_3d(parthenon::loop_pattern_mdrange_tag) == true);
}

TEST_CASE("TPTTRTVR 3D", "[wrapper]") {
  REQUIRE(test_wrapper_3d(parthenon::loop_pattern_tpttrtvr_tag) == true);
}

TEST_CASE("TPX 3D", "[wrapper]") {
  REQUIRE(test_wrapper_3d(parthenon::loop_pattern_tpx_tag) == true);
}

TEST_CASE("SIMDFOR 3D", "[wrapper]") {
  REQUIRE(test_wrapper_3d(parthenon::loop_pattern_simdfor_tag) == true);
}

TEST_CASE("1D Range 4D", "[wrapper]") {
  REQUIRE(test_wrapper_4d(parthenon::loop_pattern_range_tag) == true);
}

TEST_CASE("MDRange 4D", "[wrapper]") {
  REQUIRE(test_wrapper_4d(parthenon::loop_pattern_mdrange_tag) == true);
}

TEST_CASE("TPTTRTVR 4D", "[wrapper]") {
  REQUIRE(test_wrapper_4d(parthenon::loop_pattern_tpttrtvr_tag) == true);
}

TEST_CASE("TPX 4D", "[wrapper]") {
  REQUIRE(test_wrapper_4d(parthenon::loop_pattern_tpx_tag) == true);
}

TEST_CASE("SIMDFOR 4D", "[wrapper]") {
  REQUIRE(test_wrapper_4d(parthenon::loop_pattern_simdfor_tag) == true);
}
