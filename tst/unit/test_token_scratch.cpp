//========================================================================================
// (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001 for Los
// Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
// for the U.S. Department of Energy/National Nuclear Security Administration. All rights
// in the program are reserved by Triad National Security, LLC, and the U.S. Department
// of Energy/National Nuclear Security Administration. The Government is granted for
// itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do so.
//========================================================================================
// This file was made in part with generative AI

//! \file test_token_scratch.cpp
//! \brief Unit tests for TokenScratchPool

#include <cmath>

#ifndef CATCH_CONFIG_FAST_COMPILE
#define CATCH_CONFIG_FAST_COMPILE
#include <catch2/catch.hpp>
#endif

#include <Kokkos_Core.hpp>

#include "utils/token_scratch.hpp"

SCENARIO("TokenScratchPool basic allocation and usage", "[TokenScratch][Basic]") {
  GIVEN("A TokenScratchPool with 64KB per token") {
    constexpr size_t scratch_bytes = 64 * 1024; // 64KB per token
    constexpr int n_iterations = 100;

    parthenon::TokenScratchPool<> pool(scratch_bytes);

    WHEN("We allocate views in parallel iterations") {
      Kokkos::View<double *> results("results", n_iterations);

      Kokkos::parallel_for(
          "test_basic", n_iterations, KOKKOS_LAMBDA(const int i) {
            auto scratch = pool.acquire();

            // Allocate views
            auto doubles = scratch.template allocate_view<double>(100);
            auto ints = scratch.template allocate_view<int>(50);

            // Initialize
            for (int j = 0; j < 100; ++j) {
              doubles(j) = static_cast<double>(i + j);
            }
            for (int j = 0; j < 50; ++j) {
              ints(j) = i * j;
            }

            // Compute sum
            double sum = 0.0;
            for (int j = 0; j < 100; ++j) {
              sum += doubles(j);
            }
            for (int j = 0; j < 50; ++j) {
              sum += static_cast<double>(ints(j));
            }
            results(i) = sum;
          });

      Kokkos::fence();

      THEN("Results match expected values") {
        auto results_h =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), results);

        std::size_t nwrong = 0;
        for (int i = 0; i < n_iterations; ++i) {
          double expected = 0.0;
          for (int j = 0; j < 100; ++j) {
            expected += static_cast<double>(i + j);
          }
          for (int j = 0; j < 50; ++j) {
            expected += static_cast<double>(i * j);
          }
          nwrong += !(std::abs(results_h(i) - expected) < 1e-10);
        }
        REQUIRE(nwrong == 0);
      }
    }
  }
}

SCENARIO("TokenScratchPool handles multi-dimensional views", "[TokenScratch][MultiDim]") {
  GIVEN("A TokenScratchPool with 128KB per token") {
    using ExecSpace = Kokkos::DefaultExecutionSpace;

    constexpr size_t scratch_bytes = 128 * 1024;
    constexpr int n_blocks = 50;
    constexpr int ni = 8, nj = 8, nk = 8;

    parthenon::TokenScratchPool<ExecSpace> pool(scratch_bytes);

    WHEN("We allocate 2D and 3D views in parallel") {
      Kokkos::View<double *> block_results("block_results", n_blocks);

      Kokkos::parallel_for(
          "test_multidim", n_blocks, KOKKOS_LAMBDA(const int b) {
            auto scratch = pool.acquire();

            auto work_2d = scratch.template allocate_view<double>(ni, nj);
            auto work_3d = scratch.template allocate_view<double>(ni, nj, nk);

            // Initialize and compute
            for (int i = 0; i < ni; ++i) {
              for (int j = 0; j < nj; ++j) {
                work_2d(i, j) = static_cast<double>(i + j);
                for (int k = 0; k < nk; ++k) {
                  work_3d(i, j, k) = work_2d(i, j) * k;
                }
              }
            }

            double sum = 0.0;
            for (int i = 0; i < ni; ++i) {
              for (int j = 0; j < nj; ++j) {
                for (int k = 0; k < nk; ++k) {
                  sum += work_3d(i, j, k);
                }
              }
            }
            block_results(b) = sum;
          });

      Kokkos::fence();

      THEN("All blocks produce the same correct result") {
        auto results_h =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), block_results);

        double expected = 0.0;
        for (int i = 0; i < ni; ++i) {
          for (int j = 0; j < nj; ++j) {
            for (int k = 0; k < nk; ++k) {
              expected += static_cast<double>(i + j) * k;
            }
          }
        }

        int nwrong = 0;
        for (int b = 0; b < n_blocks; ++b) {
          nwrong += !(std::abs(results_h(b) - expected) < 1e-10);
        }
        REQUIRE(nwrong == 0);
      }
    }
  }
}

SCENARIO("TokenScratchPool handles token reuse with many iterations",
         "[TokenScratch][Reuse]") {
  GIVEN("A TokenScratchPool with 8KB per token and many iterations") {
    using ExecSpace = Kokkos::DefaultExecutionSpace;
    using MemSpace = ExecSpace::memory_space;

    constexpr size_t scratch_bytes = 8 * 1024;
    constexpr int n_iterations = 10000;

    parthenon::TokenScratchPool<ExecSpace, MemSpace> pool(scratch_bytes);

    WHEN("We run many iterations to force token reuse") {
      Kokkos::View<int *> counters("counters", n_iterations);

      Kokkos::parallel_for(
          "test_reuse", n_iterations, KOKKOS_LAMBDA(const int i) {
            auto scratch = pool.acquire();
            auto data = scratch.template allocate_view<int>(100);

            for (int j = 0; j < 100; ++j) {
              data(j) = i + j;
            }

            int sum = 0;
            for (int j = 0; j < 100; ++j) {
              sum += data(j);
            }
            counters(i) = sum;
          });

      Kokkos::fence();

      THEN("All iterations produce correct results despite token reuse") {
        auto counters_h =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), counters);

        int nwrong = 0;
        for (int i = 0; i < n_iterations; ++i) {
          int expected = 0;
          for (int j = 0; j < 100; ++j) {
            expected += i + j;
          }
          nwrong += (counters_h(i) != expected);
        }
        REQUIRE(nwrong == 0);
      }
    }
  }
}

SCENARIO("TokenScratchPool supports 4D and higher views", "[TokenScratch][Variadic]") {
  GIVEN("A TokenScratchPool with sufficient memory") {
    constexpr size_t scratch_bytes = 512 * 1024;
    parthenon::TokenScratchPool<> pool(scratch_bytes);

    WHEN("We allocate a 4D view") {
      Kokkos::View<double *> result("result", 1);

      Kokkos::parallel_for(
          "test_4d", 1, KOKKOS_LAMBDA(const int i) {
            auto scratch = pool.acquire();

            // Allocate a 4D view: 5x4x3x2 = 120 elements
            auto view_4d = scratch.template allocate_view<double>(5, 4, 3, 2);

            // Initialize and sum
            double sum = 0.0;
            for (int i1 = 0; i1 < 5; ++i1) {
              for (int i2 = 0; i2 < 4; ++i2) {
                for (int i3 = 0; i3 < 3; ++i3) {
                  for (int i4 = 0; i4 < 2; ++i4) {
                    view_4d(i1, i2, i3, i4) = static_cast<double>(i1 + i2 + i3 + i4);
                    sum += view_4d(i1, i2, i3, i4);
                  }
                }
              }
            }
            result(0) = sum;
          });

      Kokkos::fence();

      THEN("The 4D view works correctly") {
        auto result_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result);

        // Compute expected sum
        double expected = 0.0;
        for (int i1 = 0; i1 < 5; ++i1) {
          for (int i2 = 0; i2 < 4; ++i2) {
            for (int i3 = 0; i3 < 3; ++i3) {
              for (int i4 = 0; i4 < 2; ++i4) {
                expected += static_cast<double>(i1 + i2 + i3 + i4);
              }
            }
          }
        }

        REQUIRE(std::abs(result_h(0) - expected) < 1e-10);
      }
    }

    WHEN("We allocate a 5D view") {
      Kokkos::View<double *> result("result", 1);

      Kokkos::parallel_for(
          "test_5d", 1, KOKKOS_LAMBDA(const int i) {
            auto scratch = pool.acquire();

            // Allocate a 5D view: 4x3x3x2x2 = 144 elements
            auto view_5d = scratch.template allocate_view<int>(4, 3, 3, 2, 2);

            // Initialize and sum
            int sum = 0;
            for (int i1 = 0; i1 < 4; ++i1) {
              for (int i2 = 0; i2 < 3; ++i2) {
                for (int i3 = 0; i3 < 3; ++i3) {
                  for (int i4 = 0; i4 < 2; ++i4) {
                    for (int i5 = 0; i5 < 2; ++i5) {
                      view_5d(i1, i2, i3, i4, i5) =
                          i1 * 10000 + i2 * 1000 + i3 * 100 + i4 * 10 + i5;
                      sum += view_5d(i1, i2, i3, i4, i5);
                    }
                  }
                }
              }
            }
            result(0) = static_cast<double>(sum);
          });

      Kokkos::fence();

      THEN("The 5D view works correctly") {
        auto result_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result);

        // Compute expected sum
        int expected = 0;
        for (int i1 = 0; i1 < 4; ++i1) {
          for (int i2 = 0; i2 < 3; ++i2) {
            for (int i3 = 0; i3 < 3; ++i3) {
              for (int i4 = 0; i4 < 2; ++i4) {
                for (int i5 = 0; i5 < 2; ++i5) {
                  expected += i1 * 10000 + i2 * 1000 + i3 * 100 + i4 * 10 + i5;
                }
              }
            }
          }
        }

        REQUIRE(std::abs(result_h(0) - static_cast<double>(expected)) < 1e-10);
      }
    }
  }
}

SCENARIO("Variadic allocate_view handles mixed dimensions in single kernel",
         "[TokenScratch][Variadic][Mixed]") {
  GIVEN("A TokenScratchPool with sufficient memory") {
    constexpr size_t scratch_bytes = 256 * 1024;
    parthenon::TokenScratchPool<> pool(scratch_bytes);

    WHEN("We allocate views of different ranks in the same kernel") {
      Kokkos::View<double *> results("results", 5);

      Kokkos::parallel_for(
          "test_mixed_ranks", 1, KOKKOS_LAMBDA(const int iter) {
            auto scratch = pool.acquire();

            // Allocate views of ranks 1-5
            auto view_1d = scratch.template allocate_view<double>(10);
            auto view_2d = scratch.template allocate_view<double>(5, 4);
            auto view_3d = scratch.template allocate_view<double>(3, 3, 3);
            auto view_4d = scratch.template allocate_view<double>(2, 2, 2, 2);
            auto view_5d = scratch.template allocate_view<double>(2, 2, 2, 2, 2);

            // Initialize each view
            for (int i = 0; i < 10; ++i)
              view_1d(i) = 1.0;

            for (int i = 0; i < 5; ++i)
              for (int j = 0; j < 4; ++j)
                view_2d(i, j) = 2.0;

            for (int i = 0; i < 3; ++i)
              for (int j = 0; j < 3; ++j)
                for (int k = 0; k < 3; ++k)
                  view_3d(i, j, k) = 3.0;

            for (int i = 0; i < 2; ++i)
              for (int j = 0; j < 2; ++j)
                for (int k = 0; k < 2; ++k)
                  for (int l = 0; l < 2; ++l)
                    view_4d(i, j, k, l) = 4.0;

            for (int i = 0; i < 2; ++i)
              for (int j = 0; j < 2; ++j)
                for (int k = 0; k < 2; ++k)
                  for (int l = 0; l < 2; ++l)
                    for (int m = 0; m < 2; ++m)
                      view_5d(i, j, k, l, m) = 5.0;

            // Compute sums
            double sum1 = 0.0, sum2 = 0.0, sum3 = 0.0, sum4 = 0.0, sum5 = 0.0;

            for (int i = 0; i < 10; ++i)
              sum1 += view_1d(i);

            for (int i = 0; i < 5; ++i)
              for (int j = 0; j < 4; ++j)
                sum2 += view_2d(i, j);

            for (int i = 0; i < 3; ++i)
              for (int j = 0; j < 3; ++j)
                for (int k = 0; k < 3; ++k)
                  sum3 += view_3d(i, j, k);

            for (int i = 0; i < 2; ++i)
              for (int j = 0; j < 2; ++j)
                for (int k = 0; k < 2; ++k)
                  for (int l = 0; l < 2; ++l)
                    sum4 += view_4d(i, j, k, l);

            for (int i = 0; i < 2; ++i)
              for (int j = 0; j < 2; ++j)
                for (int k = 0; k < 2; ++k)
                  for (int l = 0; l < 2; ++l)
                    for (int m = 0; m < 2; ++m)
                      sum5 += view_5d(i, j, k, l, m);

            results(0) = sum1;
            results(1) = sum2;
            results(2) = sum3;
            results(3) = sum4;
            results(4) = sum5;
          });

      Kokkos::fence();

      THEN("All views work correctly") {
        auto results_h =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), results);

        REQUIRE(std::abs(results_h(0) - 10.0) < 1e-10);  // 10 * 1.0
        REQUIRE(std::abs(results_h(1) - 40.0) < 1e-10);  // 20 * 2.0
        REQUIRE(std::abs(results_h(2) - 81.0) < 1e-10);  // 27 * 3.0
        REQUIRE(std::abs(results_h(3) - 64.0) < 1e-10);  // 16 * 4.0
        REQUIRE(std::abs(results_h(4) - 160.0) < 1e-10); // 32 * 5.0
      }
    }
  }
}
