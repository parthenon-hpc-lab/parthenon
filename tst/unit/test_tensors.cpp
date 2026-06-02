//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2025 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2025. Triad National Security, LLC. All rights reserved.
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
#include <vector>

#include <catch2/catch.hpp>

#include "kokkos_abstraction.hpp"
#include "tensors/tensors.hpp"

using namespace parthenon;
using namespace parthenon::tensors;

SCENARIO("Parthenon Tensor Cores", "[TensorCores]") {
  GIVEN("An object pool for a tensor core") {
    const std::size_t nc = 5;
    const std::size_t chunk_size = 30;
    pool_map_t pool_map;
    pool_map.AddPool(nc, chunk_size);

    THEN("We can allocate a tensor core") {
      const std::size_t rl = 2;
      const std::size_t rr = 3;
      { /* scoped */
        TensorCoreHost tc(pool_map, rl, nc, rr);
        AND_THEN("The pool provided the right number of buffers to represent the ranks") {
          REQUIRE(pool_map.GetPool(nc).NumBuffersInUse() == rl * rr);
        }
        AND_THEN("We can copy a tensor core object and reference counting works") {
          auto copy = tc;
          REQUIRE(pool_map.GetPool(nc).NumBuffersInUse() == rl * rr);
        }
        AND_THEN("We can access the core on device") {
          auto tc_d = tc.GetOnDevice();
          parthenon::par_for_outer(
              PARTHENON_AUTO_LABEL, 0, 0, 0, rl - 1, 0, rr - 1,
              KOKKOS_LAMBDA(parthenon::team_mbr_t mbr, const int il, const int ir) {
                parthenon::par_for_inner(
                    mbr, 0, nc - 1,
                    // KOKKOS_LAMBDA(const int ic) {
                    [&](const int ic) { tc_d(il, ic, ir) = 100 * il + 10 * ic + ir; });
              });
          Kokkos::fence();
          int nwrong = 0;
          parthenon::par_reduce(
              PARTHENON_AUTO_LABEL, 0, rl - 1, 0, nc - 1, 0, rr - 1,
              KOKKOS_LAMBDA(const int il, const int ic, const int ir, int &nw) {
                if (tc_d(il, ic, ir) != 100 * il + 10 * ic + ir) {
                  nw += 1;
                }
              },
              nwrong);
          REQUIRE(nwrong == 0);
        }
      }
      AND_THEN("When the tensor core goes out of scope, the ranks are freed") {
        REQUIRE(pool_map.GetPool(nc).NumBuffersInUse() == 0);
      }
      AND_THEN("The pool map contains the right number of total buffers") {
        REQUIRE(pool_map.GetPool(nc).NumBuffersInPool() == chunk_size);
      }
    }
  }
}

SCENARIO("Parthenon tensor trains", "[TensorTrains][Add]") {
  GIVEN("Six cores") {
    constexpr std::size_t NCORES_PER_TRAIN = 3;
    constexpr std::size_t NC[NCORES_PER_TRAIN] = {4, 5, 6};

    constexpr std::size_t NRANKS = NCORES_PER_TRAIN + 1;
    constexpr std::size_t RANKS[NRANKS] = {1, 2, 3, 1};

    const std::size_t chunk_size = 6;
    pool_map_t pool_map;
    for (int i = 0; i < NCORES_PER_TRAIN; ++i) {
      pool_map.AddPool(NC[i], chunk_size);
    }

    std::vector<TensorCoreHost> cores1, cores2;
    for (int i = 0; i < NCORES_PER_TRAIN; ++i) {
      cores1.push_back(TensorCoreHost(pool_map, RANKS[i], NC[i], RANKS[i + 1]));
      cores2.push_back(TensorCoreHost(pool_map, RANKS[i], NC[i], RANKS[i + 1]));
    }

    WHEN("We make two tensor trains") {
      TensorTrain A("Train A", cores1);
      TensorTrain B("Train B", cores2);
      A.SetOnes();
      B.SetOnes();

      THEN("We add them and create a new TT with the result 2*A + B") {
        TensorTrain C = aXPlusY(pool_map, 2, A, B);
        ParArrayND<Real> Cdense = C.ToDenseArray3D();
        AND_THEN("The resultant array has appropriate dense extents") {
          for (std::size_t i = 0; i < NCORES_PER_TRAIN; ++i) {
            REQUIRE(Cdense.GetDim(3 - i) == NC[i]);
          }
        }
        int nwrong = 0;
        par_reduce(
            "Check if its right", 0, Cdense.GetDim(3) - 1, 0, Cdense.GetDim(2) - 1, 0,
            Cdense.GetDim(1) - 1,
            KOKKOS_LAMBDA(const int k, const int j, const int i, int &nw) {
              if (Cdense(k, j, i) != 3 * 2 * 3) nw += 1;
            },
            nwrong);
        REQUIRE(nwrong == 0);
      }
    }
  }
}

SCENARIO("TensorTrain Gram-SVD rounding", "[TensorTrains][GramSVD]") {
  GIVEN("A small tensor train with nontrivial ranks") {
    constexpr std::size_t NCORES = 3;
    constexpr std::size_t NC[NCORES] = {7, 11, 13};

    constexpr std::size_t NRANKS = NCORES + 1;
    constexpr std::size_t RANKS[NRANKS] = {1, 4, 9, 1};

    const std::size_t chunk_size = 16;
    pool_map_t pool_map;
    for (int i = 0; i < NCORES; ++i) {
      pool_map.AddPool(NC[i], chunk_size);
    }

    std::vector<TensorCoreHost> cores;
    for (int i = 0; i < NCORES; ++i) {
      cores.emplace_back(pool_map, RANKS[i], NC[i], RANKS[i + 1]);
    }

    TensorTrain T("Rounded TT", cores);

    // create a device view of the TT with relevant metadata and pointer to device cores
    TensorTrainDeviceView ttd = T.GetDeviceView();
    auto cores_d = ttd.cores;

    // Fill with trivial data
    // T.SetOnes();

    // Fill with deterministic nontrivial data
    int core = 0;
    par_for(
        PARTHENON_AUTO_LABEL, 0, RANKS[core] - 1, 0, RANKS[core + 1] - 1, 0, NC[core] - 1,
        KOKKOS_LAMBDA(const int iL, const int iR, const int i) {
          cores_d[core](iL, i, iR) = std::sin((iR + 1) * (i + 1));
        });
    Kokkos::fence();

    core = 1;
    par_for(
        PARTHENON_AUTO_LABEL, 0, RANKS[core] - 1, 0, RANKS[core + 1] - 1, 0, NC[core] - 1,
        KOKKOS_LAMBDA(const int iL, const int iR, const int i) {
          cores_d[core](iL, i, iR) = std::pow(10., -iR) * std::cos((iL + 1) * (i + 1));
        });
    Kokkos::fence();

    core = 2;
    par_for(
        PARTHENON_AUTO_LABEL, 0, RANKS[core] - 1, 0, RANKS[core + 1] - 1, 0, NC[core] - 1,
        KOKKOS_LAMBDA(const int iL, const int iR, const int i) {
          cores_d[core](iL, i, iR) = std::sin((iL + 1) * (i + 1));
        });
    Kokkos::fence();

    // Dense reference BEFORE rounding
    ParArrayND<Real> dense_before = T.ToDenseArray3D();
    Kokkos::fence();

    // Save original ranks
    std::vector<std::size_t> ranks_before;
    for (int n = 0; n < T.GetNumCores(); ++n) {
      ranks_before.push_back(T.GetRightRank(n));
    }

    WHEN("We apply Gram-SVD rounding with a loose tolerance") {
      const Real eps = 1e-6;
      T.GramSVDRound(eps);
      Kokkos::fence();

      THEN("The dense representation is preserved within tolerance") {
        ParArrayND<Real> dense_after = T.ToDenseArray3D();
        Kokkos::fence();

        int nwrong = 0;
        par_reduce(
            "Check dense error after GramSVD round", 0, dense_after.GetDim(3) - 1, 0,
            dense_after.GetDim(2) - 1, 0, dense_after.GetDim(1) - 1,
            KOKKOS_LAMBDA(const int k, const int j, const int i, int &nw) {
              const Real diff = dense_after(k, j, i) - dense_before(k, j, i);
              if (std::abs(diff) > 1e-8) nw += 1;
              printf("Dense before/after rounding: %23.15e  %23.15e\n",
                     dense_before(k, j, i), dense_after(k, j, i));
            },
            nwrong);

        REQUIRE(nwrong == 0);
      }

      THEN("Tensor-train ranks do not increase") {
        for (int n = 0; n < T.GetNumCores(); ++n) {
          REQUIRE(T.GetRightRank(n) <= ranks_before[n]);
          printf("Rank %d before: %zu, after: %zu\n", n, ranks_before[n],
                 T.GetRightRank(n));
        }
      }
    }
  }
}

SCENARIO("TensorTrain Resizing", "[TensorTrains][Resize]") {
  GIVEN("A small tensor train with nontrivial ranks") {

    const std::size_t chunk_size = 16;
    pool_map_t pool_map;
    pool_map.AddPool(8, chunk_size);

    const int shape_before[3]{4, 8, 6};
    const int shape_after[3]{1, 8, 2};

    TensorCoreHost core_host(pool_map, shape_before[0], shape_before[1], shape_before[2]);
    TensorCoreDevice core_device = core_host.GetOnDevice();

    // Fill with deterministic nontrivial data
    Kokkos::parallel_for(
        "fill the core", Kokkos::RangePolicy<>(0, 1), KOKKOS_LAMBDA(const int) {
          // TensorCoreDevice core_device = core_host.GetOnDevice();
          for (int iL = 0; iL < shape_before[0]; iL++) {
            for (int iR = 0; iR < shape_before[2]; iR++) {
              for (int ic = 0; ic < shape_before[1]; ic++) {
                core_device(iL, ic, iR) = 100 * iL + 10 * iR + ic;
              }
            }
          }
        });
    Kokkos::fence();

    // set the new shape (needs to be done on device)
    // core_device.SetShape(shape_after[0], shape_after[1], shape_after[2]);
    parthenon::par_for(
        PARTHENON_AUTO_LABEL, 0, 0, KOKKOS_LAMBDA(const int) {
          core_device.SetShape(shape_after[0], shape_after[1], shape_after[2]);
        });

    Kokkos::fence();

    // now resize on host
    core_host.ResizeToNewShape();

    THEN("Extents match new shape") {
      REQUIRE(core_host.GetLeftRank() == shape_after[0]);
      REQUIRE(core_host.GetPhysicalIndexSize() == shape_after[1]);
      REQUIRE(core_host.GetRightRank() == shape_after[2]);
    }

    // check that data in kept block is preserved. Since we have to call REQUIRE
    // on host, this also checks that the host and device data matches.
    // In the resize, the data was already copied from device to host.

    THEN("Data in kept block is preserved") {
      int nwrong = 0;
      parthenon::par_reduce(
          PARTHENON_AUTO_LABEL, 0, shape_after[0] - 1, 0, shape_after[2] - 1, 0,
          shape_after[1] - 1,
          KOKKOS_LAMBDA(const int il, const int ir, const int ic, int &nw) {
            if (core_host(il, ic, ir) != 100 * il + 10 * ir + ic) {
              nw += 1;
            }
          },
          nwrong);
      REQUIRE(nwrong == 0);
    }
  }
}
