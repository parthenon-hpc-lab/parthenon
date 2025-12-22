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

#include <catch2/catch.hpp>

#include "kokkos_abstraction.hpp"
#include "tensors/tensors.hpp"

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
                    mbr, 0, nc - 1, KOKKOS_LAMBDA(const int ic) {
                      tc_d(il, ic, ir) = 100 * il + 10 * ic + ir;
                    });
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

SCENARIO("Parthenon tensor trains", "[TensorTrains]") {
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

    std::cout << "pools created" << std::endl;

    std::vector<TensorCoreHost> cores1, cores2;
    for (int i = 0; i < NCORES_PER_TRAIN; ++i) {
      cores1.push_back(TensorCoreHost(pool_map, RANKS[i], NC[i], RANKS[i + 1]));
      cores2.push_back(TensorCoreHost(pool_map, RANKS[i], NC[i], RANKS[i + 1]));
    }

    std::cout << "vectors created" << std::endl;

    WHEN("We make two tensor trains") {
      TensorTrain A("Train A", cores1);
      TensorTrain B("Train B", cores2);
      REQUIRE(true);

      std::cout << "trains achieved" << std::endl;
    }
  }
}
