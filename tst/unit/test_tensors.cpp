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

SCENARIO("Parthenon Tensors", "[TensorCores]") {
  GIVEN("An object pool for a tensor core") {
    const std::size_t nc = 5;
    const std::size_t chunk_size = 30;
    pool_map_t pool_map;
    pool_map.AddPool(nc, chunk_size);

    THEN("We can allocate a tensor core") {
      const std::size_t rl = 2;
      const std::size_t rr = 3;
      { /* scoped */
        TensorCore tc(pool_map, rl, nc, rr);
        AND_THEN("The pool provided the right number of buffers to represent the ranks") {
          REQUIRE(pool_map.GetPool(nc).NumBuffersInUse() == rl * rr);
        }
        AND_THEN("We can access the core on device") {
          parthenon::par_for_outer(
              PARTHENON_AUTO_LABEL, 0, 0, 0, rl - 1, 0, rr - 1,
              KOKKOS_LAMBDA(parthenon::team_mbr_t mbr, const int il, const int ir) {
                parthenon::par_for_inner(
                    mbr, 0, nc - 1, KOKKOS_LAMBDA(const int ic) {
                      tc(il, ic, ir) = 100 * il + 10 * ic + ir;
                    });
              });
          Kokkos::fence();
          int nwrong = 0;
          parthenon::par_reduce(
              PARTHENON_AUTO_LABEL, 0, rl - 1, 0, nc - 1, 0, rr - 1,
              KOKKOS_LAMBDA(const int il, const int ic, const int ir, int &nw) {
                if (tc(il, ic, ir) != 100 * il + 10 * ic + ir) {
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
