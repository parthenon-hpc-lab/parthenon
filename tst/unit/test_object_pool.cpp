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

#include "basic_types.hpp"
#include "defs.hpp"
#include "kokkos_abstraction.hpp"
#include "parthenon_arrays.hpp"
#include "utils/object_pool.hpp"

using parthenon::Real;
using array_t = parthenon::ParArray1DRaw<Real>;
using pool_t = parthenon::ObjectPool<array_t>;
using pool_map_t = parthenon::ObjectPoolMap<array_t>;

SCENARIO("Object pools", "[ObjectPool]") {
  GIVEN("An object pool map containing several sizes") {
    const std::size_t NSIZES = 2;
    const std::size_t sizes[NSIZES] = {3, 5};
    const std::size_t chunk_sizes[NSIZES] = {2, 3};
    pool_map_t pool_map;
    for (std::size_t i = 0; i < NSIZES; ++i) {
      pool_map.AddPool(sizes[i], chunk_sizes[i]);
    }
    THEN("Pools for non-selected shapes are unavailable") {
      const std::size_t NOTREAL = 987654321;
      REQUIRE(!pool_map.Contains(NOTREAL));
      REQUIRE_THROWS(pool_map.GetPool(NOTREAL));
    }
    THEN("Each pool contains chunk_size arrays of the appropriate shape") {
      for (std::size_t i = 0; i < NSIZES; ++i) {
        REQUIRE(pool_map.Contains(sizes[i]));
      }
      AND_WHEN("We request a buffer") {
        auto &pool = pool_map.GetPool(sizes[0]);
        { /* Scoping */
          pool_t::owner_t buf_host = pool_map.GetOwningBuffer(sizes[0]);
          THEN("The pool contains nchunks buffers, of which one is in use") {
            REQUIRE(pool.NumBuffersInPool() == chunk_sizes[0]);
            REQUIRE(pool.NumBuffersInUse() == 1);
          }
        }
        THEN("After the destructor is called, the buffer is no longer in use") {
          REQUIRE(pool.NumBuffersInPool() == chunk_sizes[0]);
          REQUIRE(pool.NumBuffersInUse() == 0);
        }
      }
      AND_WHEN("We add buffers to the pool manually") {
        pool_map.AddFreeObjectsToPool(sizes[1], chunk_sizes[1]);
        THEN("The pool contains that many free objects") {
          auto &pool = pool_map.GetPool(sizes[1]);
          REQUIRE(pool.NumBuffersInPool() == chunk_sizes[1]);
          REQUIRE(pool.NumBuffersInUse() == 0);
        }
      }
    }
  }
}
