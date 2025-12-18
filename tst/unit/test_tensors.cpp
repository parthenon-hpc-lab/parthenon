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

#include "tensors/tensors.hpp"

using namespace parthenon::tensors;

TEST_CASE("Parthenon Tensors", "[TensorCores]") {

  auto allocation_strategy = [nc, chunk_size](pool_t *p) {
    const auto tot_size = nbuf * buf_size;
    pool_t::base_t chunk("pool buffer", tot_size);
    for (int i = 1; i < chunk_size; ++i) {
      pool->AddFreeObjectToPool(
          pool_t::base_t(chunk, std::make_pair(i * nc, (i + 1) * nc)));
    }
    return pool_t::base_t(chunk, std::make_pair(0, buf_size));
  };

  size_t nc = 5;
  size_t chunk_size = 30;

  pool_map_t object_pool;
  object_pool.emplace(nc, pool_t(allocation_strategy));

  TensorCore tc();
}
