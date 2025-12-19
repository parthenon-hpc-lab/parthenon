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

  const std::size_t nc = 5;
  const std::size_t chunk_size = 30;
  pool_map_t object_pool;
  object_pool.AddPool(nc, chunk_size);


  const std::size_t rl = 1;
  const std::size_t rr = 2;
  TensorCore tc(object_pool, 1, nc, rr);
}
