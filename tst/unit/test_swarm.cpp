//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020 The Parthenon collaboration
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

#include <cmath>
#include <iostream>
#include <random>
#include <string>

#include <catch2/catch.hpp>

#include "interface/swarm.hpp"
#include "mesh/mesh.hpp"

using Real = double;
using parthenon::MeshBlock;
using parthenon::Metadata;
using parthenon::Swarm;

constexpr int NUMINIT = 10;

TEST_CASE("Swarm memory management", "[Swarm]") {
  auto meshblock = std::make_shared<MeshBlock>(1, 1);
  Metadata m;
  Swarm swarm("test swarm", m, NUMINIT);
  swarm.SetBlockPointer(meshblock);
  REQUIRE(swarm.get_num_active() == 0);
  REQUIRE(swarm.get_max_active_index() == 0);
  auto mask = swarm.GetMask().Get().GetHostMirrorAndCopy();
  REQUIRE(mask.GetDim(1) == NUMINIT);
  for (int n = 0; n < NUMINIT; n++) {
    REQUIRE(mask(n) == false);
  }

  REQUIRE(swarm.label() == "test swarm");
  REQUIRE(swarm.info().length() == 0);
  REQUIRE(swarm.metadata() == m);

  // Add multiple variables
  std::vector<std::string> labelVector(2);
  labelVector[0] = "i";
  labelVector[1] = "j";
  Metadata m_integer({Metadata::Integer});
  swarm.Add(labelVector, m_integer);

  auto x_d = swarm.GetReal("x").Get();
  auto x_h = x_d.GetHostMirrorAndCopy();
  auto i_d = swarm.GetInteger("i").Get();
  auto i_h = i_d.GetHostMirrorAndCopy();

  auto new_mask = swarm.AddEmptyParticles(1);
  x_h(0) = 0.5;
  i_h(1) = 2;

  x_d.DeepCopy(x_h);
  i_d.DeepCopy(i_h);

  new_mask = swarm.AddEmptyParticles(11);
  x_d = swarm.GetReal("x").Get();
  i_d = swarm.GetInteger("i").Get();
  mask = swarm.GetMask().Get().GetHostMirrorAndCopy();
  // Check that swarm pool doubled in size
  REQUIRE(mask.GetDim(1) == 2 * NUMINIT);
  // Check that only the added particles have a true mask
  for (int n = 0; n < 2 * NUMINIT; n++) {
    if (n < 12) {
      REQUIRE(mask(n) == true);
    } else {
      REQUIRE(mask(n) == false);
    }
  }
  // Check that existing data was successfully copied during pool resize
  x_h = swarm.GetReal("x").Get().GetHostMirrorAndCopy();
  REQUIRE(x_h(0) == 0.5);

  // Remove particles 3 and 5
  swarm.MarkParticleForRemoval(2);
  swarm.MarkParticleForRemoval(4);
  swarm.RemoveMarkedParticles();

  // Check that partiles 3 and 5 were removed
  mask = swarm.GetMask().Get().GetHostMirrorAndCopy();
  // Check that only the added particles have a true mask
  for (int n = 0; n < 2 * NUMINIT; n++) {
    if (n < 12 && n != 2 && n != 4) {
      REQUIRE(mask(n) == true);
    } else {
      REQUIRE(mask(n) == false);
    }
  }

  // Enter some data to be moved during defragment
  x_h = swarm.GetReal("x").Get().GetHostMirrorAndCopy();
  x_h(10) = 1.1;
  x_h(11) = 1.2;
  x_d.DeepCopy(x_h);

  // Defragment the list
  swarm.Defrag();
  mask = swarm.GetMask().Get().GetHostMirrorAndCopy();
  // Check that the list is defragmented
  for (int n = 0; n < 2 * NUMINIT; n++) {
    if (n < 10) {
      REQUIRE(mask(n) == true);
    } else {
      REQUIRE(mask(n) == false);
    }
  }

  // Check that data was moved during defrag
  x_h = swarm.GetReal("x").Get().GetHostMirrorAndCopy();
  REQUIRE(x_h(2) == 1.2);
  REQUIRE(x_h(4) == 1.1);
  i_h = swarm.GetInteger("i").Get().GetHostMirrorAndCopy();
  REQUIRE(i_h(1) == 2);
}
