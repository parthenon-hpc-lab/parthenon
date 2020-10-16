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

#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

using Real = double;
using parthenon::MeshBlock;
using parthenon::Metadata;
using parthenon::ParArrayND;
using parthenon::Swarm;

constexpr int NUMINIT = 10;

TEST_CASE("Swarm memory management", "[Swarm]") {
  auto meshblock = std::make_shared<MeshBlock>(1, 1);
  printf("%s:%i\n", __FILE__, __LINE__);
  Metadata m;
  printf("%s:%i\n", __FILE__, __LINE__);
  auto swarm = std::make_shared<Swarm>("test swarm", m, NUMINIT);
  printf("%s:%i\n", __FILE__, __LINE__);
  auto swarm_d = swarm->GetDeviceContext();
  printf("%s:%i\n", __FILE__, __LINE__);
  swarm->SetBlockPointer(meshblock);
  printf("%s:%i\n", __FILE__, __LINE__);
  REQUIRE(swarm->get_num_active() == 0);
  printf("%s:%i\n", __FILE__, __LINE__);
  REQUIRE(swarm->get_max_active_index() == 0);
  printf("%s:%i\n", __FILE__, __LINE__);
  ParArrayND<int> failures_d("Number of failures", 1);
  printf("%s:%i\n", __FILE__, __LINE__);
  meshblock->par_for(
      "Reset", 0, 0, KOKKOS_LAMBDA(const int n) { failures_d(n) = 0; });
  meshblock->par_for(
      "Check mask", 0, NUMINIT - 1, KOKKOS_LAMBDA(const int n) {
        if (swarm_d.IsActive(n) == true) {
          Kokkos::atomic_add(&failures_d(0), 1);
        }
      });
  printf("%s:%i\n", __FILE__, __LINE__);
  auto failures_h = failures_d.GetHostMirrorAndCopy();
  printf("%s:%i\n", __FILE__, __LINE__);
  REQUIRE(failures_h(0) == 0);
  printf("%s:%i\n", __FILE__, __LINE__);

  REQUIRE(swarm->label() == "test swarm");
  printf("%s:%i\n", __FILE__, __LINE__);
  REQUIRE(swarm->info().length() == 0);
  printf("%s:%i\n", __FILE__, __LINE__);
  REQUIRE(swarm->metadata() == m);
  printf("%s:%i\n", __FILE__, __LINE__);

  // Add multiple variables
  std::vector<std::string> labelVector(2);
  printf("%s:%i\n", __FILE__, __LINE__);
  labelVector[0] = "i";
  printf("%s:%i\n", __FILE__, __LINE__);
  labelVector[1] = "j";
  printf("%s:%i\n", __FILE__, __LINE__);
  Metadata m_integer({Metadata::Integer});
  printf("%s:%i\n", __FILE__, __LINE__);
  swarm->Add(labelVector, m_integer);
  printf("%s:%i\n", __FILE__, __LINE__);

  auto new_mask = swarm->AddEmptyParticles(1);
  printf("%s:%i\n", __FILE__, __LINE__);
  swarm_d = swarm->GetDeviceContext();
  printf("%s:%i\n", __FILE__, __LINE__);
  auto x_d = swarm->GetReal("x").Get();
  printf("%s:%i\n", __FILE__, __LINE__);
  auto x_h = x_d.GetHostMirrorAndCopy();
  printf("%s:%i\n", __FILE__, __LINE__);
  auto i_d = swarm->GetInteger("i").Get();
  printf("%s:%i\n", __FILE__, __LINE__);
  auto i_h = i_d.GetHostMirrorAndCopy();
  printf("%s:%i\n", __FILE__, __LINE__);

  x_h(0) = 0.5;
  printf("%s:%i\n", __FILE__, __LINE__);
  i_h(1) = 2;
  printf("%s:%i\n", __FILE__, __LINE__);

  x_d.DeepCopy(x_h);
  printf("%s:%i\n", __FILE__, __LINE__);
  i_d.DeepCopy(i_h);
  printf("%s:%i\n", __FILE__, __LINE__);

  new_mask = swarm->AddEmptyParticles(11);
  printf("%s:%i\n", __FILE__, __LINE__);
  swarm_d = swarm->GetDeviceContext();
  printf("%s:%i\n", __FILE__, __LINE__);
  x_d = swarm->GetReal("x").Get();
  printf("%s:%i\n", __FILE__, __LINE__);
  i_d = swarm->GetInteger("i").Get();
  printf("%s:%i\n", __FILE__, __LINE__);
  x_h = x_d.GetHostMirrorAndCopy();
  printf("%s:%i\n", __FILE__, __LINE__);
  i_h = i_d.GetHostMirrorAndCopy();
  printf("%s:%i\n", __FILE__, __LINE__);
  meshblock->par_for(
      "Check mask", 0, 2 * NUMINIT - 1, KOKKOS_LAMBDA(const int n) {
        if (n < 12) {
          if (swarm_d.IsActive(n) == false) {
            Kokkos::atomic_add(&failures_d(0), 1);
          }
        } else {
          if (swarm_d.IsActive(n) == true) {
            Kokkos::atomic_add(&failures_d(0), 1);
          }
        }
      });
  printf("%s:%i\n", __FILE__, __LINE__);
  failures_h = failures_d.GetHostMirrorAndCopy();
  printf("%s:%i\n", __FILE__, __LINE__);
  REQUIRE(failures_h(0) == 0);
  // Check that existing data was successfully copied during pool resize
  x_h = swarm->GetReal("x").Get().GetHostMirrorAndCopy();
  printf("%s:%i\n", __FILE__, __LINE__);
  REQUIRE(x_h(0) == 0.5);
  printf("%s:%i\n", __FILE__, __LINE__);

  // Remove particles 3 and 5
  meshblock->par_for(
      "Remove particles", 0, 0, KOKKOS_LAMBDA(const int n) {
        swarm_d.MarkParticleForRemoval(2);
        swarm_d.MarkParticleForRemoval(4);
      });
  printf("%s:%i\n", __FILE__, __LINE__);
  swarm->RemoveMarkedParticles();
  printf("%s:%i\n", __FILE__, __LINE__);

  // Check that partiles 3 and 5 were removed
  meshblock->par_for(
      "Check mask", 0, 2 * NUMINIT - 1, KOKKOS_LAMBDA(const int n) {
        if (n < 12 && n != 2 && n != 4) {
          if (swarm_d.IsActive(n) == false) {
            Kokkos::atomic_add(&failures_d(0), 1);
          }
        } else {
          if (swarm_d.IsActive(n) == true) {
            Kokkos::atomic_add(&failures_d(0), 1);
          }
        }
      });
  printf("%s:%i\n", __FILE__, __LINE__);
  failures_h = failures_d.GetHostMirrorAndCopy();
  printf("%s:%i\n", __FILE__, __LINE__);
  REQUIRE(failures_h(0) == 0);
  printf("%s:%i\n", __FILE__, __LINE__);

  // Enter some data to be moved during defragment
  x_h = swarm->GetReal("x").Get().GetHostMirrorAndCopy();
  printf("%s:%i\n", __FILE__, __LINE__);
  x_h(10) = 1.1;
  printf("%s:%i\n", __FILE__, __LINE__);
  x_h(11) = 1.2;
  printf("%s:%i\n", __FILE__, __LINE__);
  x_d.DeepCopy(x_h);
  printf("%s:%i\n", __FILE__, __LINE__);

  // Defragment the list
  swarm->Defrag();
  printf("%s:%i\n", __FILE__, __LINE__);
  // Check that the list is defragmented
  meshblock->par_for(
      "Check mask", 0, 2 * NUMINIT - 1, KOKKOS_LAMBDA(const int n) {
        if (n < 10) {
          if (swarm_d.IsActive(n) == false) {
            Kokkos::atomic_add(&failures_d(0), 1);
          }
        } else {
          if (swarm_d.IsActive(n) == true) {
            Kokkos::atomic_add(&failures_d(0), 1);
          }
        }
      });
  printf("%s:%i\n", __FILE__, __LINE__);
  failures_h = failures_d.GetHostMirrorAndCopy();
  printf("%s:%i\n", __FILE__, __LINE__);
  REQUIRE(failures_h(0) == 0);
  printf("%s:%i\n", __FILE__, __LINE__);

  // Check that data was moved during defrag
  x_h = swarm->GetReal("x").Get().GetHostMirrorAndCopy();
  printf("%s:%i\n", __FILE__, __LINE__);
  REQUIRE(x_h(2) == 1.2);
  printf("%s:%i\n", __FILE__, __LINE__);
  REQUIRE(x_h(4) == 1.1);
  printf("%s:%i\n", __FILE__, __LINE__);
  i_h = swarm->GetInteger("i").Get().GetHostMirrorAndCopy();
  printf("%s:%i\n", __FILE__, __LINE__);
  REQUIRE(i_h(1) == 2);
  printf("%s:%i\n", __FILE__, __LINE__);
}
