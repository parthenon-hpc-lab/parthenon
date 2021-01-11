//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2021 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2021. Triad National Security, LLC. All rights reserved.
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
  Metadata m;
  auto swarm = std::make_shared<Swarm>("test swarm", m, NUMINIT);
  auto swarm_d = swarm->GetDeviceContext();
  swarm->SetBlockPointer(meshblock);
  REQUIRE(swarm->get_num_active() == 0);
  REQUIRE(swarm->get_max_active_index() == 0);
  ParArrayND<int> failures_d("Number of failures", 1);
  meshblock->par_for(
      "Reset", 0, 0, KOKKOS_LAMBDA(const int n) { failures_d(n) = 0; });
  meshblock->par_for(
      "Check mask", 0, NUMINIT - 1, KOKKOS_LAMBDA(const int n) {
        if (swarm_d.IsActive(n) == true) {
          Kokkos::atomic_add(&failures_d(0), 1);
        }
      });
  auto failures_h = failures_d.GetHostMirrorAndCopy();
  REQUIRE(failures_h(0) == 0);

  REQUIRE(swarm->label() == "test swarm");
  REQUIRE(swarm->info().length() == 0);
  REQUIRE(swarm->metadata() == m);

  // Add multiple variables
  std::vector<std::string> labelVector(2);
  labelVector[0] = "i";
  labelVector[1] = "j";
  Metadata m_integer({Metadata::Integer});
  swarm->Add(labelVector, m_integer);

  auto new_mask = swarm->AddEmptyParticles(1);
  swarm_d = swarm->GetDeviceContext();
  auto x_d = swarm->GetReal("x").Get();
  auto x_h = x_d.GetHostMirrorAndCopy();
  auto i_d = swarm->GetInteger("i").Get();
  auto i_h = i_d.GetHostMirrorAndCopy();

  x_h(0) = 0.5;
  i_h(1) = 2;

  x_d.DeepCopy(x_h);
  i_d.DeepCopy(i_h);

  new_mask = swarm->AddEmptyParticles(11);
  swarm_d = swarm->GetDeviceContext();
  x_d = swarm->GetReal("x").Get();
  i_d = swarm->GetInteger("i").Get();
  x_h = x_d.GetHostMirrorAndCopy();
  i_h = i_d.GetHostMirrorAndCopy();
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
  failures_h = failures_d.GetHostMirrorAndCopy();
  REQUIRE(failures_h(0) == 0);
  // Check that existing data was successfully copied during pool resize
  x_h = swarm->GetReal("x").Get().GetHostMirrorAndCopy();
  REQUIRE(x_h(0) == 0.5);

  // Remove particles 3 and 5
  meshblock->par_for(
      "Remove particles", 0, 0, KOKKOS_LAMBDA(const int n) {
        swarm_d.MarkParticleForRemoval(2);
        swarm_d.MarkParticleForRemoval(4);
      });
  swarm->RemoveMarkedParticles();

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
  failures_h = failures_d.GetHostMirrorAndCopy();
  REQUIRE(failures_h(0) == 0);

  // Enter some data to be moved during defragment
  x_h = swarm->GetReal("x").Get().GetHostMirrorAndCopy();
  x_h(10) = 1.1;
  x_h(11) = 1.2;
  x_d.DeepCopy(x_h);

  // Defragment the list
  swarm->Defrag();
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
  failures_h = failures_d.GetHostMirrorAndCopy();
  REQUIRE(failures_h(0) == 0);

  // Check that data was moved during defrag
  x_h = swarm->GetReal("x").Get().GetHostMirrorAndCopy();
  REQUIRE(x_h(2) == 1.2);
  REQUIRE(x_h(4) == 1.1);
  i_h = swarm->GetInteger("i").Get().GetHostMirrorAndCopy();
  REQUIRE(i_h(1) == 2);
}
