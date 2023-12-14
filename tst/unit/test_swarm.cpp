//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2023. Triad National Security, LLC. All rights reserved.
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

#include "bvals/bvals_interfaces.hpp"
#include "interface/swarm.hpp"
#include "mesh/mesh.hpp"

#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

using Real = double;
using parthenon::ApplicationInput;
using parthenon::BoundaryFlag;
using parthenon::DeviceAllocate;
using parthenon::DeviceDeleter;
using parthenon::Mesh;
using parthenon::MeshBlock;
using parthenon::Metadata;
using parthenon::Packages_t;
using parthenon::ParameterInput;
using parthenon::ParArrayND;
using parthenon::ParticleBound;
using parthenon::Swarm;
using parthenon::SwarmDeviceContext;
using std::endl;

constexpr int NUMINIT = 10;

class ParticleBoundIX1User : public ParticleBound {
 public:
  KOKKOS_INLINE_FUNCTION void Apply(const int n, double &x, double &y, double &z,
                                    const SwarmDeviceContext &swarm_d) const override {
    if (x < swarm_d.x_min_global_) {
      swarm_d.MarkParticleForRemoval(n);
    }
  }
};

TEST_CASE("Swarm memory management", "[Swarm]") {
  std::stringstream is;
  is << "<parthenon/mesh>" << endl;
  is << "x1min = -0.5" << endl;
  is << "x2min = -0.5" << endl;
  is << "x3min = -0.5" << endl;
  is << "x1max = 0.5" << endl;
  is << "x2max = 0.5" << endl;
  is << "x3max = 0.5" << endl;
  is << "nx1 = 4" << endl;
  is << "nx2 = 4" << endl;
  is << "nx3 = 4" << endl;
  is << "swarm_ix1_bc = user" << endl;
  is << "swarm_ox1_bc = outflow" << endl;
  is << "swarm_ix2_bc = outflow" << endl;
  is << "swarm_ox2_bc = outflow" << endl;
  is << "swarm_ix3_bc = outflow" << endl;
  is << "swarm_ox3_bc = outflow" << endl;
  auto pin = std::make_shared<ParameterInput>();
  pin->LoadFromStream(is);
  auto app_in = std::make_shared<ApplicationInput>();
  app_in->RegisterSwarmBoundaryCondition<ParticleBoundIX1User>(
      parthenon::BoundaryFace::inner_x1);
  Packages_t packages;
  auto meshblock = std::make_shared<MeshBlock>(1, 1);
  auto mesh = std::make_shared<Mesh>(pin.get(), app_in.get(), packages, 1);
  meshblock->pmy_mesh = mesh.get();
  Metadata m;
  auto swarm = std::make_shared<Swarm>("test swarm", m, NUMINIT);
  swarm->SetBlockPointer(meshblock);
  swarm->AllocateBoundaries();
  auto swarm_d = swarm->GetDeviceContext();
  REQUIRE(swarm->GetNumActive() == 0);
  REQUIRE(swarm->GetMaxActiveIndex() == 0);
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
  Metadata m_integer({Metadata::Integer, Metadata::Particle});
  swarm->Add(labelVector, m_integer);

  ParArrayND<int> new_indices;
  auto new_mask = swarm->AddEmptyParticles(1, new_indices);
  swarm_d = swarm->GetDeviceContext();
  auto x_d = swarm->Get<Real>("x").Get();
  auto x_h = x_d.GetHostMirrorAndCopy();
  auto i_d = swarm->Get<int>("i").Get();
  auto i_h = i_d.GetHostMirrorAndCopy();

  x_h(0) = 0.5;
  i_h(1) = 2;

  x_d.DeepCopy(x_h);
  i_d.DeepCopy(i_h);

  new_mask = swarm->AddEmptyParticles(11, new_indices);
  swarm_d = swarm->GetDeviceContext();
  x_d = swarm->Get<Real>("x").Get();
  i_d = swarm->Get<int>("i").Get();
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
  x_h = swarm->Get<Real>("x").Get().GetHostMirrorAndCopy();
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
  x_h = swarm->Get<Real>("x").Get().GetHostMirrorAndCopy();
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
  x_h = swarm->Get<Real>("x").Get().GetHostMirrorAndCopy();
  REQUIRE(x_h(2) == 1.2);
  REQUIRE(x_h(4) == 1.1);
  i_h = swarm->Get<int>("i").Get().GetHostMirrorAndCopy();
  REQUIRE(i_h(1) == 2);

  // "Transport" a particle across the IX1 (custom) boundary
  ParArrayND<int> bc_indices("Boundary indices", 1);
  meshblock->par_for(
      "Transport", 0, 0, KOKKOS_LAMBDA(const int n) {
        x_d(0) = -0.6;
        bc_indices(0) = 0;
      });
  swarm->ApplyBoundaries_(1, bc_indices);
  swarm->RemoveMarkedParticles();

  // Check that particle that crossed boundary has been removed
  meshblock->par_for(
      "Check boundary", 0, 0, KOKKOS_LAMBDA(const int n) {
        if (swarm_d.IsActive(0)) {
          Kokkos::atomic_add(&failures_d(0), 1);
        }
      });
  failures_h = failures_d.GetHostMirrorAndCopy();
  REQUIRE(failures_h(0) == 0);
}
