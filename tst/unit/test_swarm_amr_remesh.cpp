//========================================================================================
// (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001 for Los
// Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
// for the U.S. Department of Energy/National Nuclear Security Administration. All rights
// in the program are reserved by Triad National Security, LLC, and the U.S. Department
// of Energy/National Nuclear Security Administration. The Government is granted for
// itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do so.
//========================================================================================

// This file was made in part with generative AI.

#include <algorithm>
#include <array>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <catch2/catch.hpp>

#include "application_input.hpp"
#include "interface/packages.hpp"
#include "interface/state_descriptor.hpp"
#include "interface/swarm.hpp"
#include "mesh/mesh.hpp"
#include "mesh/swarm_amr_remesh.hpp"
#include "pack/default_names.hpp"

namespace {

using parthenon::ApplicationInput;
using parthenon::LogicalLocation;
using parthenon::Mesh;
using parthenon::Metadata;
using parthenon::Packages_t;
using parthenon::ParameterInput;
using parthenon::Real;
using parthenon::SP_Swarm;
using parthenon::StateDescriptor;
using parthenon::SwarmRemeshContext;

constexpr int kPoolSize = 4;
constexpr char kSwarmName[] = "particles";

//----------------------------------------------------------------------------------------
// Build the smallest input deck needed to construct either a one-block coarse mesh or a
// four-block refined mesh. This is a direct remesh unit test, so static refinement is
// only used here as a cheap way to synthesize the source/target AMR topologies.
std::shared_ptr<ParameterInput> MakePin(const bool refined) {
  std::stringstream is;
  is << "<parthenon/mesh>\n";
  is << "refinement = static\n";
  is << "numlevel = 1\n";
  is << "nx1 = 4\n";
  is << "x1min = 0.0\n";
  is << "x1max = 1.0\n";
  is << "nx2 = 4\n";
  is << "x2min = 0.0\n";
  is << "x2max = 1.0\n";
  is << "nx3 = 1\n";
  is << "x3min = -0.5\n";
  is << "x3max = 0.5\n";
  is << "ix1_bc = outflow\n";
  is << "ox1_bc = outflow\n";
  is << "ix2_bc = outflow\n";
  is << "ox2_bc = outflow\n";
  is << "ix3_bc = periodic\n";
  is << "ox3_bc = periodic\n";
  is << "<parthenon/meshblock>\n";
  is << "nx1 = 4\n";
  is << "nx2 = 4\n";
  is << "nx3 = 1\n";
  if (refined) {
    is << "<parthenon/static_refinement0>\n";
    is << "level = 1\n";
    is << "x1min = 0.0\n";
    is << "x1max = 1.0\n";
    is << "x2min = 0.0\n";
    is << "x2max = 1.0\n";
    is << "x3min = -0.5\n";
    is << "x3max = 0.5\n";
  }
  auto pin = std::make_shared<ParameterInput>();
  pin->LoadFromStream(is);
  return pin;
}

//----------------------------------------------------------------------------------------
// Register one swarm and no fields. The remesh routine only needs the resolved swarm
// schema plus MeshBlock-local swarm storage.
Packages_t MakePackages() {
  Packages_t packages;
  auto pkg = std::make_shared<StateDescriptor>("swarm_amr_remesh_test");
  Metadata m({Metadata::None});
  m.SetInitialSwarmPoolReservation(kPoolSize);
  pkg->AddSwarm(kSwarmName, m);
  packages.Add(pkg);
  return packages;
}

//----------------------------------------------------------------------------------------
// Construct either the coarse one-block mesh or the uniformly refined four-daughter mesh.
std::shared_ptr<Mesh> MakeMesh(const bool refined, ApplicationInput *app_in,
                               Packages_t &packages) {
  auto pin = MakePin(refined);
  return std::make_shared<Mesh>(pin.get(), app_in, packages, 0);
}

//----------------------------------------------------------------------------------------
// Fetch the single test swarm from one MeshBlock.
SP_Swarm GetSwarm(const std::shared_ptr<parthenon::MeshBlock> &pmb) {
  return pmb->meshblock_data.Get()->GetSwarmData()->Get(kSwarmName);
}

//----------------------------------------------------------------------------------------
// Insert particles by writing positions directly into the newly allocated swarm slots.
void AddParticles(const SP_Swarm &swarm,
                  const std::vector<std::array<Real, 2>> &positions) {
  REQUIRE(swarm->GetNumActive() == 0);
  swarm->AddEmptyParticles(positions.size());
  auto x1 = swarm->Get<Real>(swarm_position::x1::name()).Get();
  auto x2 = swarm->Get<Real>(swarm_position::x2::name()).Get();
  auto x3 = swarm->Get<Real>(swarm_position::x3::name()).Get();
  auto x1_h = x1.GetHostMirrorAndCopy();
  auto x2_h = x2.GetHostMirrorAndCopy();
  auto x3_h = x3.GetHostMirrorAndCopy();
  for (int n = 0; n < positions.size(); ++n) {
    x1_h(n) = positions[n][0];
    x2_h(n) = positions[n][1];
    x3_h(n) = 0.0;
  }
  x1.DeepCopy(x1_h);
  x2.DeepCopy(x2_h);
  x3.DeepCopy(x3_h);
}

//----------------------------------------------------------------------------------------
// Gather active particle positions back to host and sort them so tests can compare
// against a deterministic reference ordering.
std::vector<std::array<Real, 2>> GetParticles(const SP_Swarm &swarm) {
  auto mask_h = swarm->GetMask().GetHostMirrorAndCopy();
  auto x1_h = swarm->Get<Real>(swarm_position::x1::name()).Get().GetHostMirrorAndCopy();
  auto x2_h = swarm->Get<Real>(swarm_position::x2::name()).Get().GetHostMirrorAndCopy();
  std::vector<std::array<Real, 2>> particles;
  for (int n = 0; n <= swarm->GetMaxActiveIndex(); ++n) {
    if (mask_h(n)) particles.push_back({x1_h(n), x2_h(n)});
  }
  std::sort(particles.begin(), particles.end());
  return particles;
}

//----------------------------------------------------------------------------------------
// Find the MeshBlock with the requested logical location in the target mesh.
std::shared_ptr<parthenon::MeshBlock> FindBlock(const std::shared_ptr<Mesh> &mesh,
                                                const LogicalLocation &loc) {
  for (const auto &pmb : mesh->block_list) {
    if (pmb->loc == loc) return pmb;
  }
  return nullptr;
}

} // namespace

//----------------------------------------------------------------------------------------
// Start with one coarse block, place one particle in each quadrant, remesh to the
// refined mesh, and verify that each daughter receives exactly its own particle.
TEST_CASE("Swarm particles remesh onto refined blocks", "[swarm][amr][MPI]") {
  auto app_in = std::make_shared<ApplicationInput>();
  auto packages = MakePackages();
  auto coarse_mesh = MakeMesh(false, app_in.get(), packages);
  auto refined_mesh = MakeMesh(true, app_in.get(), packages);

  REQUIRE(coarse_mesh->block_list.size() == 1);
  REQUIRE(refined_mesh->block_list.size() == 4);
  REQUIRE(coarse_mesh->block_list.front() != nullptr);

  AddParticles(GetSwarm(coarse_mesh->block_list.front()),
               {{0.25, 0.25}, {0.75, 0.25}, {0.25, 0.75}, {0.75, 0.75}});

  const auto old_locs = coarse_mesh->GetLocList();
  const auto new_locs = refined_mesh->GetLocList();
  const std::vector<int> old_to_new = {0};
  const std::vector<int> old_ranks(old_locs.size(), 0);
  const std::vector<int> new_ranks(new_locs.size(), 0);
  const SwarmRemeshContext context(0, 0, old_to_new, old_locs, new_locs, old_ranks,
                                   new_ranks);

  parthenon::RemeshSwarms(refined_mesh->resolved_packages, coarse_mesh->block_list,
                          refined_mesh.get(), context);

  const std::map<LogicalLocation, std::array<Real, 2>> expected = {
      {LogicalLocation(0, 1, 0, 0, 0), {0.25, 0.25}},
      {LogicalLocation(0, 1, 1, 0, 0), {0.75, 0.25}},
      {LogicalLocation(0, 1, 0, 1, 0), {0.25, 0.75}},
      {LogicalLocation(0, 1, 1, 1, 0), {0.75, 0.75}},
  };
  for (const auto &[loc, particle] : expected) {
    auto particles = GetParticles(GetSwarm(FindBlock(refined_mesh, loc)));
    REQUIRE(particles.size() == 1);
    REQUIRE(particles[0] == particle);
  }
}

//----------------------------------------------------------------------------------------
// Start with one particle in each refined daughter block, remesh to the coarse mesh, and
// verify that all four particles land in the single parent block.
TEST_CASE("Swarm particles remesh onto derefined blocks", "[swarm][amr][MPI]") {
  auto app_in = std::make_shared<ApplicationInput>();
  auto packages = MakePackages();
  auto refined_mesh = MakeMesh(true, app_in.get(), packages);
  auto coarse_mesh = MakeMesh(false, app_in.get(), packages);

  REQUIRE(refined_mesh->block_list.size() == 4);
  REQUIRE(coarse_mesh->block_list.size() == 1);

  AddParticles(GetSwarm(FindBlock(refined_mesh, LogicalLocation(0, 1, 0, 0, 0))),
               {{0.25, 0.25}});
  AddParticles(GetSwarm(FindBlock(refined_mesh, LogicalLocation(0, 1, 1, 0, 0))),
               {{0.75, 0.25}});
  AddParticles(GetSwarm(FindBlock(refined_mesh, LogicalLocation(0, 1, 0, 1, 0))),
               {{0.25, 0.75}});
  AddParticles(GetSwarm(FindBlock(refined_mesh, LogicalLocation(0, 1, 1, 1, 0))),
               {{0.75, 0.75}});

  const auto old_locs = refined_mesh->GetLocList();
  const auto new_locs = coarse_mesh->GetLocList();
  const std::vector<int> old_to_new(old_locs.size(), 0);
  const std::vector<int> old_ranks(old_locs.size(), 0);
  const std::vector<int> new_ranks(new_locs.size(), 0);
  const SwarmRemeshContext context(0, old_locs.size() - 1, old_to_new, old_locs, new_locs,
                                   old_ranks, new_ranks);

  parthenon::RemeshSwarms(coarse_mesh->resolved_packages, refined_mesh->block_list,
                          coarse_mesh.get(), context);

  REQUIRE(GetParticles(GetSwarm(coarse_mesh->block_list.front())) ==
          std::vector<std::array<Real, 2>>{
              {0.25, 0.25}, {0.25, 0.75}, {0.75, 0.25}, {0.75, 0.75}});
}
