//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2026 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// This file was made in part with generative AI.

#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <catch2/catch.hpp>

#include "basic_types.hpp"
#include "globals.hpp"
#include "interface/mesh_data.hpp"
#include "interface/meshblock_data.hpp"
#include "interface/metadata.hpp"
#include "interface/state_descriptor.hpp"
#include "interface/update.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/mesh_refinement.hpp"
#include "mesh/meshblock.hpp"
#include "parameter_input.hpp"

TEST_CASE("Flux divergence helper supports sparse and legacy packs",
          "[Update][SparsePack]") {
  const int ndim = GENERATE(1, 2, 3);
  // Zero means constant flux; otherwise isolate each active direction, then sum.
  const int active = GENERATE(0, 1, 2, 3, 4);
  CAPTURE(ndim, active);
  constexpr int NSIDE = 4;
  constexpr int NBLOCKS = 2;
  constexpr int NCOMP = 2;
  // The lightweight MeshBlock constructor uses the global ghost-zone count.
  struct GhostGuard {
    int saved = parthenon::Globals::nghost;
    GhostGuard() { parthenon::Globals::nghost = 2; }
    ~GhostGuard() { parthenon::Globals::nghost = saved; }
  } ghost_guard;
  auto pkg = std::make_shared<parthenon::StateDescriptor>("update test");
  pkg->AddField("field", parthenon::Metadata({parthenon::Metadata::Cell,
                                              parthenon::Metadata::Independent,
                                              parthenon::Metadata::WithFluxes},
                                             std::vector<int>{NCOMP}));
  parthenon::BlockList_t blocks;
  blocks.reserve(NBLOCKS);
  parthenon::ParameterInput pin;
  for (int b = 0; b < NBLOCKS; ++b) {
    auto block = std::make_shared<parthenon::MeshBlock>(NSIDE, ndim);
    const parthenon::RegionSize size(
        std::array<parthenon::Real, 3>{parthenon::Real(b), -2., 3.},
        std::array<parthenon::Real, 3>{parthenon::Real(b + 2), 1., 7.},
        std::array<parthenon::Real, 3>{1., 1., 1.},
        std::array<int, 3>{NSIDE, ndim > 1 ? NSIDE : 1, ndim > 2 ? NSIDE : 1});
    block->coords = parthenon::Coordinates_t(size, &pin);
    block->coords_device =
        parthenon::ParArray0D<parthenon::Coordinates_t>("update coordinates");
    Kokkos::deep_copy(block->coords_device, block->coords);
    block->meshblock_data.Add("base", pkg, block);
    blocks.push_back(block);
  }
  parthenon::MeshData<parthenon::Real> mesh_data("base", blocks, nullptr);
  auto desc = parthenon::MakePackDescriptor(pkg.get(), std::vector<std::string>{"field"},
                                            {}, {parthenon::PDOpt::WithFluxes});
  const auto pack = desc.GetPack(&mesh_data);
  auto flat_desc = parthenon::MakePackDescriptor(
      pkg.get(), std::vector<std::string>{"field"}, {},
      {parthenon::PDOpt::WithFluxes, parthenon::PDOpt::Flatten});
  const auto flat = flat_desc.GetPack(&mesh_data);
  const parthenon::Real tol = 512 * std::numeric_limits<parthenon::Real>::epsilon();
  for (int b = 0; b < NBLOCKS; ++b) {
    CAPTURE(b);
    auto &block = blocks[b];
    const auto ib = block->cellbounds.GetBoundsI(parthenon::IndexDomain::interior);
    const auto jb = block->cellbounds.GetBoundsJ(parthenon::IndexDomain::interior);
    const auto kb = block->cellbounds.GetBoundsK(parthenon::IndexDomain::interior);
    const auto coords = block->coords;
    for (int dir = 1; dir <= ndim; ++dir) {
      parthenon::par_for(
          "initialize test flux", kb.s, kb.e + (dir == 3), jb.s, jb.e + (dir == 2), ib.s,
          ib.e + (dir == 1), KOKKOS_LAMBDA(int k, int j, int i) {
            const parthenon::Real x = dir == 1   ? coords.Xf<parthenon::X1DIR>(i)
                                      : dir == 2 ? coords.Xf<parthenon::X2DIR>(j)
                                                 : coords.Xf<parthenon::X3DIR>(k);
            for (int v = 0; v < NCOMP; ++v) {
              const parthenon::Real slope =
                  (active == dir || active == 4) ? (b + 1) * (v + 1) * dir : 0.;
              pack.flux(b, dir, v, k, j, i) = 7 + b + v + slope * x;
            }
          });
    }
    const auto legacy = block->meshblock_data.Get()->PackVariablesAndFluxes(
        std::vector<std::string>{"field"}, std::vector<std::string>{"field"});
    const int flat_lo = flat.GetLowerBoundHost(b);
    int nwrong = 0;
    parthenon::par_reduce(
        "check flux divergence", kb, jb, ib,
        KOKKOS_LAMBDA(int k, int j, int i, int &ltot) {
          for (int v = 0; v < NCOMP; ++v) {
            parthenon::Real expected = 0.;
            for (int dir = 1; dir <= ndim; ++dir) {
              if (active == dir || active == 4) expected -= (b + 1) * (v + 1) * dir;
            }
            const parthenon::Real sparse_result =
                parthenon::Update::FluxDivHelper(b, v, k, j, i, ndim, coords, pack);
            const parthenon::Real legacy_result =
                parthenon::Update::FluxDivHelper(v, k, j, i, ndim, coords, legacy);
            // A flattened sparse pack exercises the generalized no-block overload.
            const parthenon::Real flat_result = parthenon::Update::FluxDivHelper(
                flat_lo + v, k, j, i, ndim, coords, flat);
            if (!(std::abs(sparse_result - expected) <= tol * (1 + std::abs(expected))))
              ++ltot;
            if (!(std::abs(legacy_result - expected) <= tol * (1 + std::abs(expected))))
              ++ltot;
            if (!(std::abs(flat_result - expected) <= tol * (1 + std::abs(expected))))
              ++ltot;
          }
        },
        nwrong);
    REQUIRE(nwrong == 0);
  }
}
