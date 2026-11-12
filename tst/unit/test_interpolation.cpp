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
#include "kokkos_abstraction.hpp"
#include "mesh/mesh_refinement.hpp"
#include "mesh/meshblock.hpp"
#include "pack/sparse_pack/make_pack_descriptor.hpp"
#include "parameter_input.hpp"
#include "utils/interpolation.hpp"

namespace {
KOKKOS_INLINE_FUNCTION parthenon::Real InterpolationValue(const int b, const int v,
                                                          const parthenon::Real x,
                                                          const parthenon::Real y,
                                                          const parthenon::Real z,
                                                          const bool constant) {
  const parthenon::Real offset = 3 * b + 7 * v + 1;
  return constant ? offset
                  : offset + 2 * x - 3 * y + 5 * z + x * y - 2 * y * z + 3 * x * z +
                        2 * x * y * z;
}
} // namespace

TEST_CASE("Sparse pack linear interpolation", "[Interpolation][SparsePack]") {
  namespace linear = parthenon::interpolation::cent::linear;
  const int ndim = GENERATE(1, 2, 3);
  const bool constant = GENERATE(false, true);
  CAPTURE(ndim, constant);
  constexpr int NSIDE = 4;
  constexpr int NBLOCKS = 2;
  constexpr int NCOMP = 2;
  // Restore the global even if a Catch2 assertion exits the test early.
  struct GhostGuard {
    int saved = parthenon::Globals::nghost;
    GhostGuard() { parthenon::Globals::nghost = 2; }
    ~GhostGuard() { parthenon::Globals::nghost = saved; }
  } ghost_guard;
  auto pkg = std::make_shared<parthenon::StateDescriptor>("interpolation test");
  pkg->AddField("field", parthenon::Metadata({parthenon::Metadata::Cell,
                                              parthenon::Metadata::Independent},
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
        parthenon::ParArray0D<parthenon::Coordinates_t>("interpolation coordinates");
    Kokkos::deep_copy(block->coords_device, block->coords);
    block->meshblock_data.Add("base", pkg, block);
    blocks.push_back(block);
  }
  parthenon::MeshData<parthenon::Real> mesh_data("base", blocks, nullptr);
  auto desc = parthenon::MakePackDescriptor(pkg.get(), std::vector<std::string>{"field"});
  const auto pack = desc.GetPack(&mesh_data);
  const auto ib = blocks[0]->cellbounds.GetBoundsI(parthenon::IndexDomain::entire);
  const auto jb = blocks[0]->cellbounds.GetBoundsJ(parthenon::IndexDomain::entire);
  const auto kb = blocks[0]->cellbounds.GetBoundsK(parthenon::IndexDomain::entire);
  const Kokkos::Array<int, 3> ncells{ib.e - ib.s + 1, jb.e - jb.s + 1, kb.e - kb.s + 1};
  parthenon::par_for(
      "initialize interpolation field", 0, NBLOCKS - 1, kb, jb, ib,
      KOKKOS_LAMBDA(int b, int k, int j, int i) {
        const auto &c = pack.GetCoordinates(b);
        for (int v = 0; v < NCOMP; ++v) {
          pack(b, v, k, j, i) = InterpolationValue(
              b, v, c.Xc<parthenon::X1DIR>(i), ndim > 1 ? c.Xc<parthenon::X2DIR>(j) : 0.,
              ndim > 2 ? c.Xc<parthenon::X3DIR>(k) : 0., constant);
        }
      });
  const parthenon::Real tol = 512 * std::numeric_limits<parthenon::Real>::epsilon();
  int nwrong = 0;
  parthenon::par_reduce(
      "check interpolation", 0, NBLOCKS - 1,
      KOKKOS_LAMBDA(int b, int &ltot) {
        const auto &c = pack.GetCoordinates(b);
        // Centers, unequal fractional weights, and both outermost intervals.
        const Kokkos::Array<parthenon::Real, 4> fi{0., 0.23, ncells[0] - 1.,
                                                   ncells[0] - 1.37};
        const Kokkos::Array<parthenon::Real, 4> fj{0., 0.61, ncells[1] - 1.,
                                                   ncells[1] - 1.19};
        const Kokkos::Array<parthenon::Real, 4> fk{0., 0.42, ncells[2] - 1.,
                                                   ncells[2] - 1.73};
        for (int sample = 0; sample < 4; ++sample) {
          const parthenon::Real x = c.Xc<parthenon::X1DIR>(0) +
                                    fi[sample] * c.CellWidth<parthenon::X1DIR>(0, 0, 0);
          const parthenon::Real y =
              ndim > 1 ? c.Xc<parthenon::X2DIR>(0) +
                             fj[sample] * c.CellWidth<parthenon::X2DIR>(0, 0, 0)
                       : 0.;
          const parthenon::Real z =
              ndim > 2 ? c.Xc<parthenon::X3DIR>(0) +
                             fk[sample] * c.CellWidth<parthenon::X3DIR>(0, 0, 0)
                       : 0.;
          for (int v = 0; v < NCOMP; ++v) {
            const parthenon::Real expected = InterpolationValue(b, v, x, y, z, constant);
            const parthenon::Real direct =
                ndim == 1   ? linear::Do1D(b, x, pack, v, ncells)
                : ndim == 2 ? linear::Do2D(b, x, y, pack, v, ncells)
                            : linear::Do3D(b, x, y, z, pack, v, ncells);
            const parthenon::Real dispatched = linear::Do(b, x, y, z, pack, v, ncells);
            if (!(std::abs(direct - expected) <= tol * (1 + std::abs(expected)))) ++ltot;
            if (!(std::abs(dispatched - expected) <= tol * (1 + std::abs(expected))))
              ++ltot;
          }
        }
      },
      nwrong);
  REQUIRE(nwrong == 0);
}
