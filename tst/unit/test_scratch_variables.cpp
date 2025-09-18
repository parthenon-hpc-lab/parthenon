//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2025 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2025. Triad National Security, LLC. All rights reserved.
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
#include <memory>

#include <catch2/catch.hpp>

#include "basic_types.hpp"
#include "interface/packages.hpp"
#include "interface/state_descriptor.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/meshblock.hpp"
#include "pack/make_pack_descriptor.hpp"
#include "pack/scratch_variables.hpp"

using TT = parthenon::TopologicalType;
namespace parthenon {
SCRATCH_VARIABLE(first, TT::Cell, 3)
SCRATCH_VARIABLE(second, TT::Cell, 2, 4)
SCRATCH_VARIABLE(third, TT::Cell, 1)
SCRATCH_VARIABLE(fourth, TT::Cell, 5)
} // namespace parthenon
  //
namespace {
parthenon::BlockList_t
MakeBlockList(const std::shared_ptr<parthenon::StateDescriptor> pkg, const int NBLOCKS,
              const int NSIDE, const int NDIM) {
  parthenon::BlockList_t block_list;
  block_list.reserve(NBLOCKS);
  for (int i = 0; i < NBLOCKS; ++i) {
    auto pmb = std::make_shared<parthenon::MeshBlock>(NSIDE, NDIM);
    auto &pmbd = pmb->meshblock_data.Get();
    pmbd->Initialize(pkg, pmb);
    block_list.push_back(pmb);
  }
  return block_list;
}

TEST_CASE("Test registering scratch variables to different StateDescriptors",
          "[Scratch]") {
  using scratch_list_a =
      parthenon::ScratchVariableList<parthenon::first_t, parthenon::second_t>;
  using scratch_list_b =
      parthenon::ScratchVariableList<parthenon::third_t, parthenon::fourth_t>;

  THEN("ScratchVariableList correctly offsets the scratch variables") {
    // for scratch_list_a the first 8 fields are associated with the shape{2,4} second
    static_assert(
        std::is_same_v<scratch_list_a::type<parthenon::second_t>,
                       parthenon::ScratchVariable_impl<parthenon::second_t, 0>>);
    static_assert(std::is_same_v<scratch_list_a::type<parthenon::first_t>,
                                 parthenon::ScratchVariable_impl<parthenon::first_t, 8>>);
  }

  GIVEN("Two state descriptors that we register our scratch vars on") {
    parthenon::Packages_t packages;
    auto pkgA = std::make_shared<parthenon::StateDescriptor>("packageA");
    auto pkgB = std::make_shared<parthenon::StateDescriptor>("packageB");
    packages.Add(pkgA);
    packages.Add(pkgB);

    pkgA->AddField(scratch_list_a());
    pkgB->AddField(scratch_list_b());
    auto pkg = parthenon::ResolvePackages(packages);

    THEN("Packages have the right number of scratch fields present") {
      for (int n = 0; n < 11; n++) {
        CHECK(pkgA->FieldPresent("scratch_cell_" + std::to_string(n)));
      }
      CHECK(!pkgA->FieldPresent("scratch_cell_11"));

      for (int n = 0; n < 6; n++) {
        CHECK(pkgB->FieldPresent("scratch_cell_" + std::to_string(n)));
      }
      CHECK(!pkgB->FieldPresent("scratch_cell_6"));

      pkg->Fields();
      for (int n = 0; n < 11; n++) {
        CHECK(pkg->FieldPresent("scratch_cell_" + std::to_string(n)));
      }
      CHECK(!pkg->FieldPresent("scratch_cell_11"));
    }

    WHEN("We make a mesh and initialize the pkgA scratch") {
      constexpr int N = 6;
      constexpr int NDIM = 3;
      constexpr int NBLOCKS = 9;
      auto pkg = parthenon::ResolvePackages(packages);

      parthenon::BlockList_t block_list = MakeBlockList(pkg, NBLOCKS, N, NDIM);
      auto ib = block_list[0]->cellbounds.GetBoundsI(parthenon::IndexDomain::entire);
      auto jb = block_list[0]->cellbounds.GetBoundsJ(parthenon::IndexDomain::entire);
      auto kb = block_list[0]->cellbounds.GetBoundsK(parthenon::IndexDomain::entire);

      parthenon::MeshData<parthenon::Real> mesh_data("base");
      mesh_data.Initialize(block_list, nullptr);

      using First = scratch_list_a::type<parthenon::first_t>;
      using Second = scratch_list_a::type<parthenon::second_t>;

      auto descA = parthenon::MakePackDescriptor<First, Second>(pkg.get());
      auto packA = descA.GetPack(&mesh_data);

      parthenon::par_for(
          "scratch_A", 0, NBLOCKS - 1, kb, jb, ib,
          KOKKOS_LAMBDA(int b, int k, int j, int i) {
            parthenon::seq_for(
                0, 2, [&](int n) { packA(b, First(n), k, j, i) = n * k * j * i; });

            parthenon::seq_for(0, 1, 0, 3, [&](int n, int l) {
              packA(b, Second(n, l), k, j, i) = n * k + l * j * i;
            });
          });

      THEN("When we pack on the scratch in pkgB we access the same fields") {
        using Third = scratch_list_b::type<parthenon::third_t>;
        using Fourth = scratch_list_b::type<parthenon::fourth_t>;
        auto descB = parthenon::MakePackDescriptor<Third, Fourth>(pkg.get());
        auto packB = descB.GetPack(&mesh_data);

        int nwrong = 0;
        parthenon::par_reduce(
            "scratch_B", 0, NBLOCKS - 1, kb, jb, ib,
            KOKKOS_LAMBDA(int b, int k, int j, int i, int &nwrong_loc) {
              if (packB(b, Fourth(0), k, j, i) != packA(b, Second(0, 0), k, j, i)) {
                nwrong_loc += 1;
              }
              if (packB(b, Third(), k, j, i) != packA(b, Second(1, 1), k, j, i)) {
                nwrong_loc += 1;
              }
            },
            nwrong);
        REQUIRE(nwrong == 0);
      }
    }
  }
}
} // namespace
