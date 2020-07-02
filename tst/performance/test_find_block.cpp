//========================================================================================
// (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
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

#include <sstream>

#include <catch2/catch.hpp>

#include <parthenon/parthenon.hpp>

using namespace parthenon::prelude;

using parthenon::ParArray4D;

char const *test_find_block_in = R"(
<parthenon/job>
problem_id = test_find_block

<parthenon/mesh>
refinement = adaptive
numlevel = 5

nx1 = 1          # Overridden at runtime
x1min = -2.0
x1max = 2.0
ix1_bc = outflow
ox1_bc = outflow

nx2 = 1          # Overridden at runtime
x2min = -2.0
x2max = 2.0
ix2_bc = outflow
ox2_bc = outflow

nx3 = 1          # Overridden at runtime
x3min = -0.5
x3max = 0.5

<parthenon/meshblock>
nx1 = 8
nx2 = 8
nx3 = 1
)";

TEST_CASE("FindMeshBlock performance", "[Mesh][performance]") {
  SECTION("Meshes") {
    GIVEN("Square Mesh") {
      const int block_length[] = {16, 64, 128, 256, 512};

      for (auto &length : block_length) {
        std::stringstream ss;
        ss << test_find_block_in;

        parthenon::ParameterInput pin;
        pin.LoadFromStream(ss);

        pin.SetInteger("parthenon/mesh", "nx1", length);
        pin.SetInteger("parthenon/mesh", "nx2", length);

        parthenon::Mesh mesh(&pin, parthenon::Properties_t{}, parthenon::Packages_t{});

        auto const nb = mesh.GetNumMeshBlocksThisRank(0);

        std::vector<MeshBlock const *> blocks;
        blocks.resize(nb);
        BENCHMARK(std::string("Find Last Block ") + std::to_string(length) + "x" +
                  std::to_string(length)) {
          // Return just forces the compiler not to optimize it out
          return mesh.FindMeshBlock(nb - 1);
        };
      }
    } // GIVEN
  }   // SECTION
}
