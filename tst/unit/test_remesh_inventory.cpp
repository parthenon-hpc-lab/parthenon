#include <array>
#include <cstdint>

#include <catch2/catch.hpp>

#include "interface/metadata.hpp"
#include "mesh/index_shape.hpp"
#include "mesh/remesh/inventory.hpp"

TEST_CASE("Remesh inventory uses variable allocation layouts", "[remesh][inventory]") {
  const auto shapes = parthenon::GetIndexShapes(8, 6, 4, true);
  const parthenon::Metadata cell(
      {parthenon::Metadata::Cell, parthenon::Metadata::Independent},
      std::vector<int>{2, 3});
  const auto layout = parthenon::remesh::GetStorageLayout(cell, shapes, true);

  std::uint64_t expected_data = sizeof(parthenon::Real);
  std::uint64_t expected_coarse = sizeof(parthenon::Real);
  for (const auto dim : cell.GetArrayDims(shapes, false))
    expected_data *= dim;
  for (const auto dim : cell.GetArrayDims(shapes, true))
    expected_coarse *= dim;
  REQUIRE(layout.data_bytes == expected_data);
  REQUIRE(layout.coarse_bytes == expected_coarse);

  const parthenon::Metadata scratch({parthenon::Metadata::None}, std::vector<int>{7, 5});
  const auto scratch_layout = parthenon::remesh::GetStorageLayout(scratch, shapes, true);
  REQUIRE(scratch_layout.data_bytes == 7 * 5 * sizeof(parthenon::Real));
  REQUIRE(scratch_layout.coarse_bytes == 0);
}

TEST_CASE("Remesh swarm capacity follows geometric pool growth", "[remesh][inventory]") {
  using parthenon::remesh::SelectSwarmCapacity;
  REQUIRE(SelectSwarmCapacity(0, 3) == 3);
  REQUIRE(SelectSwarmCapacity(3, 3) == 3);
  REQUIRE(SelectSwarmCapacity(4, 3) == 6);
  REQUIRE(SelectSwarmCapacity(7, 3) == 12);
  REQUIRE(SelectSwarmCapacity(0, 0) == 1);
}
