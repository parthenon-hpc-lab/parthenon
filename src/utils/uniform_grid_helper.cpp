//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2026 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================

#include "utils/uniform_grid_helper.hpp"
#include "mesh/mesh.hpp"

namespace parthenon {

UniformGridHelper::UniformGridHelper(Mesh *mesh) : mesh_(mesh) {
  loc_view = parthenon::ParArray2D<std::int64_t>("logical location of local blocks",
                                                 mesh_->GetNumMeshBlocksThisRank(), 3);
  auto loc_view_h = loc_view.GetHostMirror();

  const auto level =
      mesh_->Forest().GetLegacyTreeLocation(mesh_->block_list[0]->loc).level();

  std::array<std::int64_t, 3> local_loc_min{
      std::numeric_limits<std::int64_t>::max(),
      std::numeric_limits<std::int64_t>::max(),
      std::numeric_limits<std::int64_t>::max(),
  };
  std::array<std::int64_t, 3> local_loc_max{
      std::numeric_limits<std::int64_t>::min(),
      std::numeric_limits<std::int64_t>::min(),
      std::numeric_limits<std::int64_t>::min(),
  };

  // Set rank local min and max logical locations.
  // Also check if all blocks are on the same level (we use this check instead of
  // checking for refinement=none because AMR could have been used to dynamically refine
  // a simulation. We just need to ensure that all blocks are on the same level to
  // create an effective uniform grid.)

  // lkasselm: This is currently only checked once during initialization,
  // but if the mesh is dynamically refined, this could be violated at later times.
  // Needs to be checked at each call to Gather/Scatter.

  for (int b = 0; b < mesh_->GetNumMeshBlocksThisRank(); b++) {
    auto pmb = mesh_->block_list[b];
    const auto loc = mesh_->Forest().GetLegacyTreeLocation(pmb->loc);
    for (int i = 0; i <= 2; i++) {
      local_loc_min.at(i) = std::min(loc.l(i), local_loc_min.at(i));
      local_loc_max.at(i) = std::max(loc.l(i), local_loc_max.at(i));
      loc_view_h(b, i) = loc.l(i);
    }
    PARTHENON_REQUIRE_THROWS(
        loc.level() == level,
        "uniform_grid_helper was initialized but not all blocks are on the same level.");
  }

  // convert global logical locations to rank-local logical locs
  for (int b = 0; b < mesh_->GetNumMeshBlocksThisRank(); b++) {
    for (int i = 0; i <= 2; i++) {
      loc_view_h(b, i) -= local_loc_min.at(i);
    }
  }
  Kokkos::deep_copy(loc_view, loc_view_h);

  std::array local_nlocs{
      (local_loc_max.at(0) - local_loc_min.at(0)) + 1,
      (local_loc_max.at(1) - local_loc_min.at(1)) + 1,
      (local_loc_max.at(2) - local_loc_min.at(2)) + 1,
  };
  const auto loc_max_vol = local_nlocs.at(0) * local_nlocs.at(1) * local_nlocs.at(2);

  PARTHENON_REQUIRE_THROWS(
      loc_max_vol == mesh_->GetNumMeshBlocksThisRank(),
      "Block coverage on rank cannot be matched to a contiguous "
      "array, which is required for FFTs. Try a different amount of "
      "ranks (one pack per rank, i.e. num_packs=1, will always work).");

  const auto block_size_ = mesh_->GetDefaultBlockSize();
  mesh_block_box.size[0] = block_size_.nx(parthenon::X1DIR);
  mesh_block_box.size[1] = block_size_.nx(parthenon::X2DIR);
  mesh_block_box.size[2] = block_size_.nx(parthenon::X3DIR);
  for (int i = 0; i < 3; i++) {
    local_mesh_box.size[i] = local_nlocs[i] * mesh_block_box.size[i];
    local_mesh_box.low[i] = local_loc_min[i] * mesh_block_box.size[i];
    local_mesh_box.high[i] = local_mesh_box.low[i] + local_mesh_box.size[i] - 1;
  }

  auto &md = mesh_->mesh_data.Get();
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  mesh_block_box.low[0] = ib.s;
  mesh_block_box.high[0] = ib.e;
  mesh_block_box.low[1] = jb.s;
  mesh_block_box.high[1] = jb.e;
  mesh_block_box.low[2] = kb.s;
  mesh_block_box.high[2] = kb.e;

  std::cout << "Initialized UniformGridHelper" << "\n";
  std::cout << "Mesh layout:" << "\n";
  std::cout << "Rank " << parthenon::Globals::my_rank << " local mesh box: low = ("
            << local_mesh_box.low[0] << ", " << local_mesh_box.low[1] << ", "
            << local_mesh_box.low[2] << "), high = (" << local_mesh_box.high[0] << ", "
            << local_mesh_box.high[1] << ", " << local_mesh_box.high[2] << "), size = ("
            << local_mesh_box.size[0] << ", " << local_mesh_box.size[1] << ", "
            << local_mesh_box.size[2] << ")\n";

} // UniformGridHelper::UniformGridHelper

void UniformGridHelper::GatherField(const std::string &var_name, int var_index,
                                    parthenon::ParArray1D<Real> &output) {

  // Check that var_name and var_index correspond to a valid variable in the mesh data and
  // that output array is large enough to hold the gathered data.
  auto &md = mesh_->mesh_data.Get();
  auto vars = md->PackVariables(std::vector<std::string>{var_name});
  PARTHENON_REQUIRE_THROWS(vars.GetDim(5) > 0, "GatherField: variable '" + var_name +
                                                   "' not found in mesh data");
  PARTHENON_REQUIRE_THROWS(var_index < vars.GetDim(4), "GatherField: var_index " +
                                                           std::to_string(var_index) +
                                                           " out of range");
  PARTHENON_REQUIRE_THROWS(output.size() >= local_mesh_box.size[0] *
                                                local_mesh_box.size[1] *
                                                local_mesh_box.size[2],
                           "GatherField: output array too small");

  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  const auto vi = var_index;
  auto helper = GetKernelHelper();

  parthenon::par_for(
      "UniformGridHelper::GatherField", 0, md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e,
      ib.s, ib.e, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto idx = helper.FlatIndex(b, k, j, i);
        output(idx) = vars(b, vi, k, j, i);
      });
}

void UniformGridHelper::ScatterField(const parthenon::ParArray1D<Real> &input,
                                     const std::string &var_name, int var_index) {

  auto &md = mesh_->mesh_data.Get();
  auto vars = md->PackVariables(std::vector<std::string>{var_name});
  PARTHENON_REQUIRE_THROWS(vars.GetDim(5) > 0, "ScatterField: variable '" + var_name +
                                                   "' not found in mesh data");
  PARTHENON_REQUIRE_THROWS(var_index < vars.GetDim(4), "ScatterField: var_index " +
                                                           std::to_string(var_index) +
                                                           " out of range");
  PARTHENON_REQUIRE_THROWS(input.size() >= local_mesh_box.size[0] *
                                               local_mesh_box.size[1] *
                                               local_mesh_box.size[2],
                           "ScatterField: input array too small");

  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  const auto vi = var_index;

  auto helper = GetKernelHelper();

  parthenon::par_for(
      "UniformGridHelper::ScatterField", 0, md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e,
      ib.s, ib.e, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto idx = helper.FlatIndex(b, k, j, i);
        vars(b, vi, k, j, i) = input(idx);
      });
}

} // namespace parthenon
