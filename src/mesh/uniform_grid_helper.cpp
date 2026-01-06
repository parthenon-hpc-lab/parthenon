#include "uniform_grid_helper.hpp"
#include "mesh/mesh.hpp"

namespace parthenon {

UniformGridHelper::UniformGridHelper(Mesh *mesh) : mesh_(mesh) {}

void UniformGridHelper::Initialize() {
    if (initialized_) return;
    
    // Determine global box sizes
    auto mesh_size = mesh_->mesh_size;
    global_mesh_size[0] = mesh_size.nx(X1DIR);
    global_mesh_size[1] = mesh_size.nx(X2DIR);
    global_mesh_size[2] = mesh_size.nx(X3DIR);
    
    parthenon::ParArray2D<std::int64_t> loc_view = parthenon::ParArray2D<std::int64_t>("logical location of local blocks",
                                                mesh_->GetNumMeshBlocksThisRank(), 3); 
    auto loc_view_h = loc_view.GetHostMirror();

    const auto level =
      mesh_->Forest().GetLegacyTreeLocation(mesh_->block_list[0]->loc).level();
  
    // Set rank local min and max logical locations.
    // Also check if all blocks are on the same level (we use this check instead of
    // checking for refinement=none because AMR could have been used to dynamically refine
    // a simulation. We just need to ensure that all blocks are on the same level to
    // create an effective uniform grid.)
    for (int b = 0; b < mesh_->GetNumMeshBlocksThisRank(); b++) {
        auto pmb = mesh_->block_list[b];
        const auto loc = mesh_->Forest().GetLegacyTreeLocation(pmb->loc);
        for (int i = 0; i <= 2; i++) {
        local_loc_min.at(i) = std::min(loc.l(i), local_loc_min.at(i));
        local_loc_max.at(i) = std::max(loc.l(i), local_loc_max.at(i));
        loc_view_h(b, i) = loc.l(i);
        }
        PARTHENON_REQUIRE_THROWS(loc.level() == level,
                                "Not all blocks are on the same level.");
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

    PARTHENON_REQUIRE_THROWS(loc_max_vol == mesh_->GetNumMeshBlocksThisRank(),
                            "Block coverage on rank cannot be matched to a contiguous "
                            "array, which is required for FFTs. Try a different amount of "
                            "ranks (one block per rank will always work).");

    const auto block_size_ = mesh_->GetDefaultBlockSize();
    block_size[0] = block_size_.nx(parthenon::X1DIR);
    block_size[1] = block_size_.nx(parthenon::X2DIR);
    block_size[2] = block_size_.nx(parthenon::X3DIR);
    for (int i = 0; i < 3; i++) {
        local_mesh_size[i] = local_nlocs[i] * block_size[i];
        mesh_start_idx[i] = local_loc_min[i] * block_size[i];
        mesh_end_idx[i] = mesh_start_idx[i] + local_mesh_size[i] - 1;
    }

    initialized_ = true;
}

}
