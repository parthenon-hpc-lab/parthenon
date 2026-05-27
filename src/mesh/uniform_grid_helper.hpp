#pragma once
#include <array>
#include <limits>
#include <cstdint>
#include "parthenon_arrays.hpp"

namespace parthenon {

class Mesh; 

// Assuming a uniform grid, this class helps gather information about the grid layout
// across all meshblocks on a rank.
class UniformGridHelper {

public:
    explicit UniformGridHelper(Mesh *mesh);
    void Initialize();

    // Information about the logical location of blocks on this rank
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
    
    std::array<int, 3> block_size;
    std::array<int, 3> local_mesh_size;
    std::array<int, 3> global_mesh_size;
    std::array<int, 3> mesh_start_idx;
    std::array<int, 3> mesh_end_idx;
    
    parthenon::ParArray2D<std::int64_t> loc_view; // logical location of local blocks; stored on device

    // Computes the flat array index for a given meshblock and cell index.
    // Call from within a par_for loop over blocks and interior cells.
    KOKKOS_INLINE_FUNCTION
    std::int64_t FlatIndex(int b, int k, int j, int i) const {
        const auto kk = k - kb_.s + loc_view(b, 2) * block_size[2];
        const auto jj = j - jb_.s + loc_view(b, 1) * block_size[1];
        const auto ii = i - ib_.s + loc_view(b, 0) * block_size[0];
        return (std::int64_t)kk * local_mesh_size[1] * local_mesh_size[0]
            + (std::int64_t)jj * local_mesh_size[0] + ii;
    }

private:
    Mesh *mesh_;           
    bool initialized_ = false;
    // Bounds for interior cells, needed to compute local indices for FFT packing
    IndexRange ib_, jb_, kb_;
};

} // namespace parthenon