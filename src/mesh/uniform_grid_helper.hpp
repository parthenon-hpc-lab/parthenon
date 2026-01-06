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
    
    parthenon::ParArray2D<std::int64_t> loc_view;

private:
    Mesh *mesh_;           
    bool initialized_ = false;
};

} // namespace parthenon