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

    struct KernelHelper {
    parthenon::ParArray2D<std::int64_t> loc_view;
    std::array<int, 3> block_size;
    std::array<int, 3> local_mesh_size;
    IndexRange ib, jb, kb;

    KOKKOS_INLINE_FUNCTION
    std::int64_t FlatIndex(int b, int k, int j, int i) const {
        const auto kk = k - kb.s + loc_view(b, 2) * block_size[2];
        const auto jj = j - jb.s + loc_view(b, 1) * block_size[1];
        const auto ii = i - ib.s + loc_view(b, 0) * block_size[0];
        return (std::int64_t)kk * local_mesh_size[1] * local_mesh_size[0]
             + (std::int64_t)jj * local_mesh_size[0] + ii;
        }
    };

    KernelHelper GetKernelHelper() const {
        return {loc_view, block_size, local_mesh_size, ib_, jb_, kb_};
    }

    // Gathers a single component of a named variable from meshblocks 
    // into a contiguous 1D array suitable for FFT input.
    // output must be pre-allocated with size >= size_real_space_box()
    void GatherField(const std::string &var_name,
                    const int var_index,
                    parthenon::ParArray1D<Real> &output);

    // Distributes a contiguous 1D array back to meshblocks.
    // Inverse of GatherField.
    void ScatterField(const parthenon::ParArray1D<Real> &input,
                    const std::string &var_name,
                    const int var_index);

private:
    Mesh *mesh_;           
    bool initialized_ = false;
    // Bounds for interior cells, needed to compute local indices for FFT packing
    IndexRange ib_, jb_, kb_;
};

} // namespace parthenon