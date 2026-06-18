//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2026 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================

// This file was made in part with generative AI.

#ifndef UTILS_UNIFORM_GRID_HELPER_HPP_
#define UTILS_UNIFORM_GRID_HELPER_HPP_

#include <array>
#include <cstdint>
#include <limits>
#include <string>

#include "parthenon_arrays.hpp"

namespace parthenon {

class Mesh;

struct Box3D {
  int low[3];
  int high[3];
  int size[3]; // size in each dimension: high - low + 1
};

// Assuming a uniform grid, this class helps gather information about the grid layout
// across all meshblocks on a rank.
class UniformGridHelper {
 public:
  explicit UniformGridHelper(Mesh *mesh);

  Box3D mesh_block_box;
  Box3D local_mesh_box;

  struct KernelHelper {
    parthenon::ParArray2D<std::int64_t> loc_view;
    Box3D mesh_block_box;
    Box3D local_mesh_box;

    KOKKOS_INLINE_FUNCTION
    std::int64_t FlatIndex(int b, int k, int j, int i) const {
      const auto kk = k - mesh_block_box.low[2] + loc_view(b, 2) * mesh_block_box.size[2];
      const auto jj = j - mesh_block_box.low[1] + loc_view(b, 1) * mesh_block_box.size[1];
      const auto ii = i - mesh_block_box.low[0] + loc_view(b, 0) * mesh_block_box.size[0];
      return (std::int64_t)kk * local_mesh_box.size[1] * local_mesh_box.size[0] +
             (std::int64_t)jj * local_mesh_box.size[0] + ii;
    }
  };

  KernelHelper GetKernelHelper() const {
    return {loc_view, mesh_block_box, local_mesh_box};
  }

  // Gathers a single component of a named variable from meshblocks
  // into a contiguous 1D array suitable for FFT input.
  // output must be pre-allocated with size >= size_real_space_box()
  void GatherField(const std::string &var_name, const int var_index,
                   parthenon::ParArray1D<Real> &output);

  // Distributes a contiguous 1D array back to meshblocks.
  // Inverse of GatherField.
  void ScatterField(const parthenon::ParArray1D<Real> &input, const std::string &var_name,
                    const int var_index);

 private:
  Mesh *mesh_;
  parthenon::ParArray2D<std::int64_t>
      loc_view; // logical location of local blocks; stored on device for use in kernels
};

} // namespace parthenon

#endif // UTILS_UNIFORM_GRID_HELPER_HPP_
