//========================================================================================
// (C) (or copyright) 2021-2022. Triad National Security, LLC. All rights reserved.
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
#ifndef INTERFACE_SWARM_DEVICE_CONTEXT_HPP_
#define INTERFACE_SWARM_DEVICE_CONTEXT_HPP_

#include "coordinates/coordinates.hpp"
#include "utils/utils.hpp"

namespace parthenon {

struct SwarmKey {
  KOKKOS_INLINE_FUNCTION
  SwarmKey() {}
  KOKKOS_INLINE_FUNCTION
  SwarmKey(const int cell_idx_1d, const int swarm_idx_1d)
      : cell_idx_1d_(cell_idx_1d), swarm_idx_(swarm_idx_1d) {}

  int cell_idx_1d_;
  int swarm_idx_;
};

struct SwarmKeyComparator {
  KOKKOS_INLINE_FUNCTION
  bool operator()(const SwarmKey &s1, const SwarmKey &s2) {
    return s1.cell_idx_1d_ < s2.cell_idx_1d_;
  }
};

// TODO(BRR) Template this class on coordinates/pass appropriate additional args to e.g.
// coords_.CellWidthFA()
class SwarmDeviceContext {
 public:
  KOKKOS_FUNCTION
  bool IsActive(int n) const { return mask_(n); }

  KOKKOS_FUNCTION
  bool IsOnCurrentMeshBlock(int n) const { return block_index_(n) == this_block_; }

  KOKKOS_FUNCTION
  void MarkParticleForRemoval(int n) const { marked_for_removal_(n) = true; }

  KOKKOS_FUNCTION
  bool IsMarkedForRemoval(const int n) const { return marked_for_removal_(n); }

  // TODO(BRR) This logic will change for non-uniform cartesian meshes
  KOKKOS_INLINE_FUNCTION
  int GetNeighborBlockIndex(const int &n, const double &x, const double &y,
                            const double &z, bool &is_on_current_mesh_block) const {
    int i = static_cast<int>(std::floor((x - x_min_) / ((x_max_ - x_min_) / 2.))) + 1;
    int j = static_cast<int>(std::floor((y - y_min_) / ((y_max_ - y_min_) / 2.))) + 1;
    int k = static_cast<int>(std::floor((z - z_min_) / ((z_max_ - z_min_) / 2.))) + 1;

    // Something went wrong
    if (i < 0 || i > 3 || ((j < 0 || j > 3) && ndim_ > 1) ||
        ((k < 0 || k > 3) && ndim_ > 2)) {
      PARTHENON_FAIL("Particle neighbor indices out of bounds");
    }

    // Ignore k,j indices as necessary based on problem dimension
    if (ndim_ == 1) {
      block_index_(n) = neighbor_indices_(0, 0, i);
    } else if (ndim_ == 2) {
      block_index_(n) = neighbor_indices_(0, j, i);
    } else {
      block_index_(n) = neighbor_indices_(k, j, i);
    }

    is_on_current_mesh_block = (block_index_(n) == this_block_);

    return block_index_(n);
  }

  KOKKOS_INLINE_FUNCTION
  int GetMyRank() const { return my_rank_; }

  // TODO(BRR) This logic will change for non-uniform cartesian meshes
  KOKKOS_INLINE_FUNCTION
  void Xtoijk(const Real &x, const Real &y, const Real &z, int &i, int &j, int &k) const {
    i = static_cast<int>(
            std::floor((x - x_min_) / coords_.Dxc<CoordinateDirection::X1DIR>())) +
        ib_s_;
    j = (ndim_ > 1) ? static_cast<int>(std::floor(
                          (y - y_min_) / coords_.Dxc<CoordinateDirection::X2DIR>())) +
                          jb_s_
                    : jb_s_;
    k = (ndim_ > 2) ? static_cast<int>(std::floor(
                          (z - z_min_) / coords_.Dxc<CoordinateDirection::X3DIR>())) +
                          kb_s_
                    : kb_s_;
  }

  KOKKOS_INLINE_FUNCTION
  int GetParticleCountPerCell(const int k, const int j, const int i) const {
    return cell_sorted_number_(k, j, i);
  }

  KOKKOS_INLINE_FUNCTION
  int GetFullIndex(const int k, const int j, const int i, const int n) const {
    PARTHENON_DEBUG_REQUIRE(n < cell_sorted_number_(k, j, i),
                            "Particle index out of range!");
    return cell_sorted_(cell_sorted_begin_(k, j, i) + n).swarm_idx_;
  }

  // private:
  int ib_s_;
  int jb_s_;
  int kb_s_;
  Real x_min_;
  Real x_max_;
  Real y_min_;
  Real y_max_;
  Real z_min_;
  Real z_max_;
  Real x_min_global_;
  Real x_max_global_;
  Real y_min_global_;
  Real y_max_global_;
  Real z_min_global_;
  Real z_max_global_;
  ParArray1D<bool> mask_;
  ParArray1D<bool> marked_for_removal_;
  ParArrayND<int> block_index_;
  ParArrayND<int> neighbor_indices_; // 4x4x4 array of possible block AMR regions
  ParArray1D<SwarmKey> cell_sorted_;
  ParArrayND<int> cell_sorted_begin_;
  ParArrayND<int> cell_sorted_number_;
  int ndim_;
  friend class Swarm;
  constexpr static int this_block_ = -1; // Mirrors definition in Swarm class
  constexpr static int no_block_ = -2;   // Mirrors definition in Swarm class
  int my_rank_;
  Coordinates_t coords_;
};

} // namespace parthenon

#endif // INTERFACE_SWARM_DEVICE_CONTEXT_HPP_
