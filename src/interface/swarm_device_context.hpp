//========================================================================================
// (C) (or copyright) 2021. Triad National Security, LLC. All rights reserved.
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

namespace parthenon {

class SwarmDeviceContext {
 public:
  KOKKOS_FUNCTION
  bool IsActive(int n) const { return mask_(n); }

  KOKKOS_FUNCTION
  bool IsOnCurrentMeshBlock(int n) const { return blockIndex_(n) == this_block_; }

  KOKKOS_FUNCTION
  void MarkParticleForRemoval(int n) const { marked_for_removal_(n) = true; }

  KOKKOS_FUNCTION
  bool IsMarkedForRemoval(const int n) const { return marked_for_removal_(n); }

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
      blockIndex_(n) = neighborIndices_(0, 0, i);
    } else if (ndim_ == 2) {
      blockIndex_(n) = neighborIndices_(0, j, i);
    } else {
      blockIndex_(n) = neighborIndices_(k, j, i);
    }

    is_on_current_mesh_block = (blockIndex_(n) == this_block_);

    return blockIndex_(n);
  }

  KOKKOS_INLINE_FUNCTION
  int GetMyRank() const { return my_rank_; }

  KOKKOS_INLINE_FUNCTION
  void Xtoijk(const Real &x, const Real &y, const Real &z, int &i, int &j, int &k) const {
    i = static_cast<int>(std::floor(x - x_min_)/dx1_) + ib_s_;
    j = static_cast<int>(std::floor(y - y_min_)/dx2_) + jb_s_;
    k = static_cast<int>(std::floor(z - z_min_)/dx3_) + kb_s_;
  }

// private:
  int ib_s_;
  int jb_s_;
  int kb_s_;
  Real dx1_;
  Real dx2_;
  Real dx3_;
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
  ParArrayND<bool> marked_for_removal_;
  ParArrayND<bool> mask_;
  ParArrayND<int> blockIndex_;
  ParArrayND<int> neighborIndices_; // 4x4x4 array of possible block AMR regions
  int ndim_;
  friend class Swarm;
  constexpr static int this_block_ = -1; // Mirrors definition in Swarm class
  int my_rank_;
};

} // namespace parthenon

#endif // SWARM_DEVICE_CONTEXT_HPP_
