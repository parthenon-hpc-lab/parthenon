//========================================================================================
// (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
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
#ifndef LUKEWARM_LUKEWARM_GEOMETRY_HPP_
#define LUKEWARM_LUKEWARM_GEOMETRY_HPP_

// This file was made in part with generative AI.

#include <vector>

#include <Kokkos_Core.hpp>

#include "basic_types.hpp"
#include "kokkos_abstraction.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

template <typename T>
class MeshData;

// Where a particle was found by LukewarmGeometry::Locate()
class LukewarmLocation {
 public:
  enum class Status { InMeshData, InOtherMeshData, LeftMesh, Lost };

  KOKKOS_FORCEINLINE_FUNCTION
  LukewarmLocation(const Status status, const int block, const int gid)
      : status_(status), block_(block), gid_(gid) {}

  // In block Block() of this MeshData
  KOKKOS_FORCEINLINE_FUNCTION
  bool InMeshData() const { return status_ == Status::InMeshData; }
  // In block Gid() of another MeshData, where RelocateLukewarm() will deliver it
  KOKKOS_FORCEINLINE_FUNCTION
  bool InOtherMeshData() const { return status_ == Status::InOtherMeshData; }
  // Left through a non-periodic boundary, so RelocateLukewarm() will remove it
  KOKKOS_FORCEINLINE_FUNCTION
  bool LeftMesh() const { return status_ == Status::LeftMesh; }
  // Never settled in a block, e.g., because its position is NaN
  KOKKOS_FORCEINLINE_FUNCTION
  bool Lost() const { return status_ == Status::Lost; }

  KOKKOS_FORCEINLINE_FUNCTION
  Status GetStatus() const { return status_; }
  // Local index of the block in this MeshData
  KOKKOS_FORCEINLINE_FUNCTION
  int Block() const {
    PARTHENON_DEBUG_REQUIRE(InMeshData(), "Lukewarm particle is not in this MeshData");
    return block_;
  }
  // Global id of the block, in this MeshData or another
  KOKKOS_FORCEINLINE_FUNCTION
  int Gid() const {
    PARTHENON_DEBUG_REQUIRE(InMeshData() || InOtherMeshData(),
                            "Lukewarm particle is not in a block");
    return gid_;
  }

 private:
  Status status_;
  int block_;
  int gid_;
};

// Device-side map of the blocks of one MeshData, used to follow particles from block to
// block inside a kernel. Positions are in the mesh's coordinates.
//
// Each block is split in half along every active direction, and space just outside it
// is a half-block wide. That gives a 4x4x4 grid of half-blocks, indexed 0 (beyond the
// lower face) to 3 (beyond the upper face), and each outside half-block lies in exactly
// one neighbor, whether that neighbor is coarser, finer, or at the same level.
class LukewarmGeometry {
 public:
  LukewarmGeometry() = default;
  // owners holds the id of the MeshData owning each block, by gid, and neighbors holds
  // the sorted ids of the MeshData next to this one
  LukewarmGeometry(MeshData<Real> *md, const std::vector<int> &owners,
                   const std::vector<int> &neighbors);

  KOKKOS_FORCEINLINE_FUNCTION
  int NumBlocks() const { return nblocks_; }
  // Global id of local block b
  KOKKOS_FORCEINLINE_FUNCTION
  int Gid(const int b) const { return gids_(b); }
  // Lower and upper faces of local block b along dir, as in its block_size
  KOKKOS_FORCEINLINE_FUNCTION
  Real Xmin(const int b, const CoordinateDirection dir) const {
    return bounds_(b, dir - 1, 0);
  }
  KOKKOS_FORCEINLINE_FUNCTION
  Real Xmax(const int b, const CoordinateDirection dir) const {
    return bounds_(b, dir - 1, 2);
  }

  // The daughter of local block b holding a position in it, as ox1 + 2 * ox2 + 4 * ox3
  // for LogicalLocation::GetDaughter(ox1, ox2, ox3). Along inactive directions oxd = 0.
  KOKKOS_INLINE_FUNCTION
  int Daughter(const int b, const Real x1, const Real x2, const Real x3) const {
    return (HalfIndex_(b, 0, x1) >= 2) + 2 * (HalfIndex_(b, 1, x2) >= 2) +
           4 * (HalfIndex_(b, 2, x3) >= 2);
  }

  // Local index of the block with global id gid, or -1 if it is not in this MeshData
  KOKKOS_INLINE_FUNCTION
  int LocalIndex(const int gid) const { return Search_(local_gids_, local_index_, gid); }
  // Index among the neighbors of this MeshData of the one owning the block with global id
  // gid, or -1 if no neighbor owns a block next to this MeshData with that id
  KOKKOS_INLINE_FUNCTION
  int NeighborIndex(const int gid) const {
    return Search_(remote_gids_, remote_neighbor_, gid);
  }

  KOKKOS_INLINE_FUNCTION
  bool Contains(const int b, const Real x1, const Real x2, const Real x3) const {
    return IsInside_(HalfIndex_(b, 0, x1)) && IsInside_(HalfIndex_(b, 1, x2)) &&
           IsInside_(HalfIndex_(b, 2, x3));
  }

  // Local index of the block containing the position, or -1 if no block of this
  // MeshData does. This is a linear search, so prefer Locate() when a nearby block is
  // known.
  KOKKOS_INLINE_FUNCTION
  int FindBlock(const Real x1, const Real x2, const Real x3) const {
    for (int b = 0; b < nblocks_; ++b) {
      if (Contains(b, x1, x2, x3)) return b;
    }
    return -1;
  }

  // Indices of the interior cell of local block b that contains a position in the block.
  // Rounding can put a position just inside a face into the cell beyond it, so the
  // indices are clamped to the interior.
  KOKKOS_FORCEINLINE_FUNCTION
  void Xtoijk(const int b, const Real x1, const Real x2, const Real x3, int &i, int &j,
              int &k) const {
    i = CellIndex_(b, 0, x1);
    j = CellIndex_(b, 1, x2);
    k = CellIndex_(b, 2, x3);
  }

  // Follow a particle starting in local block b to the block containing its position,
  // wrapping the position across periodic boundaries along the way. An unlocated
  // particle, with b < 0, starts from the block of this MeshData nearest to it, so that
  // one within half a block of this MeshData is found in one hop.
  KOKKOS_INLINE_FUNCTION
  LukewarmLocation Locate(int b, Real &x1, Real &x2, Real &x3) const {
    using Status = LukewarmLocation::Status;
    // A NaN position compares false against every bound, so it would otherwise wander
    // until the hop limit, or leave through the first non-periodic boundary it meets
    if (x1 != x1 || x2 != x2 || x3 != x3) return LukewarmLocation(Status::Lost, -1, -1);
    const Real x[3] = {x1, x2, x3};
    // Mesh lengths to add to the position, applied once at the end so that crossing a
    // periodic boundary and back leaves the position exactly as it was
    int shift[3] = {0, 0, 0};
    if (b < 0) b = NearestBlock_(x, shift);
    LukewarmLocation result(Status::Lost, -1, -1);
    for (int hop = 0; hop < max_hops_; ++hop) {
      const int i = HalfIndex_(b, 0, x[0] + shift[0] * length_[0]);
      const int j = HalfIndex_(b, 1, x[1] + shift[1] * length_[1]);
      const int k = HalfIndex_(b, 2, x[2] + shift[2] * length_[2]);
      if (IsInside_(i) && IsInside_(j) && IsInside_(k)) {
        result = LukewarmLocation(Status::InMeshData, b, gids_(b));
        break;
      }
      const int h = i + 4 * (j + 4 * k);
      const int gid = neighbor_gids_(b, h);
      if (gid < 0) {
        result = LukewarmLocation(Status::LeftMesh, -1, -1);
        break;
      }
      for (int d = 0; d < 3; ++d) {
        shift[d] += shifts_(b, h, d);
      }
      b = neighbor_blocks_(b, h);
      if (b < 0) {
        result = LukewarmLocation(Status::InOtherMeshData, -1, gid);
        break;
      }
    }
    x1 = x[0] + shift[0] * length_[0];
    x2 = x[1] + shift[1] * length_[1];
    x3 = x[2] + shift[2] * length_[2];
    return result;
  }

 private:
  // The value paired with key in sorted keys, or -1 if key is missing
  KOKKOS_INLINE_FUNCTION
  static int Search_(const ParArray1DRaw<int> &keys, const ParArray1DRaw<int> &values,
                     const int key) {
    int lo = 0, hi = keys.extent_int(0);
    while (lo < hi) {
      const int mid = (lo + hi) / 2;
      if (keys(mid) < key) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    return (lo < keys.extent_int(0) && keys(lo) == key) ? values(lo) : -1;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  static bool IsInside_(const int half) { return half == 1 || half == 2; }

  KOKKOS_FORCEINLINE_FUNCTION
  int CellIndex_(const int b, const int d, const Real x) const {
    const int c =
        cell_start_[d] + static_cast<int>((x - bounds_(b, d, 0)) * bounds_(b, d, 3));
    return Kokkos::min(Kokkos::max(c, cell_start_[d]), cell_end_[d]);
  }

  // Local index of the block nearest to a position, and holding it if any block of this
  // MeshData does, measured across periodic boundaries, with the mesh lengths that move
  // the position to its image nearest that block
  KOKKOS_INLINE_FUNCTION
  int NearestBlock_(const Real x[3], int shift[3]) const {
    int nearest = 0;
    Real nearest_distance = 0.0;
    for (int b = 0; b < nblocks_; ++b) {
      Real distance = 0.0;
      int image[3] = {0, 0, 0};
      for (int d = 0; d < 3; ++d) {
        if (!active_[d]) continue;
        Real gap = Gap_(b, d, x[d]);
        for (int s = -1; periodic_[d] && s <= 1; s += 2) {
          const Real image_gap = Gap_(b, d, x[d] + s * length_[d]);
          if (image_gap < gap) {
            gap = image_gap;
            image[d] = s;
          }
        }
        distance += gap * gap;
      }
      if (b == 0 || distance < nearest_distance) {
        nearest = b;
        nearest_distance = distance;
        for (int d = 0; d < 3; ++d) {
          shift[d] = image[d];
        }
      }
    }
    return nearest;
  }

  // Distance from a position to local block b along d, zero inside it
  KOKKOS_FORCEINLINE_FUNCTION
  Real Gap_(const int b, const int d, const Real x) const {
    return Kokkos::max(Kokkos::max(bounds_(b, d, 0) - x, x - bounds_(b, d, 2)), Real(0));
  }

  KOKKOS_FORCEINLINE_FUNCTION
  int HalfIndex_(const int b, const int d, const Real x) const {
    if (!active_[d]) return 1;
    if (x < bounds_(b, d, 0)) return 0;
    if (x < bounds_(b, d, 1)) return 1;
    if (x < bounds_(b, d, 2)) return 2;
    return 3;
  }

  int nblocks_ = 0;
  int max_hops_ = 0;
  // Lower face, midplane, upper face, and inverse cell width of each block, along each
  // direction. The inverse width is zero along inactive directions.
  ParArray3DRaw<Real> bounds_;
  ParArray1DRaw<int> gids_;
  // For each block and outside half-block: the local index of the block it lies in, or
  // -1 if that block belongs to another MeshData
  ParArray2DRaw<int> neighbor_blocks_;
  // ... and the global id of that block, or -1 beyond a non-periodic boundary
  ParArray2DRaw<int> neighbor_gids_;
  // ... and the multiple of the mesh length to add to a position moving into it, which
  // is nonzero only across a periodic boundary
  ParArray3DRaw<int> shifts_;
  // Sorted gids of the blocks of this MeshData, and their local indices
  ParArray1DRaw<int> local_gids_;
  ParArray1DRaw<int> local_index_;
  // Sorted gids of the blocks of other MeshData next to this one, and the index of the
  // neighbor owning each
  ParArray1DRaw<int> remote_gids_;
  ParArray1DRaw<int> remote_neighbor_;
  // Interior cell indices, which are the same for every block
  Kokkos::Array<int, 3> cell_start_;
  Kokkos::Array<int, 3> cell_end_;
  Kokkos::Array<bool, 3> active_;
  Kokkos::Array<bool, 3> periodic_;
  Kokkos::Array<Real, 3> length_;
};

} // namespace parthenon

#endif // LUKEWARM_LUKEWARM_GEOMETRY_HPP_
