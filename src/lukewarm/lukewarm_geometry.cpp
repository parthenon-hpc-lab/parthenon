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

// This file was made in part with generative AI.

#include "lukewarm/lukewarm_geometry.hpp"

#include <algorithm>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "bvals/neighbor_block.hpp"
#include "interface/mesh_data.hpp"
#include "mesh/forest/logical_location.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

namespace {
// Device copies of the keys and values of a map, which come out sorted by key
void ToDevice(const std::map<int, int> &map, const std::string &label,
              ParArray1DRaw<int> &keys, ParArray1DRaw<int> &values) {
  keys = ParArray1DRaw<int>(label + "::keys", map.size());
  values = ParArray1DRaw<int>(label + "::values", map.size());
  auto keys_h = Kokkos::create_mirror_view(keys);
  auto values_h = Kokkos::create_mirror_view(values);
  int n = 0;
  for (const auto &[key, value] : map) {
    keys_h(n) = key;
    values_h(n) = value;
    ++n;
  }
  Kokkos::deep_copy(keys, keys_h);
  Kokkos::deep_copy(values, values_h);
}
} // namespace

LukewarmGeometry::LukewarmGeometry(MeshData<Real> *md, const std::vector<int> &owners,
                                   const std::vector<int> &neighbors) {
  Mesh *pmesh = md->GetMeshPointer();
  nblocks_ = md->NumBlocks();
  // A particle inside the MeshData visits each block at most a few times, even when it
  // wraps around a periodic direction
  max_hops_ = 4 * nblocks_ + 16;

  constexpr CoordinateDirection dirs[3] = {X1DIR, X2DIR, X3DIR};
  for (int d = 0; d < 3; ++d) {
    active_[d] = !pmesh->mesh_size.symmetry(dirs[d]);
    periodic_[d] =
        pmesh->mesh_bcs[GetInnerBoundaryFace(dirs[d])] == BoundaryFlag::periodic;
    length_[d] = pmesh->mesh_size.xmax(dirs[d]) - pmesh->mesh_size.xmin(dirs[d]);
  }
  const IndexRange cells[3] = {md->GetBoundsI(IndexDomain::interior),
                               md->GetBoundsJ(IndexDomain::interior),
                               md->GetBoundsK(IndexDomain::interior)};
  for (int d = 0; d < 3; ++d) {
    cell_start_[d] = cells[d].s;
    cell_end_[d] = cells[d].e;
  }

  std::map<int, int> local_of_gid, neighbor_of_gid;
  for (int b = 0; b < nblocks_; ++b) {
    local_of_gid[md->GetBlockData(b)->GetBlockPointer()->gid] = b;
  }

  bounds_ = ParArray3DRaw<Real>("LukewarmGeometry::bounds", nblocks_, 3, 4);
  gids_ = ParArray1DRaw<int>("LukewarmGeometry::gids", nblocks_);
  neighbor_blocks_ =
      ParArray2DRaw<int>("LukewarmGeometry::neighbor_blocks", nblocks_, 64);
  neighbor_gids_ = ParArray2DRaw<int>("LukewarmGeometry::neighbor_gids", nblocks_, 64);
  shifts_ = ParArray3DRaw<int>("LukewarmGeometry::shifts", nblocks_, 64, 3);
  auto bounds = Kokkos::create_mirror_view(bounds_);
  auto gids = Kokkos::create_mirror_view(gids_);
  auto neighbor_blocks = Kokkos::create_mirror_view(neighbor_blocks_);
  auto neighbor_gids = Kokkos::create_mirror_view(neighbor_gids_);
  auto shifts = Kokkos::create_mirror_view(shifts_);

  for (int b = 0; b < nblocks_; ++b) {
    const MeshBlock *pmb = md->GetBlockData(b)->GetBlockPointer();
    const LogicalLocation &loc = pmb->loc;
    gids(b) = pmb->gid;
    // The midplane comes from a daughter so that it is right on stretched meshes
    const RegionSize daughter = pmesh->GetBlockSize(loc.GetDaughter(0, 0, 0));
    for (int d = 0; d < 3; ++d) {
      bounds(b, d, 0) = pmb->block_size.xmin(dirs[d]);
      bounds(b, d, 1) = daughter.xmax(dirs[d]);
      bounds(b, d, 2) = pmb->block_size.xmax(dirs[d]);
      bounds(b, d, 3) = active_[d] ? 1.0 / pmb->coords.Dx(d + 1) : 0.0;
    }

    for (int h = 0; h < 64; ++h) {
      const int half[3] = {h % 4, (h / 4) % 4, h / 16};
      neighbor_blocks(b, h) = -1;
      neighbor_gids(b, h) = -1;
      for (int d = 0; d < 3; ++d) {
        shifts(b, h, d) = 0;
      }
      // The half-block at one finer level, in the frame of this block's tree
      std::int64_t l[3];
      bool outside = false;
      for (int d = 0; d < 3; ++d) {
        l[d] = 2 * loc.l(d) + (active_[d] ? half[d] - 1 : 0);
        outside = outside || (active_[d] && (half[d] == 0 || half[d] == 3));
      }
      if (!outside) continue;
      const LogicalLocation target(loc.tree(), loc.level() + 1, l[0], l[1], l[2]);

      for (const auto &nb : pmb->GetNeighbors()) {
        if (!nb.origin_loc.Contains(target)) continue;
        neighbor_gids(b, h) = nb.gid;
        const auto itr = local_of_gid.find(nb.gid);
        if (itr != local_of_gid.end()) {
          neighbor_blocks(b, h) = itr->second;
        } else {
          const auto neighbor =
              std::lower_bound(neighbors.begin(), neighbors.end(), owners.at(nb.gid));
          PARTHENON_REQUIRE_THROWS(neighbor != neighbors.end() &&
                                       *neighbor == owners.at(nb.gid),
                                   "Block " + std::to_string(nb.gid) +
                                       " belongs to a MeshData that is not a neighbor");
          neighbor_of_gid[nb.gid] = neighbor - neighbors.begin();
        }
        // Find which periodic image of a point in the half-block lies in the neighbor.
        // Anything but the point itself means a periodic boundary is crossed.
        const RegionSize nb_size = pmesh->GetBlockSize(nb.loc);
        for (int d = 0; d < 3; ++d) {
          if (!active_[d]) continue;
          const Real lo = bounds(b, d, 0), mid = bounds(b, d, 1), hi = bounds(b, d, 2);
          const Real width = hi - lo;
          const Real point[4] = {lo - width / 8, (lo + mid) / 2, (mid + hi) / 2,
                                 hi + width / 8};
          const Real nb_lo = nb_size.xmin(dirs[d]), nb_hi = nb_size.xmax(dirs[d]);
          bool found = false;
          for (const int shift : {0, -1, 1}) {
            const Real p = point[half[d]] + shift * length_[d];
            if (nb_lo <= p && p < nb_hi) {
              shifts(b, h, d) = shift;
              found = true;
              break;
            }
          }
          PARTHENON_REQUIRE_THROWS(found, "Neighbor " + std::to_string(nb.gid) +
                                              " of block " + std::to_string(pmb->gid) +
                                              " is not where lukewarms expect it");
        }
        break;
      }
    }
  }

  Kokkos::deep_copy(bounds_, bounds);
  Kokkos::deep_copy(gids_, gids);
  Kokkos::deep_copy(neighbor_blocks_, neighbor_blocks);
  Kokkos::deep_copy(neighbor_gids_, neighbor_gids);
  Kokkos::deep_copy(shifts_, shifts);
  ToDevice(local_of_gid, "LukewarmGeometry::local", local_gids_, local_index_);
  ToDevice(neighbor_of_gid, "LukewarmGeometry::remote", remote_gids_, remote_neighbor_);
}

} // namespace parthenon
