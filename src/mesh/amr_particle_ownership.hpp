//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2026 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
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

// This file was made in part with generative AI

#ifndef MESH_AMR_PARTICLE_OWNERSHIP_HPP_
#define MESH_AMR_PARTICLE_OWNERSHIP_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include "defs.hpp"
#include "domain.hpp"
#include "mesh/forest/logical_location.hpp"

namespace parthenon {

class Mesh;

namespace amr {

// The remesh path preserves particle coordinates and reassigns ownership by geometric
// containment in the post-remesh leaf block domains. Keep this logic centralized so the
// production remesh path and the deterministic tests use the same convention.
inline bool PointInBlock(const RegionSize &block_size, const Real x, const Real y,
                         const Real z, const int ndim) {
  constexpr Real eps = 64.0 * std::numeric_limits<Real>::epsilon();
  const auto in_dir = [&](const int dir, const Real coord) {
    const auto cdir = static_cast<CoordinateDirection>(dir);
    const Real xmin = block_size.xmin(cdir);
    const Real xmax = block_size.xmax(cdir);
    const Real scale = std::max<Real>(1.0, std::abs(xmax - xmin));
    return coord >= xmin - eps * scale && coord <= xmax + eps * scale;
  };
  return in_dir(X1DIR, x) && (ndim < 2 || in_dir(X2DIR, y)) &&
         (ndim < 3 || in_dir(X3DIR, z));
}

// Match Parthenon's existing shared-element ownership convention for face/edge/node data:
// higher level wins, and ties at the same level are broken by larger (tree, Morton)
// ordering. Use the same rule for particles on an exactly shared block face so swarm
// ownership does not invent a second convention.
inline bool OwnershipLessThan(const LogicalLocation &a, const LogicalLocation &b) {
  if (a.level() != b.level()) return a.level() < b.level();
  if (a.tree() != b.tree()) return a.tree() < b.tree();
  return a.morton() < b.morton();
}

int FindOwningBlock(const Mesh *pmesh, const std::vector<LogicalLocation> &locs, const Real x,
                    const Real y, const Real z);

// Map a particle position to the owning cell on a uniform Cartesian block.
// This mirrors SwarmDeviceContext::Xtoijk, but is written in terms of RegionSize and
// IndexShape so it can be used in host-side ownership tests without constructing a swarm.
inline std::array<int, 3> FindCellIndices(const RegionSize &block_size,
                                          const IndexShape &cellbounds, const Real x,
                                          const Real y, const Real z, const int ndim) {
  const auto interior_i = cellbounds.GetBoundsI(IndexDomain::interior);
  const auto interior_j = cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto interior_k = cellbounds.GetBoundsK(IndexDomain::interior);

  const auto index_in_dir = [&](const CoordinateDirection dir, const Real coord,
                                const int start, const int count) {
    if (count <= 1) return start;
    const Real xmin = block_size.xmin(dir);
    const Real xmax = block_size.xmax(dir);
    const Real dx = (xmax - xmin) / static_cast<Real>(count);
    // Clamp onto the valid interior-cell range so points that lie on the upper block
    // face within roundoff still map to the final owned cell of the block.
    const Real scaled = (coord - xmin) / dx;
    const int offset =
        std::clamp(static_cast<int>(std::floor(scaled)), 0, std::max(0, count - 1));
    return start + offset;
  };

  return {index_in_dir(X1DIR, x, interior_i.s, block_size.nx(X1DIR)),
          ndim > 1 ? index_in_dir(X2DIR, y, interior_j.s, block_size.nx(X2DIR))
                   : interior_j.s,
          ndim > 2 ? index_in_dir(X3DIR, z, interior_k.s, block_size.nx(X3DIR))
                   : interior_k.s};
}

} // namespace amr
} // namespace parthenon

#endif // MESH_AMR_PARTICLE_OWNERSHIP_HPP_
