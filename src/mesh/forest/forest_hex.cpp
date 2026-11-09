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

#include <algorithm>
#include <array>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "mesh/forest/forest_hex.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {
namespace forest {

namespace {

using shared_t = std::vector<std::pair<int, int>>; // (corner in A, corner in B)

int Bit(const int corner, const int d) { return (corner >> d) & 1; }

std::string HexName(const Hex &h) { return "tree " + std::to_string(h.id); }

// Offset of the element of A spanned by the corners shared with B: +-1 in the directions
// in which all shared corners are on the same side of A
CellCentOffsets SharedOffset(const shared_t &shared) {
  std::array<int, 3> off{0, 0, 0};
  for (int d = 0; d < 3; ++d) {
    bool all_lo = true, all_hi = true;
    for (auto &[ca, cb] : shared) {
      all_lo = all_lo && (Bit(ca, d) == 0);
      all_hi = all_hi && (Bit(ca, d) == 1);
    }
    off[d] = all_hi ? 1 : (all_lo ? -1 : 0);
  }
  return CellCentOffsets(off[0], off[1], off[2]);
}

// True if ct takes every shared corner of A to the same corner of B
bool MapsSharedCorners(const LogicalCoordinateTransformation &ct, const shared_t &shared) {
  for (auto &[ca, cb] : shared) {
    for (int d = 0; d < 3; ++d) {
      const int x = Bit(ca, d) - ct.offset[d];
      const int y = ct.dir_flip[d] ? 1 - x : x;
      if (y != Bit(cb, ct.dir_connection[d])) return false;
    }
  }
  return true;
}

bool SameTransform(const LogicalCoordinateTransformation &a,
                   const LogicalCoordinateTransformation &b) {
  for (int d = 0; d < 3; ++d) {
    if (a.dir_connection[d] != b.dir_connection[d] || a.dir_flip[d] != b.dir_flip[d] ||
        a.offset[d] != b.offset[d])
      return false;
  }
  return true;
}

// The unique permutation and flip of the directions, with offset off, that glues A to B
// along their shared face
LogicalCoordinateTransformation FaceTransform(const CellCentOffsets &off,
                                              const shared_t &shared, const Hex &a,
                                              const Hex &b) {
  const std::array<std::array<int, 3>, 6> perms{
      {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}}};
  std::vector<LogicalCoordinateTransformation> found;
  for (auto &p : perms) {
    for (int flips = 0; flips < 8; ++flips) {
      LogicalCoordinateTransformation ct;
      for (int d = 0; d < 3; ++d) {
        ct.dir_connection[d] = p[d];
        ct.dir_connection_inverse[p[d]] = d;
        ct.dir_flip[d] = Bit(flips, d);
        ct.offset[d] = static_cast<int>(off[d]);
      }
      ct.use_offset = true;
      ct.UpdateIsIdentity();
      if (MapsSharedCorners(ct, shared)) found.push_back(ct);
    }
  }
  PARTHENON_REQUIRE(found.size() == 1, "The shared face of " + HexName(a) + " and " +
                                           HexName(b) + " does not fix a unique " +
                                           "logical coordinate transformation.");
  return found[0];
}

} // namespace

std::vector<std::vector<HexNeighbor>> FindHexNeighbors(const std::vector<Hex> &hexes) {
  const int nhex = hexes.size();
  std::map<std::int64_t, std::vector<std::pair<int, int>>> node_to_hex; // (hex, corner)
  for (int h = 0; h < nhex; ++h) {
    std::set<std::int64_t> unique(hexes[h].nodes.begin(), hexes[h].nodes.end());
    PARTHENON_REQUIRE(unique.size() == 8,
                      HexName(hexes[h]) + " must have eight distinct nodes.");
    for (int c = 0; c < 8; ++c)
      node_to_hex[hexes[h].nodes[c]].push_back({h, c});
  }

  // Shared corners of every pair of hexes that touch
  std::vector<std::map<int, shared_t>> touching(nhex);
  for (auto &[node, hcs] : node_to_hex) {
    for (auto &[ha, ca] : hcs) {
      for (auto &[hb, cb] : hcs) {
        if (ha != hb) touching[ha][hb].push_back({ca, cb});
      }
    }
  }

  std::vector<std::vector<HexNeighbor>> neighbors(nhex);
  // Index into neighbors[h] of each neighbor of h
  std::vector<std::map<int, int>> slot(nhex);
  auto add = [&](int ha, int hb, const CellCentOffsets &off,
                 const LogicalCoordinateTransformation &ct) {
    slot[ha][hb] = neighbors[ha].size();
    neighbors[ha].push_back(HexNeighbor{hb, off, ct});
  };

  // Face neighbors follow directly from the shared nodes
  for (int ha = 0; ha < nhex; ++ha) {
    std::set<int> faces_used;
    for (auto &[hb, shared] : touching[ha]) {
      const int ns = shared.size();
      PARTHENON_REQUIRE(ns == 1 || ns == 2 || ns == 4,
                        HexName(hexes[ha]) + " and " + HexName(hexes[hb]) +
                            " must share a face, an edge or a node.");
      const auto off = SharedOffset(shared);
      const int nnormal = std::abs(static_cast<int>(off[0])) +
                          std::abs(static_cast<int>(off[1])) +
                          std::abs(static_cast<int>(off[2]));
      PARTHENON_REQUIRE((ns == 4 && nnormal == 1) || (ns == 2 && nnormal == 2) ||
                            (ns == 1 && nnormal == 3),
                        "The nodes shared by " + HexName(hexes[ha]) + " and " +
                            HexName(hexes[hb]) + " are not a face, edge or node.");
      if (ns != 4) continue;
      PARTHENON_REQUIRE(faces_used.insert(off.GetIdx()).second,
                        HexName(hexes[ha]) + " has two neighbors on one face.");
      add(ha, hb, off, FaceTransform(off, shared, hexes[ha], hexes[hb]));
    }
  }

  // Edge and then node neighbors: compose the transformation to a neighbor F of A (face,
  // or edge for node neighbors) with that from F to its face neighbor B
  for (const int nshared : {2, 1}) {
    for (int ha = 0; ha < nhex; ++ha) {
      std::vector<HexNeighbor> found;
      for (auto &[hb, shared] : touching[ha]) {
        if (shared.size() != nshared) continue;
        const auto off = SharedOffset(shared);
        std::optional<LogicalCoordinateTransformation> ct;
        for (auto &nf : neighbors[ha]) {
          if (!slot[nf.index].count(hb)) continue;
          const auto &nb = neighbors[nf.index][slot[nf.index][hb]];
          if (!nb.offset.IsFace()) continue;
          auto cand = ComposeTransformations(nf.ct, nb.ct);
          bool offset_ok = true;
          for (int d = 0; d < 3; ++d)
            offset_ok = offset_ok && (cand.offset[d] == static_cast<int>(off[d]));
          if (!offset_ok || !MapsSharedCorners(cand, shared)) continue;
          if (!ct) {
            ct = cand;
          } else if (nshared == 2) {
            // Around an edge of valence four the two paths must agree
            PARTHENON_REQUIRE(SameTransform(*ct, cand),
                              "Inconsistent transformations around the edge shared by " +
                                  HexName(hexes[ha]) + " and " + HexName(hexes[hb]) +
                                  ".");
          }
        }
        PARTHENON_REQUIRE(ct, "No common face neighbor fixes the transformation from " +
                                  HexName(hexes[ha]) + " to " + HexName(hexes[hb]) +
                                  " (edges of valence > 4 are not supported).");
        found.push_back(HexNeighbor{hb, off, *ct});
      }
      for (auto &n : found)
        add(ha, n.index, n.offset, n.ct);
    }
  }
  return neighbors;
}

std::vector<std::array<BoundaryFlag, BOUNDARY_NFACES>>
GetHexBCs(const std::vector<Hex> &hexes,
          const std::vector<std::vector<HexNeighbor>> &neighbors,
          const std::vector<HexBC> &bcs) {
  std::map<std::array<std::int64_t, 4>, BoundaryFlag> bc_map;
  for (auto &bc : bcs) {
    auto key = bc.nodes;
    std::sort(key.begin(), key.end());
    bc_map[key] = bc.bflag;
  }

  std::vector<std::array<BoundaryFlag, BOUNDARY_NFACES>> out(hexes.size());
  for (int h = 0; h < hexes.size(); ++h) {
    std::array<bool, BOUNDARY_NFACES> shared;
    shared.fill(false);
    for (auto &n : neighbors[h]) {
      if (n.offset.IsFace()) shared[n.offset.Face()] = true;
    }
    for (int d = 0; d < 3; ++d) {
      for (int side = 0; side < 2; ++side) {
        const int f = 2 * d + side;
        if (shared[f]) {
          out[h][f] = BoundaryFlag::block;
          continue;
        }
        std::array<std::int64_t, 4> key;
        int nk = 0;
        for (int c = 0; c < 8; ++c) {
          if (Bit(c, d) == side) key[nk++] = hexes[h].nodes[c];
        }
        std::sort(key.begin(), key.end());
        PARTHENON_REQUIRE(bc_map.count(key), "Face " + std::to_string(f) + " of " +
                                                 HexName(hexes[h]) +
                                                 " has no neighbor or boundary condition.");
        out[h][f] = bc_map[key];
      }
    }
  }
  return out;
}

} // namespace forest
} // namespace parthenon
