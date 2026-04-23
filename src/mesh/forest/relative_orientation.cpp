//========================================================================================
// (C) (or copyright) 2024. Triad National Security, LLC. All rights reserved.
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

#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stack>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "basic_types.hpp"
#include "defs.hpp"
#include "mesh/forest/logical_location.hpp"
#include "mesh/forest/relative_orientation.hpp"
#include "mesh/forest/tree.hpp"
#include "utils/bit_hacks.hpp"
#include "utils/indexer.hpp"

namespace parthenon {
namespace forest {

LogicalLocation RelativeOrientation::Transform(const LogicalLocation &loc_in,
                                               std::int64_t destination) const {
  std::array<std::int64_t, 3> l_out;
  int nblock = 1LL << std::max(loc_in.level(), 0);
  for (int dir = 0; dir < 3; ++dir) {
    std::int64_t l_in = loc_in.l(dir);
    // First shift the logical location index back into the interior
    // of a bordering tree assuming they have the same coordinate
    // orientation
    // TODO(LFR): Probably remove the offset option and assume it is always true
    if (use_offset) {
      l_in -= offset[dir] * nblock;
    } else {
      l_in = (l_in + nblock) % nblock;
    }
    // Then permute (and possibly flip) the coordinate indices
    // to move to the logical coordinate system of the new tree
    if (dir_flip[dir]) {
      l_out[abs(dir_connection[dir])] = nblock - 1 - l_in;
    } else {
      l_out[abs(dir_connection[dir])] = l_in;
    }
  }
  return LogicalLocation(destination, loc_in.level(), l_out[0], l_out[1], l_out[2]);
}

LogicalLocation RelativeOrientation::TransformBack(const LogicalLocation &loc_in,
                                                   std::int64_t origin) const {
  std::array<std::int64_t, 3> l_out;
  int nblock = 1LL << std::max(loc_in.level(), 0);
  for (int dir = 0; dir < 3; ++dir) {
    std::int64_t l_in = loc_in.l(abs(dir_connection[dir]));

    // Then permute (and possibly flip) the coordinate indices
    // to move to the logical coordinate system of the new tree
    if (dir_flip[dir]) {
      l_out[dir] = nblock - 1 - l_in;
    } else {
      l_out[dir] = l_in;
    }

    // First shift the logical location index back into the interior
    // of a bordering tree assuming they have the same coordinate
    // orientation
    // TODO(LFR): Probably remove the offset option and assume it is always true
    if (use_offset) {
      l_out[dir] += offset[dir] * nblock;
    } else {
      l_out[dir] = (l_out[dir] + nblock) % nblock;
    }
  }
  return LogicalLocation(origin, loc_in.level(), l_out[0], l_out[1], l_out[2]);
}
} // namespace forest
} // namespace parthenon
