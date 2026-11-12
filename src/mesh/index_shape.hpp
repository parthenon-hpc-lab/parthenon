#ifndef MESH_INDEX_SHAPE_HPP_
#define MESH_INDEX_SHAPE_HPP_

#include <algorithm>
#include <array>

#include "defs.hpp"
#include "globals.hpp"
#include "mesh/domain.hpp"

namespace parthenon {

inline std::array<IndexShape, 3>
GetIndexShapes(const int nx1, const int nx2, const int nx3, const bool multilevel,
               const std::array<bool, 3> &symmetry = {false, false, false}) {
  IndexShape cellbounds(nx3, nx2, nx1, Globals::nghost);
  IndexShape f_cellbounds(2 * nx3, 2 * nx2, 2 * nx1, Globals::nghost);
  IndexShape c_cellbounds(nx3 / 2, nx2 / 2, nx1 / 2, 0);
  if (multilevel) {
    const int cnx1 = symmetry[0] || nx1 == 0 ? 0 : std::max(1, nx1 / 2);
    const int cnx2 = symmetry[1] || nx2 == 0 ? 0 : std::max(1, nx2 / 2);
    const int cnx3 = symmetry[2] || nx3 == 0 ? 0 : std::max(1, nx3 / 2);
    c_cellbounds = IndexShape(cnx3, cnx2, cnx1, Globals::nghost);
  }
  return {cellbounds, f_cellbounds, c_cellbounds};
}

inline std::array<IndexShape, 3> GetIndexShapes(const RegionSize &block_size,
                                                const bool multilevel = true) {
  const std::array<bool, 3> symmetry{
      block_size.symmetry(X1DIR), block_size.symmetry(X2DIR), block_size.symmetry(X3DIR)};
  return GetIndexShapes(symmetry[0] ? 0 : block_size.nx(X1DIR),
                        symmetry[1] ? 0 : block_size.nx(X2DIR),
                        symmetry[2] ? 0 : block_size.nx(X3DIR), multilevel, symmetry);
}

} // namespace parthenon

#endif // MESH_INDEX_SHAPE_HPP_
