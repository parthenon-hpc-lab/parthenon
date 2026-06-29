//========================================================================================
// (C) (or copyright) 2023. Triad National Security, LLC. All rights reserved.
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
#ifndef MESH_FOREST_FOREST_NODE_HPP_
#define MESH_FOREST_FOREST_NODE_HPP_

#include <array>
#include <map>
#include <memory>
#include <set>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "basic_types.hpp"
#include "defs.hpp"
#include "utils/hash.hpp"

namespace parthenon {
namespace forest {

class Face;
class Node {
 public:
  Node(int id_in, std::array<Real, NDIM> pos) : id(id_in), x(pos), associated_faces{} {}

  static std::shared_ptr<Node> create(int id, std::array<Real, NDIM> pos) {
    return std::make_shared<Node>(id, pos);
  }

  std::uint32_t id;
  std::array<Real, NDIM> x;
  std::unordered_set<std::weak_ptr<Face>, WeakPtrHash<Face>, WeakPtrEqual<Face>>
      associated_faces;
};
} // namespace forest
} // namespace parthenon

#endif // MESH_FOREST_FOREST_NODE_HPP_
