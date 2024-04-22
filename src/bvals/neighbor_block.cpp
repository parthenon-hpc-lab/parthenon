//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2023. Triad National Security, LLC. All rights reserved.
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
//! \file neighbor_block.cpp
//  \brief utility functions for neighbors and buffers

#include "bvals/neighbor_block.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring> // memcpy()
#include <iomanip>
#include <iostream>  // endl
#include <sstream>   // stringstream
#include <stdexcept> // runtime_error
#include <string>    // c_str()
#include <unordered_set>

#include "globals.hpp"
#include "mesh/forest/logical_location.hpp"
#include "mesh/mesh.hpp"
#include "utils/buffer_utils.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

NeighborBlock::NeighborBlock()
    : rank{-1}, gid{-1}, bufid{-1}, targetid{-1}, loc(), fi1{-1}, fi2{-1}, block_size(),
      offsets(0, 0, 0), ownership(true) {}

NeighborBlock::NeighborBlock(Mesh *mesh, LogicalLocation loc, int rank, int gid,
                             std::array<int, 3> offsets_in, int bid, int target_id,
                             int fi1, int fi2)
    : rank{rank}, gid{gid}, bufid{bid}, targetid{target_id}, loc{loc}, fi1{fi1}, fi2{fi2},
      block_size(mesh->GetBlockSize(loc)), offsets(offsets_in), ownership(true) {}

BufferID::BufferID(int dim, bool multilevel) {
  std::vector<int> x1offsets = dim > 0 ? std::vector<int>{0, -1, 1} : std::vector<int>{0};
  std::vector<int> x2offsets = dim > 1 ? std::vector<int>{0, -1, 1} : std::vector<int>{0};
  std::vector<int> x3offsets = dim > 2 ? std::vector<int>{0, -1, 1} : std::vector<int>{0};
  for (auto ox3 : x3offsets) {
    for (auto ox2 : x2offsets) {
      for (auto ox1 : x1offsets) {
        const int type = std::abs(ox1) + std::abs(ox2) + std::abs(ox3);
        if (type == 0) continue;
        std::vector<int> f1s =
            (dim - type) > 0 && multilevel ? std::vector<int>{0, 1} : std::vector<int>{0};
        std::vector<int> f2s =
            (dim - type) > 1 && multilevel ? std::vector<int>{0, 1} : std::vector<int>{0};
        for (auto f1 : f1s) {
          for (auto f2 : f2s) {
            NeighborIndexes ni{ox1, ox2, ox3, f1, f2};
            nis.push_back(ni);
          }
        }
      }
    }
  }
}

} // namespace parthenon
