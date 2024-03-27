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

//----------------------------------------------------------------------------------------
// \!fn void NeighborBlock::SetNeighbor(int irank, int ilevel, int igid, int ilid,
//                          int iox1, int iox2, int iox3, NeighborConnect itype,
//                          int ibid, int itargetid, int ifi1=0, int ifi2=0)
// \brief Set neighbor information

void NeighborBlock::SetNeighbor(LogicalLocation inloc, int irank, int ilevel, int igid,
                                int ilid, int iox1, int iox2, int iox3,
                                NeighborConnect itype, int ibid, int itargetid,
                                int ifi1, // =0
                                int ifi2  // =0
) {
  snb.rank = irank;
  snb.level = ilevel;
  snb.gid = igid;
  snb.lid = ilid;
  ni.ox1 = iox1;
  ni.ox2 = iox2;
  ni.ox3 = iox3;
  ni.type = itype;
  ni.fi1 = ifi1;
  ni.fi2 = ifi2;
  bufid = ibid;
  targetid = itargetid;
  loc = inloc;
  if (ni.type == NeighborConnect::face) {
    if (ni.ox1 == -1)
      fid = BoundaryFace::inner_x1;
    else if (ni.ox1 == 1)
      fid = BoundaryFace::outer_x1;
    else if (ni.ox2 == -1)
      fid = BoundaryFace::inner_x2;
    else if (ni.ox2 == 1)
      fid = BoundaryFace::outer_x2;
    else if (ni.ox3 == -1)
      fid = BoundaryFace::inner_x3;
    else if (ni.ox3 == 1)
      fid = BoundaryFace::outer_x3;
  }
  if (ni.type == NeighborConnect::edge) {
    if (ni.ox3 == 0)
      eid = ((((ni.ox1 + 1) >> 1) | ((ni.ox2 + 1) & 2)));
    else if (ni.ox2 == 0)
      eid = (4 + (((ni.ox1 + 1) >> 1) | ((ni.ox3 + 1) & 2)));
    else if (ni.ox1 == 0)
      eid = (8 + (((ni.ox2 + 1) >> 1) | ((ni.ox3 + 1) & 2)));
  }
  return;
}

NeighborConnect NCFromOffsets(const std::array<int, 3> offsets) {
  int connect_indicator =
      std::abs(offsets[0]) + std::abs(offsets[1]) + std::abs(offsets[2]);
  NeighborConnect nc = NeighborConnect::none;
  if (connect_indicator == 1) {
    nc = NeighborConnect::face;
  } else if (connect_indicator == 2) {
    nc = NeighborConnect::edge;
  } else if (connect_indicator == 3) {
    nc = NeighborConnect::corner;
  }
  return nc;
}

NeighborBlock::NeighborBlock(Mesh *mesh, LogicalLocation loc, int rank, int gid,
                             std::array<int, 3> offsets, int ibid, int itargetid, int fi1,
                             int fi2)
    : NeighborBlock(mesh, loc, rank, gid, 0, offsets, NCFromOffsets(offsets), ibid,
                    itargetid, fi1, fi2) {}

NeighborBlock::NeighborBlock(Mesh *mesh, LogicalLocation loc, int rank, int gid, int lid,
                             std::array<int, 3> offsets, NeighborConnect type, int bid,
                             int target_id, int fi1, int fi2)
    : snb{rank, loc.level(), lid, gid}, ni{offsets[0], offsets[1], offsets[2],
                                           fi1,        fi2,        type},
      bufid{bid}, eid{0}, targetid{target_id}, fid{BoundaryFace::undef}, loc{loc},
      ownership(true), block_size(mesh->GetBlockSize(loc)) {
  // TODO(LFR): Look and see if this stuff gets used anywhere
  if (ni.type == NeighborConnect::face) {
    if (ni.ox1 == -1)
      fid = BoundaryFace::inner_x1;
    else if (ni.ox1 == 1)
      fid = BoundaryFace::outer_x1;
    else if (ni.ox2 == -1)
      fid = BoundaryFace::inner_x2;
    else if (ni.ox2 == 1)
      fid = BoundaryFace::outer_x2;
    else if (ni.ox3 == -1)
      fid = BoundaryFace::inner_x3;
    else if (ni.ox3 == 1)
      fid = BoundaryFace::outer_x3;
  }
  if (ni.type == NeighborConnect::edge) {
    if (ni.ox3 == 0)
      eid = ((((ni.ox1 + 1) >> 1) | ((ni.ox2 + 1) & 2)));
    else if (ni.ox2 == 0)
      eid = (4 + (((ni.ox1 + 1) >> 1) | ((ni.ox3 + 1) & 2)));
    else if (ni.ox1 == 0)
      eid = (8 + (((ni.ox2 + 1) >> 1) | ((ni.ox3 + 1) & 2)));
  }
}

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
            NeighborIndexes ni{ox1, ox2, ox3, f1, f2, NeighborConnect::face};
            nis.push_back(ni);
          }
        }
      }
    }
  }
}

} // namespace parthenon
