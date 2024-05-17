//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2021. Triad National Security, LLC. All rights reserved.
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
#ifndef BVALS_NEIGHBOR_BLOCK_HPP_
#define BVALS_NEIGHBOR_BLOCK_HPP_
//! \file neighbor_block.hpp
//  \brief defines enums, structs, and abstract classes

// TODO(felker): deduplicate forward declarations
// TODO(felker): consider moving enums and structs in a new file? bvals_structs.hpp?

#include <memory>
#include <string>
#include <vector>

#include "parthenon_mpi.hpp"

#include "defs.hpp"
#include "mesh/forest/block_ownership.hpp"
#include "mesh/forest/logical_location.hpp"
#include "parthenon_arrays.hpp"
#include "utils/cell_center_offsets.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

// forward declarations
class Mesh;
struct RegionSize;

//----------------------------------------------------------------------------------------
//! \struct NeighborBlock
//  \brief

struct NeighborBlock {
  // MPI rank and global id of neighbor block
  int rank, gid;
  // Swarm communication buffer identifier
  int bufid, targetid;
  // LogicalLocation of neighbor block
  LogicalLocation loc;
  // offsets of neighbor block if it is on a finer level
  // TODO(LFR): Remove these
  int fi1, fi2;
  // Size of the neighbor block
  RegionSize block_size;
  // Offset of the neighbor block relative to origin block
  CellCentOffsets offsets;
  // Ownership of neighbor block of different topological elements
  block_ownership_t ownership;

  NeighborBlock();
  NeighborBlock(Mesh *mesh, LogicalLocation loc, int rank, int gid,
                std::array<int, 3> offsets, int bid, int target_id, int ifi1, int ifi2);
};

//----------------------------------------------------------------------------------------
//! \class BufferID
//  \brief Class for determining unique indices for communication buffers based on offsets
// TODO(LFR): This is only necessary for swarm communication and can go away when that is
// updated.
class BufferID {
  // Array contains ox1, ox2, ox3, fi1, fi2
  using NeighborIndexes = std::array<int, 5>;
  std::vector<NeighborIndexes> nis;

 public:
  BufferID(int dim, bool multilevel);

  int GetID(int ox1, int ox2, int ox3, int f1, int f2) const {
    NeighborIndexes in{ox1, ox2, ox3, f1, f2};
    for (int i = 0; i < nis.size(); ++i) {
      if (nis[i] == in) return i;
    }
    return -1;
  }

  int size() const { return nis.size(); }
};

} // namespace parthenon

#endif // BVALS_NEIGHBOR_BLOCK_HPP_
