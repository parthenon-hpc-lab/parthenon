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
#include "utils/error_checking.hpp"

namespace parthenon {

// forward declarations
class Mesh;
class MeshBlock;
class MeshBlockTree;
class Field;
class ParameterInput;
struct RegionSize;

// TODO(felker): nest these enum definitions inside bvals/ classes, when possible.

// DEPRECATED(felker): maintain old-style (ALL_CAPS) enumerators as unscoped,unnamed types
// Keep for compatibility with user-provided pgen/ files. Use only new types internally.

// GCC 6 added Enumerator Attr (v6.1 released on 2016-04-27)
// TODO(felker): replace with C++14 [[deprecated]] attributes if we ever bump --std=c++14
#if (defined(__GNUC__) && __GNUC__ >= 6) || (defined(__clang__) && __clang_major__ >= 3)
enum {
  FACE_UNDEF __attribute__((deprecated)) = -1,
  INNER_X1 __attribute__((deprecated)),
  OUTER_X1 __attribute__((deprecated)),
  INNER_X2 __attribute__((deprecated)),
  OUTER_X2 __attribute__((deprecated)),
  INNER_X3 __attribute__((deprecated)),
  OUTER_X3 __attribute__((deprecated))
};
enum {
  BLOCK_BNDRY __attribute__((deprecated)) = -1,
  BNDRY_UNDEF __attribute__((deprecated)),
  REFLECTING_BNDRY __attribute__((deprecated)),
  OUTFLOW_BNDRY __attribute__((deprecated)),
  PERIODIC_BNDRY __attribute__((deprecated))
};
#else
enum { FACE_UNDEF = -1, INNER_X1, OUTER_X1, INNER_X2, OUTER_X2, INNER_X3, OUTER_X3 };
enum {
  BLOCK_BNDRY = -1,
  BNDRY_UNDEF,
  REFLECTING_BNDRY,
  OUTFLOW_BNDRY,
  USER_BNDRY,
  PERIODIC_BNDRY,
  POLAR_BNDRY,
  POLAR_BNDRY_WEDGE
};
#endif

// TODO(felker): BoundaryFace must be unscoped enum, for now. Its enumerators are used as
// int to index raw arrays (not ParArrayNDs)--> enumerator vals are explicitly specified

// identifiers for types of neighbor blocks (connectivity with current MeshBlock)
enum class NeighborConnect {
  none,
  face,
  edge,
  corner
}; // degenerate/shared part of block

//----------------------------------------------------------------------------------------
//! \struct NeighborConnect
//  \brief data to describe MeshBlock neighbors

struct NeighborIndexes { // aggregate and POD
  int ox1, ox2, ox3;     // 3-vec of offsets in {-1,0,+1} relative to this block's (i,j,k)
  int fi1, fi2; // 2-vec for identifying refined neighbors (up to 4x face neighbors
                // in 3D), entries in {0, 1}={smaller, larger} LogicalLocation::lxi
  NeighborConnect type;
  // User-provided ctor is unnecessary and prevents the type from being POD and aggregate.
  // This struct's implicitly-defined or defaulted default ctor is trivial, implying that
  // NeighborIndexes is a trivial type. Combined with standard layout --> POD. Advantages:
  //   - No user-provided ctor: value initialization first performs zero initialization
  //     (then default initialization if ctor is non-trivial)
  //   - Aggregate type: supports aggregate initialization {}
  //   - POD type: safely copy objects via memcpy, no memory padding in the beginning of
  //     object, C portability, supports static initialization
  bool operator==(const NeighborIndexes &rhs) const {
    return (ox1 == rhs.ox1) && (ox2 == rhs.ox2) && (ox3 == rhs.ox3) && (fi1 == rhs.fi1) &&
           (fi2 == rhs.fi2) && (type == rhs.type);
  }
};

//----------------------------------------------------------------------------------------
//! \struct NeighborBlock
//  \brief

struct NeighborBlock {
  NeighborIndexes ni;
  
  int bufid, eid, targetid;
  BoundaryFace fid;
  LogicalLocation loc;
  block_ownership_t ownership;
  RegionSize block_size;

  int rank_, gid_; 
  int rank() const { return rank_; }
  int gid() const { return gid_; }

  NeighborBlock() = default;
  NeighborBlock(Mesh *mesh, LogicalLocation loc, int rank, int gid,
                std::array<int, 3> offsets, NeighborConnect type, int bid, int target_id,
                int ifi1, int ifi2);
  NeighborBlock(Mesh *mesh, LogicalLocation loc, int rank, int gid,
                std::array<int, 3> offsets, int bid, int target_id, int ifi1, int ifi2);
};

//----------------------------------------------------------------------------------------
//! \class BufferID
//  \brief Class for determining unique indices for communication buffers based on offsets
// TODO(LFR): This is only necessary for swarm communication and can go away when that is
// updated.
class BufferID {
  std::vector<NeighborIndexes> nis;

 public:
  BufferID(int dim, bool multilevel);

  int GetID(int ox1, int ox2, int ox3, int f1, int f2) const {
    NeighborIndexes in{ox1, ox2, ox3, f1, f2, NeighborConnect::face};
    for (int i = 0; i < nis.size(); ++i) {
      if (nis[i] == in) return i;
    }
    return -1;
  }

  int size() const { return nis.size(); }
};

} // namespace parthenon

#endif // BVALS_NEIGHBOR_BLOCK_HPP_
