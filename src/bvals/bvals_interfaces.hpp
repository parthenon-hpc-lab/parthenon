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
#ifndef BVALS_BVALS_INTERFACES_HPP_
#define BVALS_BVALS_INTERFACES_HPP_
//! \file bvals_interfaces.hpp
//  \brief defines enums, structs, and abstract classes

// TODO(felker): deduplicate forward declarations
// TODO(felker): consider moving enums and structs in a new file? bvals_structs.hpp?

#include <memory>
#include <string>
#include <vector>

#include "parthenon_mpi.hpp"

#include "defs.hpp"
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

// identifiers for status of MPI boundary communications
enum class BoundaryStatus { waiting, arrived, completed };

//----------------------------------------------------------------------------------------
//! \struct SimpleNeighborBlock
//  \brief Struct storing only the basic info about a MeshBlocks neighbors. Typically used
//  for convenience to store redundant info from subset of the more complete NeighborBlock
//  objects, e.g. for describing neighbors around pole at same radius and polar angle

struct SimpleNeighborBlock { // aggregate and POD
  int rank;                  // MPI rank of neighbor
  int level;                 // refinement (logical, not physical) level of neighbor
  int lid;                   // local ID of neighbor
  int gid;                   // global ID of neighbor
  bool operator==(const SimpleNeighborBlock &rhs) const {
    return (rank == rhs.rank) && (level == rhs.level) && (gid == rhs.gid);
  }
};

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

class BufferID {
  std::vector<NeighborIndexes> nis;
 public: 
  BufferID(int dim, bool multilevel) { 
    std::vector<int> x1offsets = dim > 0 ? std::vector<int>{0, -1, 1} : std::vector<int>{0};
    std::vector<int> x2offsets = dim > 1 ? std::vector<int>{0, -1, 1} : std::vector<int>{0};
    std::vector<int> x3offsets = dim > 2 ? std::vector<int>{0, -1, 1} : std::vector<int>{0};
    for (auto ox3 : x3offsets) { 
      for (auto ox2 : x2offsets) { 
        for (auto ox1 : x1offsets) { 
          const int type = std::abs(ox1) + std::abs(ox2) + std::abs(ox3);
          if (type == 0) continue;
          std::vector<int> f1s = (dim - type) > 0 && multilevel ? std::vector<int>{0, 1} : std::vector<int>{0};
          std::vector<int> f2s = (dim - type) > 1 && multilevel ? std::vector<int>{0, 1} : std::vector<int>{0};
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

  int GetID(int ox1, int ox2, int ox3, int f1, int f2) const { 
    NeighborIndexes in{ox1, ox2, ox3, f1, f2, NeighborConnect::face};
    for (int i = 0; i < nis.size(); ++i) { 
      if (nis[i] == in) return i;
    }
    return -1;
  }

  int size() const {return nis.size();}
};

//----------------------------------------------------------------------------------------
//! \struct NeighborBlock
//  \brief

struct NeighborBlock { // aggregate and POD type. Inheritance breaks standard-layout-> POD
                       // : SimpleNeighborBlock, NeighborIndexes {
  // composition:
  SimpleNeighborBlock snb;
  NeighborIndexes ni;

  int bufid, eid, targetid;
  BoundaryFace fid;
  LogicalLocation loc;
  block_ownership_t ownership;
  RegionSize block_size;

  void SetNeighbor(LogicalLocation inloc, int irank, int ilevel, int igid, int ilid,
                   int iox1, int iox2, int iox3, NeighborConnect itype, int ibid,
                   int itargetid, int ifi1 = 0, int ifi2 = 0);
  NeighborBlock() = default;
  NeighborBlock(Mesh *mesh, LogicalLocation loc, int rank, int gid, int lid,
                std::array<int, 3> offsets, NeighborConnect type, int bid, int target_id,
                int ifi1, int ifi2);
  NeighborBlock(Mesh *mesh, LogicalLocation loc, int rank, int gid,
                std::array<int, 3> offsets, int bufid_in, int ifi1, int ifi2);
};

//----------------------------------------------------------------------------------------
//! \struct BoundaryData
//  \brief structure storing boundary information

template <int n = NMAX_NEIGHBORS>
struct BoundaryData { // aggregate and POD (even when MPI_PARALLEL is defined)
  static constexpr int kMaxNeighbor = n;
  // KGF: "nbmax" only used in bvals_var.cpp, Init/DestroyBoundaryData()
  int nbmax; // actual maximum number of neighboring MeshBlocks
  // currently, sflag[] is only used by Multgrid (send buffers are reused each stage in
  // red-black comm. pattern; need to check if they are available)
  BoundaryStatus flag[kMaxNeighbor], sflag[kMaxNeighbor];
  BufArray1D<Real> buffers;
  BufArray1D<Real> send[kMaxNeighbor], recv[kMaxNeighbor];
  // host mirror view of recv
  BufArray1D<Real>::host_mirror_type recv_h[kMaxNeighbor];
  int recv_size[kMaxNeighbor];
#ifdef MPI_PARALLEL
  MPI_Request req_send[kMaxNeighbor], req_recv[kMaxNeighbor];
#endif
};

//----------------------------------------------------------------------------------------
// Interfaces = abstract classes containing ONLY pure virtual functions
//              Merely lists functions and their argument lists that must be implemented
//              in derived classes to form a somewhat-strict contract for functionality
//              (can always implement as a do-nothing/no-op function if a derived class
//              instance is the exception to the rule and does not use a particular
//              interface function)
//----------------------------------------------------------------------------------------

//----------------------------------------------------------------------------------------
//! \class BoundaryCommunication
//  \brief contains methods for managing BoundaryStatus flags and MPI requests

class BoundaryCommunication {
 public:
  BoundaryCommunication() {}
  virtual ~BoundaryCommunication() {}
  // create unique tags for each MeshBlock/buffer/quantity and initialize MPI requests:
  virtual void SetupPersistentMPI() = 0;
  // call MPI_Start() on req_recv[]
  virtual void StartReceiving(BoundaryCommSubset phase) = 0;
  // call MPI_Wait() on req_send[] and set flag[] to BoundaryStatus::waiting
  virtual void ClearBoundary(BoundaryCommSubset phase) = 0;
};

//----------------------------------------------------------------------------------------
// Concrete classes
//----------------------------------------------------------------------------------------

//----------------------------------------------------------------------------------------
//! \class BoundarySwarm
class BoundarySwarm : public BoundaryCommunication {
 public:
  explicit BoundarySwarm(std::weak_ptr<MeshBlock> pmb, const std::string &label);
  ~BoundarySwarm() = default;

  std::vector<ParArrayND<int>> vars_int;
  std::vector<ParArrayND<Real>> vars_real;

  // (usuallly the std::size_t unsigned integer type)
  std::vector<BoundaryCommunication *>::size_type bswarm_index;

  // BoundaryCommunication
  void SetupPersistentMPI() final;
  void StartReceiving(BoundaryCommSubset phase) final{};
  void ClearBoundary(BoundaryCommSubset phase) final{};
  void Receive(BoundaryCommSubset phase);
  void Send(BoundaryCommSubset phase);

  BoundaryData<> bd_var_;
  std::weak_ptr<MeshBlock> pmy_block;
  Mesh *pmy_mesh_;
  int send_tag[NMAX_NEIGHBORS], recv_tag[NMAX_NEIGHBORS];
  int particle_size, send_size[NMAX_NEIGHBORS], recv_size[NMAX_NEIGHBORS];

 protected:
  int nl_, nu_;
  void InitBoundaryData(BoundaryData<> &bd);

 private:
  std::shared_ptr<MeshBlock> GetBlockPointer() {
    if (pmy_block.expired()) {
      PARTHENON_THROW("Invalid pointer to MeshBlock!");
    }
    return pmy_block.lock();
  }

#ifdef MPI_PARALLEL
  // Unique communicator for this swarm.
  MPI_Comm swarm_comm;
#endif
};

} // namespace parthenon

#endif // BVALS_BVALS_INTERFACES_HPP_
