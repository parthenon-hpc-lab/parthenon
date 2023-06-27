//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
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
#ifndef DEFS_HPP_
#define DEFS_HPP_
//! \file defs.hpp
//  \brief contains Athena++ general purpose types, structures, enums, etc.

#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "basic_types.hpp"
#include "config.hpp"
#include "parthenon_arrays.hpp"

namespace parthenon {

// for OpenMP 4.0 SIMD vectorization, control width of SIMD lanes
#if defined(__AVX512F__)
#define SIMD_WIDTH 8
#elif defined(__AVX__)
#define SIMD_WIDTH 4
#elif defined(__SSE2__)
#define SIMD_WIDTH 2
#else
#define SIMD_WIDTH 4
#endif

#define CACHELINE_BYTES 64

#define NMAX_NEIGHBORS 56

// forward declarations needed for function pointer type aliases
class MeshBlock;
class ParameterInput;

template <int NDIM = 3>
constexpr uint64_t GetInterleaveConstant(int power) {
  // For power = 2, NDIM = 3, this should return
  // ...000011000011
  // For power = 1, NDIM = 3, this should return
  // ...001001001001
  // For power = 2, NDIM = 2, this should return
  // ...001100110011
  // etc.
  uint64_t i_const = ~((~static_cast<uint64_t>(0)) << power); // std::pow(2, power) - 1;
  int cur_shift =
      sizeof(uint64_t) * 8 * NDIM; // Works for anything that will fit in uint64_t
  while (cur_shift >= NDIM * power) {
    i_const = (i_const << cur_shift) | i_const;
    cur_shift /= 2;
  }
  return i_const;
}

template <int NDIM = 3, int N_VALID_BITS = 21>
uint64_t InterleaveZeros(uint64_t x) {
  // This is a standard bithack for interleaving zeros in binary numbers to make a Morton
  // number
  if constexpr (N_VALID_BITS >= 64)
    x = (x | x << 64 * (NDIM - 1)) & GetInterleaveConstant<NDIM>(64);
  if constexpr (N_VALID_BITS >= 32)
    x = (x | x << 32 * (NDIM - 1)) & GetInterleaveConstant<NDIM>(32);
  if constexpr (N_VALID_BITS >= 16)
    x = (x | x << 16 * (NDIM - 1)) & GetInterleaveConstant<NDIM>(16);
  if constexpr (N_VALID_BITS >= 8)
    x = (x | x << 8 * (NDIM - 1)) & GetInterleaveConstant<NDIM>(8);
  if constexpr (N_VALID_BITS >= 4)
    x = (x | x << 4 * (NDIM - 1)) & GetInterleaveConstant<NDIM>(4);
  if constexpr (N_VALID_BITS >= 2)
    x = (x | x << 2 * (NDIM - 1)) & GetInterleaveConstant<NDIM>(2);
  if constexpr (N_VALID_BITS >= 1)
    x = (x | x << 1 * (NDIM - 1)) & GetInterleaveConstant<NDIM>(1);
  return x;
}

struct MortonNumber {
  uint64_t most, mid, least;

  MortonNumber(int level, uint64_t x, uint64_t y, uint64_t z)
      : most(GetMortonBits(level, x, y, z, 2)), mid(GetMortonBits(level, x, y, z, 1)),
        least(GetMortonBits(level, x, y, z, 0)) {}

 private:
  uint64_t GetMortonBits(int level, uint64_t x, uint64_t y, uint64_t z, int chunk) {
    constexpr int NBITS = 21;
    constexpr uint64_t lowest_nbits_mask = ~((~static_cast<uint64_t>(0)) << NBITS);

    // Shift the by level location to the global location
    x = x << (NBITS - level);
    y = y << (NBITS - level);
    z = z << (NBITS - level);

    // Get the chunk signifigance NBITS bits of each direction
    x = x >> (chunk * NBITS) & lowest_nbits_mask;
    y = y >> (chunk * NBITS) & lowest_nbits_mask;
    z = z >> (chunk * NBITS) & lowest_nbits_mask;

    // Return the interleaved section of the morton number
    return InterleaveZeros<3, NBITS>(z) << 2 | InterleaveZeros<3, NBITS>(y) << 1 |
           InterleaveZeros<3, NBITS>(x);
  }
};

inline bool operator<(const MortonNumber &lhs, const MortonNumber &rhs) {
  if (lhs.most == rhs.most && lhs.mid == rhs.mid) return lhs.least < rhs.least;
  if (lhs.most == rhs.most) return lhs.mid < rhs.mid;
  return lhs.most < rhs.most;
}

inline bool operator>(const MortonNumber &lhs, const MortonNumber &rhs) {
  if (lhs.most == rhs.most && lhs.mid == rhs.mid) return lhs.least > rhs.least;
  if (lhs.most == rhs.most) return lhs.mid > rhs.mid;
  return lhs.most > rhs.most;
}

inline bool operator==(const MortonNumber &lhs, const MortonNumber &rhs) {
  return (lhs.most == rhs.most) && (lhs.mid == rhs.mid) && (lhs.least == rhs.least);
}

//--------------------------------------------------------------------------------------
//! \struct LogicalLocation
//  \brief stores logical location and level of MeshBlock

struct LogicalLocation { // aggregate and POD type
  // These values can exceed the range of std::int32_t even if the root grid has only a
  // single MeshBlock if >30 levels of AMR are used, since the corresponding max index =
  // 1*2^31 > INT_MAX = 2^31 -1 for most 32-bit signed integer type impelementations

 private:
  std::int64_t lx1_, lx2_, lx3_;
  MortonNumber
      morton_; // Morton number needs to have the same number of bits as lx1 through lx3
  int level_;

 public:
  LogicalLocation(int lev, std::int64_t l1, std::int64_t l2, std::int64_t l3)
      : lx1_(l1), lx2_(l2), lx3_(l3), level_(lev), morton_(lev, l1, l2, l3) {}
  LogicalLocation() : LogicalLocation(0, 0, 0, 0) {}

  const auto &lx1() const { return lx1_; }
  const auto &lx2() const { return lx2_; }
  const auto &lx3() const { return lx3_; }
  const auto &level() const { return level_; }
  const auto &morton() const { return morton_; }

  // operators useful for sorting
  bool operator==(LogicalLocation &ll) {
    return ((ll.level() == level_) && (ll.lx1() == lx1_) && (ll.lx2() == lx2_) &&
            (ll.lx3() == lx3_));
  }
  static bool Lesser(const LogicalLocation &left, const LogicalLocation &right) {
    return left.level() < right.level();
  }
  static bool Greater(const LogicalLocation &left, const LogicalLocation &right) {
    return left.level() > right.level();
  }

  bool IsContainedIn(const LogicalLocation &container) const {
    if (container.level() > level_) return false;
    const int shifted_lx1 = lx1_ >> (level_ - container.level());
    const int shifted_lx2 = lx2_ >> (level_ - container.level());
    const int shifted_lx3 = lx3_ >> (level_ - container.level());
    return (shifted_lx1 == container.lx1()) && (shifted_lx2 == container.lx2()) &&
           (shifted_lx3 == container.lx3());
  }

  bool Contains(const LogicalLocation &containee) const {
    if (containee.level() < level_) return false;
    const int shifted_lx1 = containee.lx1() >> (containee.level() - level_);
    const int shifted_lx2 = containee.lx2() >> (containee.level() - level_);
    const int shifted_lx3 = containee.lx3() >> (containee.level() - level_);
    return (shifted_lx1 == lx1_) && (shifted_lx2 == lx2_) && (shifted_lx3 == lx3_);
  }

  LogicalLocation GetSameLevelNeighbor(int ox1, int ox2, int ox3) const {
    return LogicalLocation(level_, lx1_ + ox1, lx2_ + ox2, lx3_ + ox3);
  }

  LogicalLocation GetParent() const {
    if (level_ == 0) return *this;
    return LogicalLocation(level_ - 1, lx1_ >> 1, lx2_ >> 1, lx3_ >> 1);
  }

  std::vector<LogicalLocation> GetDaughters() const {
    std::vector<LogicalLocation> daughters;
    for (int i : {0, 1}) {
      for (int j : {0, 1}) {
        for (int k : {0, 1}) {
          daughters.push_back(GetDaughter(i, j, k));
        }
      }
    }
    return daughters;
  }

  LogicalLocation GetDaughter(int ox1, int ox2, int ox3) const {
    return LogicalLocation(level_ + 1, (lx1_ << 1) + ox1, (lx2_ << 1) + ox2,
                           (lx3_ << 1) + ox3);
  }
};

inline bool operator<(const LogicalLocation &lhs, const LogicalLocation &rhs) {
  if (lhs.morton() == rhs.morton()) return lhs.level() < rhs.level();
  return lhs.morton() < rhs.morton();
}

inline bool operator>(const LogicalLocation &lhs, const LogicalLocation &rhs) {
  if (lhs.morton() == rhs.morton()) return lhs.level() > rhs.level();
  return lhs.morton() > rhs.morton();
}

/// Defines the maximum size of the static array used in the IndexShape objects
constexpr int NDIM = 3;
static_assert(NDIM >= 3,
              "IndexShape cannot be used when NDIM is set to a value less than 3");
//----------------------------------------------------------------------------------------
//! \struct RegionSize
//  \brief physical size and number of cells in a Mesh or a MeshBlock

struct RegionSize { // aggregate and POD type; do NOT reorder member declarations:
  Real x1min, x2min, x3min;
  Real x1max, x2max, x3max;
  Real x1rat, x2rat, x3rat; // ratio of dxf(i)/dxf(i-1)
  // the size of the root grid or a MeshBlock should not exceed std::int32_t limits
  int nx1, nx2, nx3; // number of active cells (not including ghost zones)
};

//----------------------------------------------------------------------------------------
// enums used everywhere
// (not specifying underlying integral type (C++11) for portability & performance)

// TODO(felker): C++ Core Guidelines Enum.5: Don’t use ALL_CAPS for enumerators
// (avoid clashes with preprocessor macros). Enumerated type definitions in this file and:
// io_wrapper.hpp, bvals.hpp, field_diffusion.hpp,
// task_list.hpp, ???

//------------------
// named, weakly typed / unscoped enums:
//------------------

// needed for arrays dimensioned over grid directions
// enumerator type only used in Mesh::EnrollUserMeshGenerator()
// X0DIR time-like direction
// X1DIR x, r, etc...
// X2DIR y, theta, etc...
// X3DIR z, phi, etc...
enum CoordinateDirection { NODIR = -1, X0DIR = 0, X1DIR = 1, X2DIR = 2, X3DIR = 3 };

// identifiers for all 6 faces of a MeshBlock
constexpr int BOUNDARY_NFACES = 6;
enum BoundaryFace {
  undef = -1,
  inner_x1 = 0,
  outer_x1 = 1,
  inner_x2 = 2,
  outer_x2 = 3,
  inner_x3 = 4,
  outer_x3 = 5
};

//------------------
// strongly typed / scoped enums (C++11):
//------------------
// KGF: Except for the 2x MG* enums, these may be unnessary w/ the new class inheritance
// Now, only passed to BoundaryVariable::InitBoundaryData(); could replace w/ bool switch
enum class BoundaryQuantity { cc, fc, cc_flcor, fc_flcor };
enum class BoundaryCommSubset { mesh_init, all };
// TODO(felker): consider generalizing/renaming to QuantityFormulation
enum class UserHistoryOperation { sum, max, min };

//----------------------------------------------------------------------------------------
// function pointer prototypes for user-defined modules set at runtime

using MeshGenFunc = Real (*)(Real x, RegionSize rs);
using HistoryOutputFunc = Real (*)(MeshBlock *pmb, int iout);
using MGBoundaryFunc = void (*)(ParArrayND<Real> &dst, Real time, int nvar, int is,
                                int ie, int js, int je, int ks, int ke, int ngh, Real x0,
                                Real y0, Real z0, Real dx, Real dy, Real dz);

//----------------------------------------------------------------------------------------
// Opaque pointer to application data
class MeshBlockApplicationData {
 public:
  // make this pure virtual so that this class cannot be instantiated
  // (only derived classes can be instantiated)
  virtual ~MeshBlockApplicationData() = 0;
};
using pMeshBlockApplicationData_t = std::unique_ptr<MeshBlockApplicationData>;

// we still need to define this somewhere, though
inline MeshBlockApplicationData::~MeshBlockApplicationData() {}

// Convience definitions
constexpr uint64_t GiB = 1024 * 1024 * 1024;
constexpr uint64_t MiB = 1024 * 1024;
constexpr uint64_t KiB = 1024;

} // namespace parthenon

#endif // DEFS_HPP_
