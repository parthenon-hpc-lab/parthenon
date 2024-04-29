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
#ifndef DEFS_HPP_
#define DEFS_HPP_
//! \file defs.hpp
//  \brief contains Athena++ general purpose types, structures, enums, etc.

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>

#include "basic_types.hpp"
#include "config.hpp"
#include "mesh/forest/logical_location.hpp"
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

// Define internally created variable names
inline const std::string internal_varname_seperator("::");
inline const std::string internal_fluxname("bnd_flux");

/// Defines the maximum size of the static array used in the IndexShape objects
constexpr int NDIM = 3;
static_assert(NDIM >= 3,
              "IndexShape cannot be used when NDIM is set to a value less than 3");
//----------------------------------------------------------------------------------------
//! \struct RegionSize
//  \brief physical size and number of cells in a Mesh or a MeshBlock

//------------------
// named, weakly typed / unscoped enums:
//------------------

struct RegionSize {
  RegionSize() = default;
  RegionSize(std::array<Real, 3> xmin, std::array<Real, 3> xmax, std::array<Real, 3> xrat,
             std::array<int, 3> nx)
      : xmin_(xmin), xmax_(xmax), xrat_(xrat),
        nx_(nx), symmetry_{nx[0] == 1, nx[1] == 1, nx[2] == 1} {}
  RegionSize(std::array<Real, 3> xmin, std::array<Real, 3> xmax, std::array<Real, 3> xrat,
             std::array<int, 3> nx, std::array<bool, 3> symmetry)
      : xmin_(xmin), xmax_(xmax), xrat_(xrat), nx_(nx), symmetry_(symmetry) {}

  std::array<Real, 3> xmin_, xmax_, xrat_; // xrat is ratio of dxf(i)/dxf(i-1)
  std::array<int, 3> nx_;
  std::array<bool, 3> symmetry_;

  Real &xmin(CoordinateDirection dir) { return xmin_[dir - 1]; }
  const Real &xmin(CoordinateDirection dir) const { return xmin_[dir - 1]; }

  Real &xmax(CoordinateDirection dir) { return xmax_[dir - 1]; }
  const Real &xmax(CoordinateDirection dir) const { return xmax_[dir - 1]; }

  Real &xrat(CoordinateDirection dir) { return xrat_[dir - 1]; }
  const Real &xrat(CoordinateDirection dir) const { return xrat_[dir - 1]; }

  int &nx(CoordinateDirection dir) { return nx_[dir - 1]; }
  const int &nx(CoordinateDirection dir) const { return nx_[dir - 1]; }

  // Returns global coordinate position within a block based on block local
  // coordinate u running from zero to one
  Real LogicalToActualPosition(Real u, CoordinateDirection dir) const {
    return u * (xmax_[dir - 1] - xmin_[dir - 1]) + xmin_[dir - 1];
  }
  // A "symmetry" direction is a a direction that posesses a translational symmetry
  // (or rotational symmetry, etc. for non-cartesian coordinate systems) in the given
  // problem. In practice, this mean that the Parthenon mesh was setup to have only
  // size one in a symmetry direction, the block size is one in those directions,
  // and there are no ghost zones in that direction. Since we support multi-grid
  // mesh hierarchies where blocks are not all the same size above the root grid
  // and can end up having size one even in a non-symmetry direction, we need a different
  // identifier for checking if a direction is a symmetry direction beyond just
  // checking if the size is one in that direction.
  bool &symmetry(CoordinateDirection dir) { return symmetry_[dir - 1]; }
  const bool &symmetry(CoordinateDirection dir) const { return symmetry_[dir - 1]; }
};

//----------------------------------------------------------------------------------------
// enums used everywhere
// (not specifying underlying integral type (C++11) for portability & performance)

// TODO(felker): C++ Core Guidelines Enum.5: Don’t use ALL_CAPS for enumerators
// (avoid clashes with preprocessor macros). Enumerated type definitions in this file and:
// io_wrapper.hpp, bvals.hpp, field_diffusion.hpp,
// tasks.hpp, ???

// identifiers for boundary conditions
enum class BoundaryFlag { block = -1, undef, reflect, outflow, periodic, user };

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

inline BoundaryFace GetInnerBoundaryFace(CoordinateDirection dir) {
  if (dir == X1DIR) {
    return BoundaryFace::inner_x1;
  } else if (dir == X2DIR) {
    return BoundaryFace::inner_x2;
  } else if (dir == X3DIR) {
    return BoundaryFace::inner_x3;
  }
  return BoundaryFace::undef;
}

inline BoundaryFace GetOuterBoundaryFace(CoordinateDirection dir) {
  if (dir == X1DIR) {
    return BoundaryFace::outer_x1;
  } else if (dir == X2DIR) {
    return BoundaryFace::outer_x2;
  } else if (dir == X3DIR) {
    return BoundaryFace::outer_x3;
  }
  return BoundaryFace::undef;
}

//------------------
// strongly typed / scoped enums (C++11):
//------------------
// KGF: Except for the 2x MG* enums, these may be unnessary w/ the new class inheritance
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
