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
#ifndef ATHENA_HPP_
#define ATHENA_HPP_
//! \file athena.hpp
//  \brief contains Athena++ general purpose types, structures, enums, etc.

// C headers

// C++ headers
#include <cmath>
#include <cstdint>  // std::int64_t

// Athena++ headers
#include "athena_arrays.hpp"
#include <defs.hpp>

namespace parthenon {
// primitive type alias that allows code to run with either floats or doubles
#if SINGLE_PRECISION_ENABLED
using Real = float;
#ifdef MPI_PARALLEL
#define MPI_ATHENA_REAL MPI_FLOAT
#endif
#else
using Real = double;
#ifdef MPI_PARALLEL
#define MPI_ATHENA_REAL MPI_DOUBLE
#endif
#endif

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

// forward declarations needed for function pointer type aliases
class MeshBlock;
class Coordinates;
class ParameterInput;
class FieldDiffusion;

//--------------------------------------------------------------------------------------
//! \struct LogicalLocation
//  \brief stores logical location and level of MeshBlock

struct LogicalLocation { // aggregate and POD type
  // These values can exceed the range of std::int32_t even if the root grid has only a
  // single MeshBlock if >30 levels of AMR are used, since the corresponding max index =
  // 1*2^31 > INT_MAX = 2^31 -1 for most 32-bit signed integer type impelementations
  std::int64_t lx1, lx2, lx3;
  int level;

  // operators useful for sorting
  bool operator==(LogicalLocation &ll) {
    return ((ll.level == level) && (ll.lx1 == lx1) && (ll.lx2 == lx2) && (ll.lx3 == lx3));
  }
  static bool Lesser(const LogicalLocation &left, const LogicalLocation &right) {
    return left.level < right.level;
  }
  static bool Greater(const LogicalLocation & left, const LogicalLocation &right) {
    return left.level > right.level;
  }
};

//----------------------------------------------------------------------------------------
//! \struct RegionSize
//  \brief physical size and number of cells in a Mesh or a MeshBlock

struct RegionSize {  // aggregate and POD type; do NOT reorder member declarations:
  Real x1min, x2min, x3min;
  Real x1max, x2max, x3max;
  Real x1rat, x2rat, x3rat; // ratio of dxf(i)/dxf(i-1)
  // the size of the root grid or a MeshBlock should not exceed std::int32_t limits
  int nx1, nx2, nx3;        // number of active cells (not including ghost zones)
};

//---------------------------------------------------------------------------------------
//! \struct FaceField
//  \brief container for face-centered fields

struct FaceField {
  ParArrayND<Real> x1f, x2f, x3f;
  FaceField() = default;
  FaceField(int ncells3, int ncells2, int ncells1)
    : x1f("x1f",ncells3, ncells2, ncells1+1), x2f("x2f",ncells3, ncells2+1, ncells1),
      x3f("x3f",ncells3+1, ncells2, ncells1) {}
  FaceField(int ncells6, int ncells5, int ncells4, int ncells3, int ncells2, int ncells1)
    : x1f("x1f",ncells6, ncells5, ncells4, ncells3, ncells2, ncells1+1)
    , x2f("x2f",ncells6, ncells5, ncells4, ncells3, ncells2+1, ncells1)
    , x3f("x3f",ncells6, ncells5, ncells4, ncells3+1, ncells2, ncells1)
  {}
};

//----------------------------------------------------------------------------------------
//! \struct EdgeField
//  \brief container for edge-centered fields

struct EdgeField {
  ParArrayND<Real> x1e, x2e, x3e;
  EdgeField() = default;
  EdgeField(int ncells3, int ncells2, int ncells1)
    : x1e("x1e",ncells3+1, ncells2+1, ncells1), x2e("x2e",ncells3+1, ncells2, ncells1+1),
      x3e("x3e",ncells3, ncells2+1, ncells1+1) {}
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
enum CoordinateDirection {X1DIR=0, X2DIR=1, X3DIR=2};

//------------------
// strongly typed / scoped enums (C++11):
//------------------
// KGF: Except for the 2x MG* enums, these may be unnessary w/ the new class inheritance
// Now, only passed to BoundaryVariable::InitBoundaryData(); could replace w/ bool switch
enum class BoundaryQuantity {cc, fc, cc_flcor, fc_flcor};
enum class BoundaryCommSubset {mesh_init, gr_amr, all};
// TODO(felker): consider generalizing/renaming to QuantityFormulation
enum class UserHistoryOperation {sum, max, min};

//----------------------------------------------------------------------------------------
// function pointer prototypes for user-defined modules set at runtime

using BValFunc = void (*)(
    MeshBlock *pmb, Coordinates *pco, ParArrayND<Real> &prim, FaceField &b,
    Real time, Real dt,
    int is, int ie, int js, int je, int ks, int ke, int ngh);
using AMRFlagFunc = int (*)(MeshBlock *pmb);
using MeshGenFunc = Real (*)(Real x, RegionSize rs);
using SrcTermFunc = void (*)(
    MeshBlock *pmb, const Real time, const Real dt,
    const ParArrayND<Real> &prim, const ParArrayND<Real> &bcc, ParArrayND<Real> &cons);
using TimeStepFunc = Real (*)(MeshBlock *pmb);
using HistoryOutputFunc = Real (*)(MeshBlock *pmb, int iout);
using MetricFunc = void (*)(
    Real x1, Real x2, Real x3, ParameterInput *pin,
    ParArrayND<Real> &g, ParArrayND<Real> &g_inv,
    ParArrayND<Real> &dg_dx1, ParArrayND<Real> &dg_dx2, ParArrayND<Real> &dg_dx3);
using MGBoundaryFunc = void (*)(
    ParArrayND<Real> &dst,Real time, int nvar,
    int is, int ie, int js, int je, int ks, int ke, int ngh,
    Real x0, Real y0, Real z0, Real dx, Real dy, Real dz);
using FieldDiffusionCoeffFunc = void (*)(
    FieldDiffusion *pfdif, MeshBlock *pmb,
    const ParArrayND<Real> &w,
    const ParArrayND<Real> &bmag,
    int is, int ie, int js, int je, int ks, int ke);

} // namespace parthenon

#endif // ATHENA_HPP_
