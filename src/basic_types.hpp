//========================================================================================
// (C) (or copyright) 2021-2022. Triad National Security, LLC. All rights reserved.
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
#ifndef BASIC_TYPES_HPP_
#define BASIC_TYPES_HPP_

#include <limits>
#include <string>
#include <unordered_map>

#include <Kokkos_Core.hpp>

#include "config.hpp"

namespace parthenon {

// primitive type alias that allows code to run with either floats or doubles
#if SINGLE_PRECISION_ENABLED
using Real = float;
#ifdef MPI_PARALLEL
#define MPI_PARTHENON_REAL MPI_FLOAT
#endif
#else
using Real = double;
#ifdef MPI_PARALLEL
#define MPI_PARTHENON_REAL MPI_DOUBLE
#endif
#endif

enum class TaskStatus { fail, complete, incomplete, iterate, skip };
enum class AmrTag : int { derefine = -1, same = 0, refine = 1 };
enum class RefinementOp_t { Prolongation, Restriction, None };

// JMM: Not clear this is the best place for this but it minimizes
// circular dependency nonsense.
constexpr int NUM_BNDRY_TYPES = 5;
enum class BoundaryType : int { local, nonlocal, any, flxcor_send, flxcor_recv };

// Enumeration for accessing a field on different locations of the grid:
// C = cell center of (i, j, k)
// FX = x-face at (i - 1/2, j, k)
// FY = y-face at (i, j - 1/2, k)
// FZ = z-face at (i, j, k - 1/2)
// EXY = edge at (i - 1/2, j - 1/2, k)
// EXZ = edge at (i - 1/2, j, k - 1/2)
// EXY = edge at (i, j - 1/2, k - 1/2)
// NXYZ = node at (i - 1/2, j - 1/2, k - 1/2)
//
// The values of the enumeration are chosen so we can do te % 3 to get
// the correct index for each type of element in Variable::data
enum class TopologicalElement : std::size_t {
  C = 0,
  FX = 3,
  FY = 4,
  FZ = 5,
  EYZ = 6,
  EXZ = 7,
  EXY = 8,
  NXYZ = 9
};
enum class TopologicalType { Cell, Face, Edge, Node };

KOKKOS_FORCEINLINE_FUNCTION
TopologicalType GetTopologicalType(TopologicalElement el) {
  using te = TopologicalElement;
  using tt = TopologicalType;
  if (el == te::C) {
    return tt::Cell;
  } else if (el == te::NXYZ) {
    return tt::Node;
  } else if (el == te::FX || el == te::FY || el == te::FZ) {
    return tt::Face;
  } else {
    return tt::Edge;
  }
}

using TE = TopologicalElement;
KOKKOS_INLINE_FUNCTION int TopologicalOffsetI(TE el) noexcept {
  return (el == TE::FX || el == TE::EXY || el == TE::EYZ || el == TE::NXYZ);
}
KOKKOS_INLINE_FUNCTION int TopologicalOffsetJ(TE el) noexcept {
  return (el == TE::FY || el == TE::EXY || el == TE::EYZ || el == TE::NXYZ);
}
KOKKOS_INLINE_FUNCTION int TopologicalOffsetK(TE el) noexcept {
  return (el == TE::FZ || el == TE::EXZ || el == TE::EYZ || el == TE::NXYZ);
}

inline constexpr bool IsSubmanifold(TopologicalElement container,
                                    TopologicalElement containee) {
  if (container == TE::C) {
    return true;
  } else if (container == TE::FX) {
    return containee == TE::FX || containee == TE::EXY || containee == TE::EXZ ||
           containee == TE::NXYZ;
  } else if (container == TE::FY) {
    return containee == TE::FY || containee == TE::EXY || containee == TE::EYZ ||
           containee == TE::NXYZ;
  } else if (container == TE::FZ) {
    return containee == TE::FZ || containee == TE::EXZ || containee == TE::EYZ ||
           containee == TE::NXYZ;
  } else if (container == TE::EXY) {
    return containee == TE::EXY || containee == TE::NXYZ;
  } else if (container == TE::EXZ) {
    return containee == TE::EXZ || containee == TE::NXYZ;
  } else if (container == TE::EYZ) {
    return containee == TE::EYZ || containee == TE::NXYZ;
  } else if (container == TE::NXYZ) {
    return containee == TE::NXYZ;
  } else {
    return false;
  }
}
struct SimTime {
  SimTime() = default;
  SimTime(const Real tstart, const Real tstop, const int nmax, const int ncurr,
          const int nout, const int nout_mesh,
          const Real dt_in = std::numeric_limits<Real>::max())
      : start_time(tstart), time(tstart), tlim(tstop), dt(dt_in), nlim(nmax),
        ncycle(ncurr), ncycle_out(nout), ncycle_out_mesh(nout_mesh) {}
  // beginning time, current time, maximum time, time step
  Real start_time, time, tlim, dt;
  // current cycle number, maximum number of cycles, cycles between diagnostic output
  int ncycle, nlim, ncycle_out, ncycle_out_mesh;

  bool KeepGoing() { return ((time < tlim) && (nlim < 0 || ncycle < nlim)); }
};

template <typename T>
using Dictionary = std::unordered_map<std::string, T>;

} // namespace parthenon

#endif // BASIC_TYPES_HPP_
