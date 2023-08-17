//========================================================================================
// (C) (or copyright) 2021-2023. Triad National Security, LLC. All rights reserved.
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

// needed for arrays dimensioned over grid directions
// X0DIR time-like direction
// X1DIR x, r, etc...
// X2DIR y, theta, etc...
// X3DIR z, phi, etc...
enum CoordinateDirection { NODIR = -1, X0DIR = 0, X1DIR = 1, X2DIR = 2, X3DIR = 3 };
enum class BlockLocation { Left = 0, Center = 1, Right = 2 };

enum class TaskStatus { fail, complete, incomplete, iterate, skip };
enum class AmrTag : int { derefine = -1, same = 0, refine = 1 };
enum class RefinementOp_t { Prolongation, Restriction, None };

// JMM: Not clear this is the best place for this but it minimizes
// circular dependency nonsense.
constexpr int NUM_BNDRY_TYPES = 9;
enum class BoundaryType : int {
  local,
  nonlocal,
  any,
  flxcor_send,
  flxcor_recv,
  gmg_restrict_send,
  gmg_restrict_recv,
  gmg_prolongate_send,
  gmg_prolongate_recv
};

constexpr bool IsSender(BoundaryType btype) {
  if (btype == BoundaryType::flxcor_recv) return false;
  if (btype == BoundaryType::gmg_restrict_recv) return false;
  if (btype == BoundaryType::gmg_prolongate_recv) return false;
  return true;
}

constexpr bool IsReceiver(BoundaryType btype) {
  if (btype == BoundaryType::flxcor_send) return false;
  if (btype == BoundaryType::gmg_restrict_send) return false;
  if (btype == BoundaryType::gmg_prolongate_send) return false;
  return true;
}

// Enumeration for accessing a field on different locations of the grid:
// CC = cell center of (i, j, k)
// F1 = x-normal face at (i - 1/2, j, k)
// F2 = y-normal face at (i, j - 1/2, k)
// F3 = z-normal face at (i, j, k - 1/2)
// E1 = x-aligned edge at (i, j - 1/2, k - 1/2)
// E2 = y-aligned edge at (i - 1/2, j, k - 1/2)
// E3 = z-aligned edge at (i - 1/2, j - 1/2, k)
// NN = node at (i - 1/2, j - 1/2, k - 1/2)
//
// Some select topological elements around cell (i,j,k) with o corresponding
// to faces, x corresponding to edges, and + corresponding to nodes (the indices
// denote the array index of each element):
// clang-format off
//
//                      E1(i,j+1,k+1)
//              NN_+---------x---------+_NN(i+1,j+1,k+1)
//     (i,j+1,k+1)/|  F3(i,j,k+1)     /|
//               / |      |          / |
//           E2_x  |      o         x__|_E2(i+1,j,k+1)
//    (i,j,k+1)/   x         o     /   x___E3(i+1,j+1,k)
//            /    |         |___ /____|_F2(i,j+1,k)
//        NN_+---------x---------+_____|_NN(i+1,j,k+1)
// (i,j,k+1) |  o  |  E1         |  o__|____F1(i+1,j,k)
//        F1_|__|  +-(i,j,k+1)---|-----+______NN(i+1,j+1,k)
//   (i,j,k) |    /     F3(i,j,k)|    /
//        E3_x   /     o  |      x___/___E3(i+1,j,k)
//   (i,j,k) |  x      |  o      |  x______E2(i+1,j,k)
//        E2_|_/|    F2(i,j,k)   | /
//   (i,j,k) |/                  |/
//           +---------x---------+
//           NN        E1        NN
//           (i,j,k)   (i,j,k)   (i+1,j,k)
//
// clang-format on
// The values of the enumeration are chosen so we can do te % 3 to get
// the correct index for each type of element in Variable::data
enum class TopologicalElement : std::size_t {
  CC = 0,
  F1 = 3,
  F2 = 4,
  F3 = 5,
  E1 = 6,
  E2 = 7,
  E3 = 8,
  NN = 9
};
enum class TopologicalType { Cell, Face, Edge, Node };

KOKKOS_FORCEINLINE_FUNCTION
TopologicalType GetTopologicalType(TopologicalElement el) {
  using TE = TopologicalElement;
  using TT = TopologicalType;
  if (el == TE::CC) {
    return TT::Cell;
  } else if (el == TE::NN) {
    return TT::Node;
  } else if (el == TE::F1 || el == TE::F2 || el == TE::F3) {
    return TT::Face;
  } else {
    return TT::Edge;
  }
}

inline std::vector<TopologicalElement> GetTopologicalElements(TopologicalType tt) { 
  using TE = TopologicalElement;
  using TT = TopologicalType;
   if (tt == TT::Node) return {TE::NN};
   if (tt == TT::Edge) return {TE::E1, TE::E2, TE::E3};
   if (tt == TT::Face) return {TE::F1, TE::F2, TE::F3};
   return {TE::CC};
}
using TE = TopologicalElement;
// Returns one if the I coordinate of el is offset from the zone center coordinates,
// and zero otherwise
KOKKOS_INLINE_FUNCTION int TopologicalOffsetI(TE el) noexcept {
  return (el == TE::F1 || el == TE::E2 || el == TE::E3 || el == TE::NN);
}
KOKKOS_INLINE_FUNCTION int TopologicalOffsetJ(TE el) noexcept {
  return (el == TE::F2 || el == TE::E3 || el == TE::E1 || el == TE::NN);
}
KOKKOS_INLINE_FUNCTION int TopologicalOffsetK(TE el) noexcept {
  return (el == TE::F3 || el == TE::E2 || el == TE::E1 || el == TE::NN);
}

// Returns wether or not topological element containee is a boundary of
// topological element container
inline constexpr bool IsSubmanifold(TopologicalElement containee,
                                    TopologicalElement container) {
  if (container == TE::CC) {
    return containee != TE::CC;
  } else if (container == TE::F1) {
    return containee == TE::E2 || containee == TE::E3 || containee == TE::NN;
  } else if (container == TE::F2) {
    return containee == TE::E3 || containee == TE::E1 || containee == TE::NN;
  } else if (container == TE::F3) {
    return containee == TE::E1 || containee == TE::E2 || containee == TE::NN;
  } else if (container == TE::E3) {
    return containee == TE::NN;
  } else if (container == TE::E2) {
    return containee == TE::NN;
  } else if (container == TE::E1) {
    return containee == TE::NN;
  } else if (container == TE::NN) {
    return false;
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
