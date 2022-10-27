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

#include "config.hpp"

namespace parthenon {

// primitive type alias that allows code to run with either floats or doubles
#if SINGLE_PRECISION_ENABLED
using Real = float;
using Real64 = double;
#ifdef MPI_PARALLEL
#define MPI_PARTHENON_REAL MPI_FLOAT
#define MPI_PARTHENON_REAL64 MPI_DOUBLE
#endif
#else
using Real = double;
using Real64 = double;
#ifdef MPI_PARALLEL
#define MPI_PARTHENON_REAL MPI_DOUBLE
#define MPI_PARTHENON_REAL64 MPI_DOUBLE
#endif
#endif

enum class TaskStatus { fail, complete, incomplete, iterate, skip };
enum class AmrTag : int { derefine = -1, same = 0, refine = 1 };
enum class RefinementOp_t { Prolongation, Restriction, None };

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
