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
#ifndef BASIC_TYPES_HPP_
#define BASIC_TYPES_HPP_

#include <limits>

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

enum class TaskStatus { fail, complete, incomplete };
enum class AmrTag : int { derefine = -1, same = 0, refine = 1 };

struct SimTime {
  SimTime() = default;
  SimTime(const Real tstart, const Real tstop, const int nmax, const int ncurr,
          const int nout)
      : start_time(tstart), time(tstart), tlim(tstop),
        dt(std::numeric_limits<Real>::max()), nlim(nmax), ncycle(ncurr),
        ncycle_out(nout) {}
  // beginning time, current time, maximum time, time step
  Real start_time, time, tlim, dt;
  // current cycle number, maximum number of cycles, cycles between diagnostic output
  int ncycle, nlim, ncycle_out;

  bool KeepGoing() { return ((time < tlim) && (nlim < 0 || ncycle < nlim)); }
};

} // namespace parthenon

#endif // BASIC_TYPES_HPP_
