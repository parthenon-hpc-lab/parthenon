//========================================================================================
// (C) (or copyright) 2021. Triad National Security, LLC. All rights reserved.
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

#ifndef PARTHENON_MPI_HPP_
#define PARTHENON_MPI_HPP_

//! \file parthenon_mpi.hpp
//  \brief Helper file to include MPI if it's enabled and otherwise not include it. One
//         issue was that some header files attempted to include MPI by checking #ifdef
//         MPI_PARALLEL, but they didn't include config.hpp, which defined MPI_PARALLEL

#include "basic_types.hpp"
#include "config.hpp"

#ifdef MPI_PARALLEL
#include <mpi.h>

namespace parthenon {

template <typename T>
struct MPITypeMap {
  static MPI_Datatype type() { return MPI_DATATYPE_NULL; }
};

template <>
inline MPI_Datatype MPITypeMap<Real>::type() {
  return MPI_PARTHENON_REAL;
}

template <>
inline MPI_Datatype MPITypeMap<int>::type() {
  return MPI_INT;
}

template <>
inline MPI_Datatype MPITypeMap<bool>::type() {
  return MPI_CXX_BOOL;
}

} // namespace parthenon

#endif

#endif // PARTHENON_MPI_HPP_
