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

#ifndef UTILS_MPI_TYPES_HPP_
#define UTILS_MPI_TYPES_HPP_

#include "basic_types.hpp"
#include <parthenon_mpi.hpp>
#include <utils/error_checking.hpp>

#ifdef MPI_PARALLEL
namespace parthenon {

template <typename T>
struct MPITypeMap {
  static MPI_Datatype type() {
    PARTHENON_THROW("Type not available in MPITypeMap.");
    return MPI_DATATYPE_NULL;
  }
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

namespace parthenon {

#ifdef MPI_PARALLEL
using mpi_request_t = MPI_Request;
using mpi_comm_t = MPI_Comm;
#else
using mpi_request_t = int;
using mpi_comm_t = int;
#endif

} // namespace parthenon 

#endif // UTILS_MPI_TYPES_HPP_
