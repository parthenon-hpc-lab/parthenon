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
#ifndef UTILS_ALL_REDUCE_HPP_
#define UTILS_ALL_REDUCE_HPP_

#include <vector>

#include <config.hpp>
#include <parthenon_mpi.hpp>
#include <utils/error_checking.hpp>

namespace parthenon {

enum class Op { MAX, MIN, SUM, PROD, LAND, BAND, LOR, BOR, LXOR, BXOR, MAXLOC, MINLOC };
#ifndef MPI_PARALLEL
#define MPI_
#endif

// Some helper functions
template <typename U>
void *GetPtr(std::vector<U> &v) {
  return v.data();
}
template <typename U>
void *GetPtr(U &v) {
  return &v;
}

template <typename U>
int GetSize(std::vector<U> &v) {
  return v.size();
}
template <typename U>
int GetSize(U &v) {
  return 1;
}

#ifdef MPI_PARALLEL
template <typename U>
MPI_Datatype GetType(std::vector<U> &v) {
  return MPITypeMap<U>::type();
}
template <typename U>
MPI_Datatype GetType(U &v) {
  return MPITypeMap<U>::type();
}
#endif

template <typename T>
struct AllReduce {
  T val;
#ifdef MPI_PARALLEL
  MPI_Request req;
  MPI_Comm comm;
#endif
  bool active = false;
  AllReduce() {
#ifdef MPI_PARALLEL
    MPI_Comm_dup(MPI_COMM_WORLD, &comm);
#endif
  }

  TaskStatus StartReduce(MPI_Op op) {
#ifdef MPI_PARALLEL
    auto type = GetType(val);
    PARTHENON_REQUIRE_THROWS(
        type != MPI_DATATYPE_NULL,
        "Invalid type passed to StartReduce. Add type to parthenon_mpi.hpp");
    MPI_Iallreduce(MPI_IN_PLACE, GetPtr(val), GetSize(val), type, op, comm, &req);
#endif
    active = true;
    return TaskStatus::complete;
  }

  TaskStatus CheckReduce() {
    if (!active) return TaskStatus::complete;
    int check = 1;
#ifdef MPI_PARALLEL
    MPI_Test(&req, &check, MPI_STATUS_IGNORE);
#endif
    if (check) {
      active = false;
      return TaskStatus::complete;
    }
    return TaskStatus::incomplete;
  }
};

#undef MPI_

} // namespace parthenon

#endif // UTILS_ALL_REDUCE_HPP_
