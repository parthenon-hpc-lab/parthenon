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
#ifndef UTILS_REDUCTIONS_HPP_
#define UTILS_REDUCTIONS_HPP_

#include <vector>

#include <config.hpp>
#include <globals.hpp>
#include <parthenon_mpi.hpp>
#include <utils/error_checking.hpp>
#include <utils/mpi_types.hpp>

namespace parthenon {

#ifndef MPI_PARALLEL
enum MPI_Op {
  MPI_MAX,
  MPI_MIN,
  MPI_SUM,
  MPI_PROD,
  MPI_LAND,
  MPI_BAND,
  MPI_LOR,
  MPI_BOR,
  MPI_LXOR,
  MPI_BXOR,
  MPI_MAXLOC,
  MPI_MINLOC
};
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
struct ReductionBase {
  T val;
#ifdef MPI_PARALLEL
  MPI_Request req;
  MPI_Comm comm;
#endif
  bool active = false;
  ReductionBase() {
#ifdef MPI_PARALLEL
    MPI_Comm_dup(MPI_COMM_WORLD, &comm);
#endif
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

template <typename T>
struct AllReduce : public ReductionBase<T> {
  TaskStatus StartReduce(MPI_Op op) {
    if (this->active) return TaskStatus::complete;
#ifdef MPI_PARALLEL
    MPI_Iallreduce(MPI_IN_PLACE, GetPtr(this->val), GetSize(this->val),
                   GetType(this->val), op, this->comm, &(this->req));
#endif
    this->active = true;
    return TaskStatus::complete;
  }
};

template <typename T>
struct Reduce : public ReductionBase<T> {
  TaskStatus StartReduce(const int n, MPI_Op op) {
    if (this->active) return TaskStatus::complete;
#ifdef MPI_PARALLEL
    if (Globals::my_rank == n) {
      MPI_Ireduce(MPI_IN_PLACE, GetPtr(this->val), GetSize(this->val), GetType(this->val),
                  op, n, this->comm, &(this->req));

    } else {
      MPI_Ireduce(GetPtr(this->val), nullptr, GetSize(this->val), GetType(this->val), op,
                  n, this->comm, &(this->req));
    }
#endif
    this->active = true;
    return TaskStatus::complete;
  }
};

} // namespace parthenon

#endif // UTILS_REDUCTIONS_HPP_
