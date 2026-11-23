//========================================================================================
// (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
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

// This file was made in part with generative AI.

// Optional background thread that periodically calls MPI_Iprobe so that the MPI library
// can progress outstanding requests (e.g. rendezvous sends) while the main thread is busy
// outside of MPI. MPI is initialized before the input file is read, so this is controlled
// by environment variables:
//
//   PARTHENON_MPI_PROGRESS_US   unset: no progress thread (MPI_Init as usual)
//                               N > 0: call MPI_Iprobe, then sleep N microseconds
//                               0:     call MPI_Iprobe, then yield (busy loop)
//   PARTHENON_MPI_PROGRESS_CPU  auto (default): pin the thread to the highest CPU in the
//                               process affinity mask and remove that CPU from the main
//                               thread's mask (only if the mask has more than one CPU)
//                               none: don't pin
//                               N: pin the thread to CPU N
//
// Requesting the thread requires MPI_THREAD_MULTIPLE. If MPI does not provide it, a
// warning is printed and no thread is started.

#ifndef UTILS_MPI_PROGRESS_HPP_
#define UTILS_MPI_PROGRESS_HPP_

namespace parthenon {
namespace mpi_progress {

// Initialize MPI (if it is not initialized already), requesting MPI_THREAD_MULTIPLE if
// the progress thread is requested. Returns the MPI error code.
int InitMPI(int *argc, char ***argv);

// Start the progress thread if it was requested and MPI supports it. Collective.
void Start();

// Stop the progress thread (if running) and print its statistics. Collective if the
// thread was started.
void Stop();

bool Running();

} // namespace mpi_progress
} // namespace parthenon

#endif // UTILS_MPI_PROGRESS_HPP_
