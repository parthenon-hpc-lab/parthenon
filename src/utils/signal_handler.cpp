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
//! \file signal_handler.cpp
//  \brief contains functions that implement a simple SignalHandler
//  These functions are based on TAG's signal handler written for Athena 8/19/2004

#include <sys/stat.h>
#include <unistd.h> // alarm() Unix OS utility; not in C standard --> no <cunistd>

// first 2x macros and signal() are the only ISO C features; rest are POSIX C extensions
#include <csignal>
#include <iostream>

#include "parthenon_mpi.hpp"

#include "globals.hpp"
#include "utils.hpp"

namespace parthenon {
namespace SignalHandler {

//----------------------------------------------------------------------------------------
//! \fn void SignalHandlerInit()
//  \brief install handlers for selected signals

void SignalHandlerInit() {
  // +1 for "output_now" trigger
  for (int n = 0; n < nsignal + 1; n++) {
    signalflag[n] = 0;
  }
  // C++11 standard guarantees that <csignal> places C-standard signal.h contents in std::
  // namespace. POSIX C extensions are likely only placed in global namespace (not std::)
  std::signal(SIGTERM, SetSignalFlag);
  std::signal(SIGINT, SetSignalFlag);
  std::signal(SIGALRM, SetSignalFlag);

  // populate set of signals to block while the handler is running; prevent premption
  sigemptyset(&mask);
  sigaddset(&mask, SIGTERM);
  sigaddset(&mask, SIGINT);
  sigaddset(&mask, SIGALRM);
}

//----------------------------------------------------------------------------------------
//! \fn int CheckSignalFlags()
//  \brief Synchronize and check signal flags and return true if any of them is caught

OutputSignal CheckSignalFlags() {
  if (Globals::my_rank == 0) {
    // TODO(the person bumping std to C++17): use std::filesystem::exists
    struct stat buffer;
    // if file "output_now" exists
    if (stat("output_now", &buffer) == 0) {
      signalflag[nsignal] = 1;
    }
  }
  // Currently, only checking for nonzero return code at the end of each timestep in
  // main.cpp; i.e. if an issue prevents a process from reaching the end of a cycle, the
  // signals will never be handled by that process / the solver may hang
  sigprocmask(SIG_BLOCK, &mask, nullptr);
#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(MPI_Allreduce(
      MPI_IN_PLACE, const_cast<void *>(reinterpret_cast<volatile void *>(signalflag)),
      nsignal + 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD));
#endif
  sigprocmask(SIG_UNBLOCK, &mask, nullptr);
  for (int n = 0; n < nsignal; n++) {
    if (signalflag[n] != 0) {
      return OutputSignal::final;
    }
  }
  if (signalflag[nsignal] != 0) {
    // reset signalflag and cleanup trigger file
    signalflag[nsignal] = 0;
    if (Globals::my_rank == 0) {
      // Cleanup trigger file.
      // Fail hard in case there's an issue as otherwise this could lead to
      // excessive file system load by writing dumps triggered every single cycle.
      PARTHENON_REQUIRE_THROWS(
          remove("output_now") == 0,
          "Could not remove 'output_now' file that triggered output.");
    }
    return OutputSignal::now;
  }
  return OutputSignal::none;
}

//----------------------------------------------------------------------------------------
//! \fn int GetSignalFlag(int s)
//  \brief Gets a signal flag assuming the signalflag array is already synchronized.
//         Returns -1 if the specified signal is not handled.

int GetSignalFlag(int s) {
  int ret = -1;
  switch (s) {
  case SIGTERM:
    ret = signalflag[ITERM];
    break;
  case SIGINT:
    ret = signalflag[IINT];
    break;
  case SIGALRM:
    ret = signalflag[IALRM];
    break;
  default:
    // nothing
    break;
  }
  return ret;
}

//----------------------------------------------------------------------------------------
//! \fn void SetSignalFlag(int s)
//  \brief Sets signal flags and reinstalls the signal handler function.

void SetSignalFlag(int s) {
  // Signal handler functions must have C linkage; C++ linkage is implemantation-defined
  switch (s) {
  case SIGTERM:
    signalflag[ITERM] = 1;
    signal(s, SetSignalFlag);
    break;
  case SIGINT:
    signalflag[IINT] = 1;
    signal(s, SetSignalFlag);
    break;
  case SIGALRM:
    signalflag[IALRM] = 1;
    signal(s, SetSignalFlag);
    break;
  default:
    // nothing
    break;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void SetWallTimeAlarm(int t)
//  \brief Set the wall time limit alarm

void SetWallTimeAlarm(int t) {
  alarm(t);
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void CancelWallTimeAlarm()
//  \brief Cancel the wall time limit alarm

void CancelWallTimeAlarm() {
  alarm(0);
  return;
}

void Report() {
  if (SignalHandler::GetSignalFlag(SIGTERM) != 0) {
    std::cout << std::endl << "Terminating on Terminate signal" << std::endl;
  } else if (SignalHandler::GetSignalFlag(SIGINT) != 0) {
    std::cout << std::endl << "Terminating on Interrupt signal" << std::endl;
  } else if (SignalHandler::GetSignalFlag(SIGALRM) != 0) {
    std::cout << std::endl << "Terminating on wall-time limit" << std::endl;
  }
}

} // namespace SignalHandler
} // namespace parthenon
