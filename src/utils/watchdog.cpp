//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2025-2026 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2025-2026. Triad National Security, LLC. All rights reserved.
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
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file watchdog.cpp
//  \brief WatchDog implementation ported from the Einstein Toolkit and AthenaK

#include <pthread.h>
#include <unistd.h>

#include <cstdio>
#include <ctime>

#include "globals.hpp"
#include "utils.hpp"

namespace parthenon {
namespace WatchDog {
static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
static time_t timestamp;
static struct {
  int timeout_sec;
  int mpi_rank;
} param;

static void *patrol(void *args) {
  time_t time_old, time_new, ltime;
  char tstamp[128];

  pthread_mutex_lock(&mutex);
  time_old = timestamp;
  pthread_mutex_unlock(&mutex);

  if (0 == param.mpi_rank) {
    ltime = time(NULL);
    asctime_r(localtime(&ltime), &tstamp[0]); // NOLINT
    tstamp[24] = '\0';
    fprintf(stderr, "[WATCHDOG (%s)] Starting.\n", tstamp);
    fflush(stderr);
  }

  while (true) {
    unsigned int left = param.timeout_sec;
    while (left > 0) {
      left = sleep(left);
    }

    pthread_mutex_lock(&mutex);
    time_new = timestamp;
    pthread_mutex_unlock(&mutex);

    ltime = time(NULL);
    asctime_r(localtime(&ltime), &tstamp[0]); // NOLINT
    tstamp[24] = '\0';
    if (time_new == time_old) {
      fprintf(stderr, "[WATCHDOG (%s)] Rank %d is not progressing.\n", tstamp,
              param.mpi_rank);
      fprintf(stderr, "[WATCHDOG (%s)] Terminating...\n", tstamp);
      fflush(stderr);
      abort();
    } else {
      if (0 == param.mpi_rank) {
        fprintf(stderr, "[WATCHDOG (%s)] Everything is fine.\n", tstamp);
        fflush(stderr);
      }
      time_old = time_new;
    }
  }
}

void WatchDog(int timeout) {
  pthread_mutex_lock(&mutex);
  timestamp = time(NULL);
  pthread_mutex_unlock(&mutex);

  static bool first_time = true;
  if (first_time) {
    param.timeout_sec = timeout;
    param.mpi_rank = Globals::my_rank;
    pthread_t dog; /* not used beyond passed to phread_ceate */
    int ierr = pthread_create(&dog, NULL, patrol, NULL);
    if (ierr) {
      printf("#### FATAL ERROR in %s at line %d\n", __FILE__, __LINE__);
      puts("Could not start WatchDog thread");
      std::exit(EXIT_FAILURE);
    }
    first_time = false;
  }
}
} // namespace WatchDog
} // namespace parthenon
