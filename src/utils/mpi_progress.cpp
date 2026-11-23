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

#include "utils/mpi_progress.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

#include "globals.hpp"
#include "parthenon_mpi.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {
namespace mpi_progress {

namespace {
struct State {
  bool requested{false};
  int interval_us{0};
  std::string cpu_setting{"auto"};
  int pinned_cpu{-1};
  bool removed_from_main{false};

  std::thread thread;
  std::atomic<bool> running{false};
  std::atomic<bool> stop{false};
  long long iterations{0}; // written by the progress thread, read after join
  double t_start{0.0}, t_stop{0.0};
#ifdef MPI_PARALLEL
  MPI_Comm comm{MPI_COMM_NULL};
#endif
};

State &S() {
  static State s;
  return s;
}

double Now() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

void ReadEnvironment() {
  auto &s = S();
  const char *us = std::getenv("PARTHENON_MPI_PROGRESS_US");
  if (us != nullptr && std::string(us) != "") {
    s.requested = true;
    s.interval_us = std::atoi(us);
    if (s.interval_us < 0) s.interval_us = 0;
  }
  const char *cpu = std::getenv("PARTHENON_MPI_PROGRESS_CPU");
  if (cpu != nullptr && std::string(cpu) != "") s.cpu_setting = cpu;
}

// Decide which CPU to pin the progress thread to, and remove it from the main thread's
// affinity mask when it was chosen automatically. Must be called on the main thread.
void ChooseCPU() {
  auto &s = S();
#ifdef __linux__
  if (s.cpu_setting == "none") return;
  if (s.cpu_setting != "auto") {
    s.pinned_cpu = std::atoi(s.cpu_setting.c_str());
    return;
  }
  cpu_set_t mask;
  CPU_ZERO(&mask);
  if (sched_getaffinity(0, sizeof(mask), &mask) != 0) return;
  if (CPU_COUNT(&mask) < 2) return;
  for (int c = CPU_SETSIZE - 1; c >= 0; --c) {
    if (CPU_ISSET(c, &mask)) {
      s.pinned_cpu = c;
      break;
    }
  }
  CPU_CLR(s.pinned_cpu, &mask);
  s.removed_from_main = (sched_setaffinity(0, sizeof(mask), &mask) == 0);
#endif
}

void Loop() {
#ifdef MPI_PARALLEL
  auto &s = S();
#ifdef __linux__
  if (s.pinned_cpu >= 0) {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(s.pinned_cpu, &mask);
    pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask);
  }
#endif
  const auto interval = std::chrono::microseconds(s.interval_us);
  long long n = 0;
  while (!s.stop.load(std::memory_order_relaxed)) {
    int flag;
    MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, s.comm, &flag, MPI_STATUS_IGNORE);
    ++n;
    if (s.interval_us > 0) {
      std::this_thread::sleep_for(interval);
    } else {
      std::this_thread::yield();
    }
  }
  s.iterations = n;
#endif
}
} // namespace

int InitMPI(int *argc, char ***argv) {
#ifdef MPI_PARALLEL
  ReadEnvironment();
  int initialized;
  PARTHENON_MPI_CHECK(MPI_Initialized(&initialized));
  if (initialized) return MPI_SUCCESS;
  if (!S().requested) return MPI_Init(argc, argv);
  int provided;
  return MPI_Init_thread(argc, argv, MPI_THREAD_MULTIPLE, &provided);
#else
  return 0;
#endif
}

void Start() {
#ifdef MPI_PARALLEL
  auto &s = S();
  if (s.running) return;
  // MPI may have been initialized by the application, in which case InitMPI did not run
  ReadEnvironment();
  if (!s.requested) return;
  int provided;
  PARTHENON_MPI_CHECK(MPI_Query_thread(&provided));
  if (provided < MPI_THREAD_MULTIPLE) {
    if (Globals::my_rank == 0)
      std::printf("### WARNING: PARTHENON_MPI_PROGRESS_US is set but MPI does not provide "
                  "MPI_THREAD_MULTIPLE (provided = %d), not starting the MPI progress "
                  "thread\n",
                  provided);
    s.requested = false;
    return;
  }
  PARTHENON_MPI_CHECK(MPI_Comm_dup(MPI_COMM_WORLD, &s.comm));
  ChooseCPU();
  s.stop = false;
  s.t_start = Now();
  s.thread = std::thread(Loop);
  s.running = true;

  // Report the pinning of every rank on rank 0
  int info[3] = {s.pinned_cpu, s.removed_from_main ? 1 : 0, 0};
#ifdef __linux__
  info[2] = sched_getcpu();
#endif
  std::vector<int> all(Globals::my_rank == 0 ? 3 * Globals::nranks : 0);
  PARTHENON_MPI_CHECK(
      MPI_Gather(info, 3, MPI_INT, all.data(), 3, MPI_INT, 0, MPI_COMM_WORLD));
  if (Globals::my_rank == 0) {
    std::printf("MPI progress thread started: interval %d us (%s), CPU setting '%s'\n",
                s.interval_us, s.interval_us > 0 ? "sleep" : "yield",
                s.cpu_setting.c_str());
    for (int r = 0; r < Globals::nranks; ++r)
      std::printf("  rank %d: progress thread CPU %d%s, main thread on CPU %d\n", r,
                  all[3 * r], all[3 * r + 1] ? " (removed from main thread mask)" : "",
                  all[3 * r + 2]);
    std::fflush(stdout);
  }
#endif
}

void Stop() {
#ifdef MPI_PARALLEL
  auto &s = S();
  if (!s.running) return;
  s.stop = true;
  s.thread.join();
  s.t_stop = Now();
  s.running = false;
  const double dt = s.t_stop - s.t_start;
  double loc[2] = {static_cast<double>(s.iterations), dt};
  double mx[2], mn[2];
  PARTHENON_MPI_CHECK(MPI_Reduce(loc, mx, 2, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
  PARTHENON_MPI_CHECK(MPI_Reduce(loc, mn, 2, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD));
  if (Globals::my_rank == 0) {
    std::printf("MPI progress thread: %.0f-%.0f iterations over %.3f s, mean period "
                "%.2f-%.2f us\n",
                mn[0], mx[0], dt, 1e6 * dt / std::max(mx[0], 1.0),
                1e6 * dt / std::max(mn[0], 1.0));
    std::fflush(stdout);
  }
  PARTHENON_MPI_CHECK(MPI_Comm_free(&s.comm));
#endif
}

bool Running() { return S().running; }

} // namespace mpi_progress
} // namespace parthenon
