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

// Lightweight tracing of coalesced boundary communication for diagnosing strong scaling
// problems. All events are recorded into preallocated host memory and written to per-rank
// CSV files at the end of the trace window, so tracing does not do any I/O or MPI calls
// during a step. Controlled by the <parthenon/comms_trace> input block:
//
//   enabled          (false) record per-exchange and per-step timing
//   census           (false) print the coalesced message census after each buffer build
//   census_messages  (true)  include the full per-message list in the printed census
//   cycle_start      (0)     first cycle to trace
//   cycle_end        (-1)    first cycle not traced (< 0 traces until the end of the run)
//   summary_interval (0)     print a per-cycle summary every this many cycles (0 = never)
//   prefix           ("comms_trace") prefix of the output files
//   max_records      (4000000) maximum number of exchange records kept per rank
//   long_task_ms     (1.0)   record every task call that takes longer than this

#ifndef UTILS_COMMS_TRACE_HPP_
#define UTILS_COMMS_TRACE_HPP_

#include <cstdint>
#include <string>
#include <vector>

namespace parthenon {
class ParameterInput;

namespace comms_trace {

// Time stamps (seconds) of one coalesced message exchange as seen from one rank. Unset
// times are negative.
//   send: t[0] pack start, t[1] fence done, t[2] data Isend posted,
//         t[3] status Isend posted, t[4] completion first observed (lazy, only checked
//         when the buffer is next needed)
//   recv: t[0] ready (first poll with all var-boundary buffers stale),
//         t[1] status envelope seen, t[2] data envelope seen, t[3] status received,
//         t[4] data received, t[5] unpack launched
struct ExchangeRecord {
  static constexpr int NT = 6;
  int cycle{-1};
  int build{-1};    // Index of the comm buffer build (see NewBuild)
  int in_step{0};   // 1 if inside Step(), 0 if after it (e.g. during remeshing)
  int exchange{-1}; // Per-buffer exchange counter
  int is_send{0};
  int peer{-1};
  int btype{-1};
  int lane{-1};
  int partition_tag{-1};
  int nbuf{0};
  std::int64_t bytes{0};
  int npolls{0};
  double max_poll_gap{0.0};
  double t[NT]{-1.0, -1.0, -1.0, -1.0, -1.0, -1.0};
};

// Per-rank, per-cycle summary
struct StepRecord {
  int cycle{-1};
  double t_start{-1.0};     // before Step()
  double t_step_done{-1.0}; // after Step()
  double t_end{-1.0};       // after SetGlobalTimeStep()
  double allreduce_time{0.0};
  double task_complete_time{0.0};
  double task_incomplete_time{0.0};
  std::int64_t ncalls_complete{0};
  std::int64_t ncalls_incomplete{0};
  double fence_time{0.0};
  double max_recv_wait{0.0}; // max over receives of (data received - ready)
  int max_recv_wait_peer{-1};
  int max_recv_wait_btype{-1};
  int nexchanges{0};
};

// Monotonic wall clock time in seconds (steady_clock, shared by processes on a node)
double Now();

// Read the input parameters and work out which node each rank is on. Collective.
void Initialize(ParameterInput *pin);

bool CensusEnabled();
bool CensusPrintMessages();
const std::string &Prefix();
int NodeOfRank(int rank);
int NumNodes();
const std::string &HostOfRank(int rank);

// True if new exchanges should be traced (enabled and inside the trace window)
bool Active();
// True if records may still be added (exchanges opened in the window can finish later)
bool Recording();

int CurrentCycle();
bool InStep();

// Called every time the comm buffers are rebuilt, returns the index of the new build.
// Must be called on every rank in the same order.
int NewBuild();
int CurrentBuild();

// Driver hooks, all collective
void BeginStep(int cycle);
void StepDone();
void EndStep();
void Finalize();

// Accumulators for the current step
void AddTaskTime(bool complete, double dt);
void AddFenceTime(double dt);
void AddAllreduceTime(double dt);

// Task calls longer than this (seconds) should be passed to RecordTask
double LongTaskThreshold();
// Record a single long task call starting at t0 and taking dt. fn is the address of the
// task function (or nullptr), which is resolved to a symbol name when writing the files.
// Thread safe.
void RecordTask(const std::string &label, const void *fn, double t0, double dt,
                bool complete);

// Store a finished (or, for sends, posted) exchange. Returns a pointer to the stored copy
// that stays valid for the rest of the run so that late events can be filled in, or
// nullptr if the record was not stored.
ExchangeRecord *RecordExchange(const ExchangeRecord &rec);

} // namespace comms_trace
} // namespace parthenon

#endif // UTILS_COMMS_TRACE_HPP_
