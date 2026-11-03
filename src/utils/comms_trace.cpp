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

#include "utils/comms_trace.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <limits>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include <cxxabi.h>
#include <dlfcn.h>

#include "globals.hpp"
#include "parameter_input.hpp"
#include "parthenon_mpi.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {
namespace comms_trace {

namespace {
struct ClockSync {
  double local{0.0};  // Local time at which the offset was measured
  double offset{0.0}; // Add to local time to get rank 0 time
  double rtt{0.0};    // Round trip time of the best ping-pong
};

struct TaskRecord {
  int cycle;
  int in_step;
  double t0;
  double dt;
  int complete;
  const void *fn;
  std::string label;
};

// Name of the function at addr, its module and its offset in the module (for addr2line
// when the symbol is not exported, e.g. in an executable linked without -rdynamic)
struct FunctionInfo {
  std::string name{"?"};
  std::string module{"?"};
  std::uintptr_t offset{0};
};

FunctionInfo ResolveFunction(const void *addr) {
  FunctionInfo out;
  if (addr == nullptr) return out;
  Dl_info info;
  if (dladdr(addr, &info) == 0) return out;
  if (info.dli_fname != nullptr) out.module = info.dli_fname;
  out.offset = reinterpret_cast<std::uintptr_t>(addr) -
               reinterpret_cast<std::uintptr_t>(info.dli_fbase);
  if (info.dli_sname != nullptr &&
      reinterpret_cast<std::uintptr_t>(info.dli_saddr) ==
          reinterpret_cast<std::uintptr_t>(addr)) {
    int status;
    char *demangled = abi::__cxa_demangle(info.dli_sname, nullptr, nullptr, &status);
    out.name = (status == 0 && demangled != nullptr) ? demangled : info.dli_sname;
    std::free(demangled);
  }
  return out;
}

struct State {
  bool initialized{false};
  bool enabled{false};
  bool census{false};
  bool census_messages{true};
  int cycle_start{0};
  int cycle_end{-1};
  int summary_interval{0};
  std::string prefix{"comms_trace"};
  std::size_t max_records{4000000};
  double long_task{1.0e-3};

  bool started{false};
  bool finalized{false};
  bool step_open{false};
  bool in_step{false};
  int cycle{-1};
  int build{-1};

  std::vector<int> node_of_rank{0};
  std::vector<std::string> hosts{"localhost"};
  int nnodes{1};

  // deque so that growing never copies existing records in the middle of a step
  std::deque<ExchangeRecord> exchanges;
  std::size_t dropped{0};
  std::vector<StepRecord> steps;
  std::vector<TaskRecord> tasks;
  std::mutex tasks_mutex;
  StepRecord cur;
  std::size_t summary_from{0};

  ClockSync sync_start, sync_end;
#ifdef MPI_PARALLEL
  MPI_Comm comm{MPI_COMM_NULL};
#endif
};

State &S() {
  static State s;
  return s;
}

const char *BTypeName(int btype) {
  static const char *names[] = {"local",       "nonlocal",          "any",
                                "flxcor_send", "flxcor_recv",       "gmg_same",
                                "gmg_restrict_send", "gmg_restrict_recv",
                                "gmg_prolongate_send", "gmg_prolongate_recv"};
  if (btype < 0 || btype >= static_cast<int>(sizeof(names) / sizeof(names[0])))
    return "unknown";
  return names[btype];
}

// Estimate the offset of the local clock relative to rank 0 with ping-pongs, keeping the
// one with the smallest round trip time. Collective.
ClockSync SyncClocks() {
  ClockSync out;
#ifdef MPI_PARALLEL
  auto &s = S();
  constexpr int ntrials = 20;
  PARTHENON_MPI_CHECK(MPI_Barrier(s.comm));
  if (Globals::my_rank == 0) {
    for (int r = 1; r < Globals::nranks; ++r) {
      for (int k = 0; k < ntrials; ++k) {
        double dummy;
        PARTHENON_MPI_CHECK(
            MPI_Recv(&dummy, 1, MPI_DOUBLE, r, 0, s.comm, MPI_STATUS_IGNORE));
        double t = Now();
        PARTHENON_MPI_CHECK(MPI_Send(&t, 1, MPI_DOUBLE, r, 0, s.comm));
      }
    }
    out.local = Now();
  } else {
    out.rtt = std::numeric_limits<double>::max();
    for (int k = 0; k < ntrials; ++k) {
      double t0 = Now();
      double ts;
      PARTHENON_MPI_CHECK(MPI_Send(&t0, 1, MPI_DOUBLE, 0, 0, s.comm));
      PARTHENON_MPI_CHECK(MPI_Recv(&ts, 1, MPI_DOUBLE, 0, 0, s.comm, MPI_STATUS_IGNORE));
      double t1 = Now();
      if (t1 - t0 < out.rtt) {
        out.rtt = t1 - t0;
        out.local = 0.5 * (t0 + t1);
        out.offset = ts - out.local;
      }
    }
  }
  PARTHENON_MPI_CHECK(MPI_Barrier(s.comm));
#else
  out.local = Now();
#endif
  return out;
}

// Print a per-cycle summary of the steps recorded since the last summary. Collective.
void PrintSummary() {
  auto &s = S();
  const int nsteps = static_cast<int>(s.steps.size() - s.summary_from);
  if (nsteps <= 0) return;
  constexpr int NV = 7;
  std::vector<double> local(NV * nsteps);
  for (int i = 0; i < nsteps; ++i) {
    const auto &st = s.steps[s.summary_from + i];
    local[NV * i + 0] = st.t_step_done - st.t_start;
    local[NV * i + 1] = st.allreduce_time;
    local[NV * i + 2] = st.task_incomplete_time;
    local[NV * i + 3] = st.fence_time;
    local[NV * i + 4] = st.max_recv_wait;
    local[NV * i + 5] = st.max_recv_wait_peer;
    local[NV * i + 6] = st.max_recv_wait_btype;
  }
  std::vector<double> all;
  if (Globals::my_rank == 0) all.resize(NV * nsteps * Globals::nranks);
#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(MPI_Gather(local.data(), NV * nsteps, MPI_DOUBLE, all.data(),
                                 NV * nsteps, MPI_DOUBLE, 0, s.comm));
#else
  all = local;
#endif
  if (Globals::my_rank == 0) {
    std::printf("comms_trace summary (times in ms; the rank with the smallest allreduce "
                "wait is the last to finish its step)\n");
    std::printf("%8s | %8s %5s %8s | %8s %5s %8s %5s | %8s %5s | %8s %5s | %8s %11s %s\n",
                "cycle", "step_max", "rank", "step_min", "ared_min", "rank", "ared_max",
                "rank", "spin_max", "rank", "fence_mx", "rank", "recv_wt", "rank<-peer",
                "btype");
    for (int i = 0; i < nsteps; ++i) {
      auto v = [&](int r, int k) { return all[NV * (nsteps * r + i) + k]; };
      int r_step_max = 0, r_step_min = 0, r_ared_min = 0, r_ared_max = 0, r_spin = 0,
          r_fence = 0, r_wait = 0;
      for (int r = 1; r < Globals::nranks; ++r) {
        if (v(r, 0) > v(r_step_max, 0)) r_step_max = r;
        if (v(r, 0) < v(r_step_min, 0)) r_step_min = r;
        if (v(r, 1) < v(r_ared_min, 1)) r_ared_min = r;
        if (v(r, 1) > v(r_ared_max, 1)) r_ared_max = r;
        if (v(r, 2) > v(r_spin, 2)) r_spin = r;
        if (v(r, 3) > v(r_fence, 3)) r_fence = r;
        if (v(r, 4) > v(r_wait, 4)) r_wait = r;
      }
      char rp[32];
      if (v(r_wait, 5) < 0)
        std::snprintf(rp, sizeof(rp), "-");
      else
        std::snprintf(rp, sizeof(rp), "%d<-%d", r_wait, static_cast<int>(v(r_wait, 5)));
      std::printf(
          "%8d | %8.3f %5d %8.3f | %8.3f %5d %8.3f %5d | %8.3f %5d | %8.3f %5d | %8.3f "
          "%11s %s\n",
          s.steps[s.summary_from + i].cycle, 1e3 * v(r_step_max, 0), r_step_max,
          1e3 * v(r_step_min, 0), 1e3 * v(r_ared_min, 1), r_ared_min,
          1e3 * v(r_ared_max, 1), r_ared_max, 1e3 * v(r_spin, 2), r_spin,
          1e3 * v(r_fence, 3), r_fence, 1e3 * v(r_wait, 4), rp,
          v(r_wait, 5) < 0 ? "-" : BTypeName(static_cast<int>(v(r_wait, 6))));
    }
    std::fflush(stdout);
  }
  s.summary_from = s.steps.size();
}

void WriteFiles() {
  auto &s = S();
  const int rank = Globals::my_rank;
  {
    const std::string fname = s.prefix + ".exchanges." + std::to_string(rank) + ".csv";
    FILE *f = std::fopen(fname.c_str(), "w");
    if (f != nullptr) {
      std::fprintf(f, "cycle,build,in_step,exchange,is_send,rank,peer,btype,lane,"
                      "partition_tag,nbuf,bytes,npolls,max_poll_gap,t0,t1,t2,t3,t4,t5\n");
      for (const auto &e : s.exchanges) {
        std::fprintf(f, "%d,%d,%d,%d,%d,%d,%d,%s,%d,%d,%d,%lld,%d,%.9f", e.cycle, e.build,
                     e.in_step, e.exchange, e.is_send, rank, e.peer, BTypeName(e.btype),
                     e.lane, e.partition_tag, e.nbuf, static_cast<long long>(e.bytes),
                     e.npolls, e.max_poll_gap);
        for (int n = 0; n < ExchangeRecord::NT; ++n)
          std::fprintf(f, ",%.9f", e.t[n]);
        std::fprintf(f, "\n");
      }
      std::fclose(f);
    }
  }
  {
    const std::string fname = s.prefix + ".steps." + std::to_string(rank) + ".csv";
    FILE *f = std::fopen(fname.c_str(), "w");
    if (f != nullptr) {
      std::fprintf(f, "cycle,rank,t_start,t_step_done,t_end,allreduce,task_complete,"
                      "task_incomplete,ncalls_complete,ncalls_incomplete,fence,"
                      "max_recv_wait,max_recv_wait_peer,max_recv_wait_btype,"
                      "nexchanges\n");
      for (const auto &st : s.steps) {
        std::fprintf(f, "%d,%d,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%lld,%lld,%.9f,%.9f,%d,%s,%d\n",
                     st.cycle, rank, st.t_start, st.t_step_done, st.t_end,
                     st.allreduce_time, st.task_complete_time, st.task_incomplete_time,
                     static_cast<long long>(st.ncalls_complete),
                     static_cast<long long>(st.ncalls_incomplete), st.fence_time,
                     st.max_recv_wait, st.max_recv_wait_peer,
                     BTypeName(st.max_recv_wait_btype), st.nexchanges);
      }
      std::fclose(f);
    }
  }

  {
    const std::string fname = s.prefix + ".tasks." + std::to_string(rank) + ".csv";
    FILE *f = std::fopen(fname.c_str(), "w");
    if (f != nullptr) {
      std::fprintf(f, "cycle,rank,in_step,t0,dt,complete,label,function,module,offset,"
                      "addr\n");
      std::map<const void *, FunctionInfo> fns;
      for (const auto &t : s.tasks) {
        if (fns.count(t.fn) == 0) fns[t.fn] = ResolveFunction(t.fn);
        const auto &fi = fns.at(t.fn);
        auto quote = [](std::string str) {
          std::replace(str.begin(), str.end(), '"', '\'');
          return "\"" + str + "\"";
        };
        std::fprintf(f, "%d,%d,%d,%.9f,%.9f,%d,%s,%s,%s,0x%llx,0x%llx\n", t.cycle, rank,
                     t.in_step, t.t0, t.dt, t.complete, quote(t.label).c_str(),
                     quote(fi.name).c_str(), quote(fi.module).c_str(),
                     static_cast<unsigned long long>(fi.offset),
                     static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(t.fn)));
      }
      std::fclose(f);
    }
  }

  // Clock offsets, gathered to rank 0
  constexpr int NC = 7;
  double loc[NC] = {s.sync_start.local, s.sync_start.offset, s.sync_start.rtt,
                    s.sync_end.local,   s.sync_end.offset,   s.sync_end.rtt,
                    static_cast<double>(s.dropped)};
  std::vector<double> all(Globals::my_rank == 0 ? NC * Globals::nranks : 0);
#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(
      MPI_Gather(loc, NC, MPI_DOUBLE, all.data(), NC, MPI_DOUBLE, 0, s.comm));
#else
  all.assign(loc, loc + NC);
#endif
  if (Globals::my_rank == 0) {
    const std::string fname = s.prefix + ".clock.csv";
    FILE *f = std::fopen(fname.c_str(), "w");
    if (f != nullptr) {
      std::fprintf(f, "rank,host,node,sync0_local,sync0_offset,sync0_rtt,sync1_local,"
                      "sync1_offset,sync1_rtt,dropped_records\n");
      for (int r = 0; r < Globals::nranks; ++r) {
        const double *c = &all[NC * r];
        std::fprintf(f, "%d,%s,%d,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%lld\n", r,
                     HostOfRank(r).c_str(), NodeOfRank(r), c[0], c[1], c[2], c[3], c[4],
                     c[5], static_cast<long long>(c[6]));
      }
      std::fclose(f);
    }
    std::printf(
        "comms_trace: wrote %s.{exchanges,steps,tasks}.<rank>.csv and %s.clock.csv\n",
                s.prefix.c_str(), s.prefix.c_str());
    std::fflush(stdout);
  }
}
} // namespace

double Now() {
  // MPI_Wtime can have only microsecond resolution (e.g. Open MPI on macOS)
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

void Initialize(ParameterInput *pin) {
  auto &s = S();
  if (s.initialized) return;
  s.initialized = true;
  const std::string block = "parthenon/comms_trace";
  s.enabled = pin->GetOrAddBoolean(block, "enabled", false,
                                   "Record timing of coalesced communication");
  s.census = pin->GetOrAddBoolean(block, "census", false,
                                  "Print the coalesced message census after each "
                                  "buffer build");
  s.census_messages = pin->GetOrAddBoolean(
      block, "census_messages", true, "Include every message in the printed census");
  s.cycle_start = pin->GetOrAddInteger(block, "cycle_start", 0, "First traced cycle");
  s.cycle_end =
      pin->GetOrAddInteger(block, "cycle_end", -1, "First cycle that is not traced");
  s.summary_interval = pin->GetOrAddInteger(
      block, "summary_interval", 0, "Print a summary every this many traced cycles");
  s.prefix = pin->GetOrAddString(block, "prefix", "comms_trace", "Output file prefix");
  s.max_records = pin->GetOrAddInteger(block, "max_records", 4000000,
                                       "Maximum number of exchange records per rank");
  s.long_task = 1.0e-3 * pin->GetOrAddReal(block, "long_task_ms", 1.0,
                                           "Record task calls longer than this (ms)");

#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(MPI_Comm_dup(MPI_COMM_WORLD, &s.comm));
  char name[MPI_MAX_PROCESSOR_NAME + 1] = {0};
  int len;
  PARTHENON_MPI_CHECK(MPI_Get_processor_name(name, &len));
  std::vector<char> names(Globals::nranks * (MPI_MAX_PROCESSOR_NAME + 1));
  PARTHENON_MPI_CHECK(MPI_Allgather(name, MPI_MAX_PROCESSOR_NAME + 1, MPI_CHAR,
                                    names.data(), MPI_MAX_PROCESSOR_NAME + 1, MPI_CHAR,
                                    s.comm));
  s.hosts.resize(Globals::nranks);
  s.node_of_rank.resize(Globals::nranks);
  std::map<std::string, int> node_ids;
  for (int r = 0; r < Globals::nranks; ++r) {
    s.hosts[r] = std::string(&names[r * (MPI_MAX_PROCESSOR_NAME + 1)]);
    if (node_ids.count(s.hosts[r]) == 0) {
      const int id = node_ids.size();
      node_ids[s.hosts[r]] = id;
    }
    s.node_of_rank[r] = node_ids.at(s.hosts[r]);
  }
  s.nnodes = node_ids.size();
#endif
}

bool CensusEnabled() { return S().census; }
bool CensusPrintMessages() { return S().census_messages; }
const std::string &Prefix() { return S().prefix; }
int NodeOfRank(int rank) {
  const auto &s = S();
  return (rank >= 0 && rank < static_cast<int>(s.node_of_rank.size()))
             ? s.node_of_rank[rank]
             : -1;
}
int NumNodes() { return S().nnodes; }
const std::string &HostOfRank(int rank) {
  static const std::string unknown{"unknown"};
  const auto &s = S();
  return (rank >= 0 && rank < static_cast<int>(s.hosts.size())) ? s.hosts[rank] : unknown;
}

bool Active() {
  const auto &s = S();
  return s.step_open && !s.finalized;
}

bool Recording() {
  const auto &s = S();
  return s.enabled && s.started && !s.finalized;
}

int CurrentCycle() { return S().cycle; }
bool InStep() { return S().in_step; }
int NewBuild() { return ++S().build; }
int CurrentBuild() { return S().build; }

void BeginStep(int cycle) {
  auto &s = S();
  if (!s.enabled || s.finalized) return;
  const bool in_window =
      (cycle >= s.cycle_start) && ((s.cycle_end < 0) || (cycle < s.cycle_end));
  if (!in_window) return;
  if (!s.started) {
    s.sync_start = SyncClocks();
    s.started = true;
  }
  s.cycle = cycle;
  s.cur = StepRecord();
  s.cur.cycle = cycle;
  s.step_open = true;
  s.in_step = true;
  s.cur.t_start = Now();
}

void StepDone() {
  auto &s = S();
  if (s.step_open) s.cur.t_step_done = Now();
  s.in_step = false;
}

void EndStep() {
  auto &s = S();
  if (!s.step_open) return;
  s.cur.t_end = Now();
  s.steps.push_back(s.cur);
  s.step_open = false;
  s.in_step = false;
  if ((s.summary_interval > 0) &&
      (s.steps.size() - s.summary_from >= static_cast<std::size_t>(s.summary_interval)))
    PrintSummary();
  if ((s.cycle_end >= 0) && (s.cur.cycle + 1 >= s.cycle_end)) Finalize();
}

void Finalize() {
  auto &s = S();
  if (!s.enabled || s.finalized) return;
  if (!s.started) {
    s.finalized = true;
    return;
  }
  if (s.step_open) EndStep();
  if (s.finalized) return; // EndStep may have finalized
  PrintSummary();
  s.sync_end = SyncClocks();
  s.finalized = true;
  WriteFiles();
}

void AddTaskTime(bool complete, double dt) {
  auto &s = S();
  if (!s.step_open) return;
  if (complete) {
    s.cur.task_complete_time += dt;
    s.cur.ncalls_complete++;
  } else {
    s.cur.task_incomplete_time += dt;
    s.cur.ncalls_incomplete++;
  }
}

void AddFenceTime(double dt) {
  auto &s = S();
  if (s.step_open) s.cur.fence_time += dt;
}

void AddAllreduceTime(double dt) {
  auto &s = S();
  if (s.step_open) s.cur.allreduce_time += dt;
}

double LongTaskThreshold() { return S().long_task; }

void RecordTask(const std::string &label, const void *fn, double t0, double dt,
                bool complete) {
  auto &s = S();
  if (!s.step_open) return;
  std::lock_guard<std::mutex> lock(s.tasks_mutex);
  if (s.tasks.size() >= s.max_records) return;
  s.tasks.push_back({s.cycle, s.in_step ? 1 : 0, t0, dt, complete ? 1 : 0, fn, label});
}

ExchangeRecord *RecordExchange(const ExchangeRecord &rec) {
  auto &s = S();
  if (!Recording()) return nullptr;
  ExchangeRecord *stored = nullptr;
  // push_back on a deque does not invalidate references to existing elements
  if (s.exchanges.size() < s.max_records) {
    s.exchanges.push_back(rec);
    stored = &s.exchanges.back();
  } else {
    s.dropped++;
  }
  if (s.step_open) {
    s.cur.nexchanges++;
    if (!rec.is_send && rec.t[0] >= 0.0 && rec.t[4] >= 0.0) {
      const double wait = rec.t[4] - rec.t[0];
      if (wait > s.cur.max_recv_wait) {
        s.cur.max_recv_wait = wait;
        s.cur.max_recv_wait_peer = rec.peer;
        s.cur.max_recv_wait_btype = rec.btype;
      }
    }
  }
  return stored;
}

} // namespace comms_trace
} // namespace parthenon
