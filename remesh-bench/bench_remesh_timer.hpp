// Benchmark-only instrumentation for comparing remesh cost between branches.
// NOT for upstreaming. Copied into src/mesh/ of the benchmark worktrees by apply_patch.py.
//
// Each remesh event prints one line on rank 0:
//   REMESH_TIMING event=.. nb_old=.. nb_new=.. n_move=.. n_ref=.. n_deref=.. n_xrank=..
//                 barrier=0|1 <span>=<s> ... total=<s> [msgs_<span>=<n> bytes_<span>=<n>]
// Span times are seconds, max over ranks (so they do not sum to total). Spans prefixed
// d_ only occur on develop, p_ only on the bvals-remesh branch, f_ are the shared
// finalize steps. msgs/bytes count MPI_Isend calls (summed over ranks) made during the
// span, printed only when nonzero.
//
// BENCH_BARRIER=1 puts an MPI_Barrier after every mark, so time a rank spends waiting
// for slower ranks goes into "wait" instead of into the next span that communicates.
// BENCH_PER_RANK=1 additionally prints one REMESH_RANK line per rank per event.
// BENCH_WAIT_SENDS=1 (experiment, bvals-remesh branch only): ExchangeBoundaryBuffers waits
// for its own MPI sends to complete before returning, as develop does with MPI_Waitall.
#ifndef MESH_BENCH_REMESH_TIMER_HPP_
#define MESH_BENCH_REMESH_TIMER_HPP_

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <Kokkos_Core.hpp>

#include "globals.hpp"
#include "mesh/forest/logical_location.hpp"
#include "parthenon_mpi.hpp"

namespace parthenon {
namespace bench {

#define BENCH_SPANS(X)                                                                   \
  X(lb)                                                                                  \
  X(build)                                                                               \
  X(d_restrict)                                                                          \
  X(d_send)                                                                              \
  X(d_recv)                                                                              \
  X(p_nbrs)                                                                              \
  X(p_halo)                                                                              \
  X(p_reset)                                                                             \
  X(p_tagmap)                                                                            \
  X(p_bufs)                                                                              \
  X(p_cinfo)                                                                             \
  X(p_post)                                                                              \
  X(p_send)                                                                              \
  X(p_alloc)                                                                             \
  X(p_recv)                                                                              \
  X(p_unpack)                                                                            \
  X(p_sendwait)                                                                          \
  X(p_counters)                                                                          \
  X(prolong)                                                                             \
  X(f_misc)                                                                              \
  X(f_nbrs)                                                                              \
  X(f_bufs)                                                                              \
  X(f_comm)                                                                              \
  X(f_rest)                                                                              \
  X(wait)

#define BENCH_ENUM(name) name,
#define BENCH_NAME(name) #name,
enum Span : int { BENCH_SPANS(BENCH_ENUM) NSPAN };
inline const char *SpanName(int s) {
  static const char *names[] = {BENCH_SPANS(BENCH_NAME)};
  return names[s];
}
#undef BENCH_ENUM
#undef BENCH_NAME

struct RemeshTimer {
  using clock = std::chrono::steady_clock;
  clock::time_point start, last;
  std::array<double, NSPAN> t{};
  std::array<long long, NSPAN> msgs{}, bytes{};
  long long pend_msgs = 0, pend_bytes = 0;
  std::array<int, 6> counts{};
  const bool barrier;

  static int &Event() {
    static int event = 0;
    return event;
  }
  // The live timer, if any; lets the MPI_Isend shim and helper functions find it
  static RemeshTimer *&Current() {
    static RemeshTimer *current = nullptr;
    return current;
  }
  static double Dt(clock::time_point a, clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
  }
  static void Barrier() {
#ifdef MPI_PARALLEL
    MPI_Barrier(MPI_COMM_WORLD);
#endif
  }
  static bool BarrierRequested() {
    const char *e = std::getenv("BENCH_BARRIER");
    return e != nullptr && std::strcmp(e, "0") != 0;
  }

  RemeshTimer() : barrier(BarrierRequested()) {
    Kokkos::fence();
    Barrier();
    start = last = clock::now();
    Current() = this;
  }
  ~RemeshTimer() { Current() = nullptr; }

  // Close the segment that ends here and attribute it (and its sends) to span s. Must be
  // reached by every rank in the same order when BENCH_BARRIER is set.
  void Mark(Span s) {
    Kokkos::fence();
    auto now = clock::now();
    t[s] += Dt(last, now);
    msgs[s] += pend_msgs;
    bytes[s] += pend_bytes;
    pend_msgs = pend_bytes = 0;
    if (barrier) {
      Barrier();
      const auto after = clock::now();
      t[wait] += Dt(now, after);
      now = after;
    }
    last = now;
  }

  // Classify the new blocks by how their data gets there (identical on all ranks)
  void Count(const std::vector<LogicalLocation> &oldloc,
             const std::vector<LogicalLocation> &newloc, const std::vector<int> &oldrank,
             const std::vector<int> &newrank, const std::vector<int> &newtoold) {
    counts = {static_cast<int>(oldloc.size()), static_cast<int>(newloc.size()), 0, 0, 0,
              0};
    for (int nn = 0; nn < newloc.size(); ++nn) {
      const int on = newtoold[nn];
      const bool xrank = newrank[nn] != oldrank[on];
      if (newloc[nn].level() == oldloc[on].level()) {
        if (xrank) counts[2]++;
      } else if (newloc[nn].level() > oldloc[on].level()) {
        counts[3]++;
      } else {
        counts[4]++;
      }
      if (xrank) counts[5]++;
    }
  }

  void Finish() {
    Mark(f_rest);
    Kokkos::fence();
    Barrier();
    const double total = Dt(start, clock::now());
    Current() = nullptr;
    std::array<double, NSPAN + 1> tl, tmax;
    std::copy(t.begin(), t.end(), tl.begin());
    tl[NSPAN] = total;
    tmax = tl;
    std::array<long long, 2 * NSPAN> nl, nsum;
    std::copy(msgs.begin(), msgs.end(), nl.begin());
    std::copy(bytes.begin(), bytes.end(), nl.begin() + NSPAN);
    nsum = nl;
#ifdef MPI_PARALLEL
    MPI_Reduce(tl.data(), tmax.data(), NSPAN + 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(nl.data(), nsum.data(), 2 * NSPAN, MPI_LONG_LONG, MPI_SUM, 0,
               MPI_COMM_WORLD);
#endif
    const int ev = ++Event();
    // BENCH_PER_RANK=1: rank 0 also prints every rank's own (unreduced) nonzero spans.
    // Gathered rather than printed per rank, since mpirun splits long lines from
    // different ranks into interleaved chunks.
    const char *per_rank = std::getenv("BENCH_PER_RANK");
    if (per_rank != nullptr && std::strcmp(per_rank, "0") != 0) {
      std::vector<double> all(tl.begin(), tl.end());
#ifdef MPI_PARALLEL
      all.resize((NSPAN + 1) * Globals::nranks);
      MPI_Gather(tl.data(), NSPAN + 1, MPI_DOUBLE, all.data(), NSPAN + 1, MPI_DOUBLE, 0,
                 MPI_COMM_WORLD);
#endif
      for (int r = 0; Globals::my_rank == 0 && r < all.size() / (NSPAN + 1); ++r) {
        std::printf("REMESH_RANK event=%d rank=%d", ev, r);
        for (int s = 0; s < NSPAN + 1; ++s) {
          const double v = all[r * (NSPAN + 1) + s];
          if (v > 0.0) std::printf(" %s=%.6e", s < NSPAN ? SpanName(s) : "total", v);
        }
        std::printf("\n");
      }
    }
    if (Globals::my_rank == 0) {
      // One printf per line so output from other ranks cannot land in the middle
      char line[8192];
      const int len = sizeof(line);
      int n = std::snprintf(line, len,
                            "REMESH_TIMING event=%d nb_old=%d nb_new=%d n_move=%d "
                            "n_ref=%d n_deref=%d n_xrank=%d barrier=%d",
                            ev, counts[0], counts[1], counts[2], counts[3], counts[4],
                            counts[5], static_cast<int>(barrier));
      for (int s = 0; s < NSPAN && n < len; ++s)
        n += std::snprintf(line + n, len - n, " %s=%.6e", SpanName(s), tmax[s]);
      if (n < len) n += std::snprintf(line + n, len - n, " total=%.6e", tmax[NSPAN]);
      for (int s = 0; s < NSPAN && n < len; ++s) {
        if (nsum[s] > 0 || nsum[NSPAN + s] > 0)
          n += std::snprintf(line + n, len - n, " msgs_%s=%lld bytes_%s=%lld",
                             SpanName(s), nsum[s], SpanName(s), nsum[NSPAN + s]);
      }
      std::printf("%s\n", line);
      std::fflush(stdout);
    }
  }
};

inline bool WaitSendsRequested() {
  static const bool wait = [] {
    const char *e = std::getenv("BENCH_WAIT_SENDS");
    return e != nullptr && std::strcmp(e, "0") != 0;
  }();
  return wait;
}

// Mark on the live timer from code that cannot see it (no-op outside a remesh)
inline void MarkCurrent(Span s) {
  if (RemeshTimer::Current() != nullptr) RemeshTimer::Current()->Mark(s);
}

} // namespace bench
} // namespace parthenon

#if defined(MPI_PARALLEL) && !defined(BENCH_NO_PMPI)
// PMPI shim: count every MPI_Isend issued while a remesh timer is live. Defined in the
// one translation unit that includes this header; the static libparthenon object wins
// over libmpi at link time, so all call sites in the executable go through it.
extern "C" int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest,
                         int tag, MPI_Comm comm, MPI_Request *request) {
  if (auto *timer = parthenon::bench::RemeshTimer::Current()) {
    int size = 0;
    PMPI_Type_size(datatype, &size);
    timer->pend_msgs++;
    timer->pend_bytes += static_cast<long long>(count) * size;
  }
  return PMPI_Isend(buf, count, datatype, dest, tag, comm, request);
}
#endif

#endif // MESH_BENCH_REMESH_TIMER_HPP_
