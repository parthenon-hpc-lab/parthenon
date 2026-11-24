# Remesh benchmark harness (develop vs PR #1499)

Benchmark-only tooling, on the benchmark branch `lroberts36/remesh-bvals-merge-bench`
(PR #1499 merged with `lroberts36/faster-coalesced-comms`). Not for merging.

## Quick start on Rocinante (Slurm, Cray PE)
    git clone -b lroberts36/remesh-bvals-merge-bench \
      https://github.com/parthenon-hpc-lab/parthenon.git parthenon-bench
    cd parthenon-bench/remesh-bench
    rocinante/setup.sh                       # worktrees, timer patch, 3 builds, Kokkos Tools
    sbatch -N 1 -t 0:30:00 -p <partition> rocinante/bench.sbatch --only b32 --ranks 8 --reps 1
    sbatch -A xd -p standard -N 1 -t 3:00:00 rocinante/bench.sbatch --ranks 8 56 112      # ~1.5 h
    sbatch -A xd -p standard -N 8 -t 2:00:00 rocinante/bench.sbatch --ranks 224 448 896   # ~45 min
- Modules are in `rocinante/env.sh` (PrgEnv-intel, cray-mpich, craype-x86-spr, cmake 3.31.6,
  cray-python). Kokkos 5 needs C++20; `setup.sh` checks that and suggests PrgEnv-gnu if not.
- `--machine rocinante` (`MACHINES` in `bench.py`): shapes b32 (128^3), b32m (256^3), b32l
  (512^3), b64 (256^3), b64m (512^3), at 8-896 ranks (runs of about a minute or less), 112 ranks per node with `-c 2
  --cpu-bind=cores` (a core, both hyperthreads, per rank). Default variants develop, pr_rw,
  merge, merge_pt; num_vars 1 and 4; coalesced on/off; nlim 200; 2 reps from the sbatch.
- The RSS watchdog cannot see ranks on other nodes, so on Slurm only the per-run timeout
  (900 s) applies and memory is left to Slurm. Cases needing more nodes than the job has
  are skipped. The job's first lines show each rank's CPU mask, which decides whether the
  progress thread can pin itself to the sibling hyperthread.
- Results: `results-rocinante-<jobid>/results.csv` plus per-run `out.log`.

## Layout
- `src-develop/`, `src-pr/`, `src-merge/` – git worktrees (merge base `c348b850a`, PR head
  `d1a0ed773`, this branch), with timers inserted by `apply_patch.py` (uncommitted; `git
  diff` there to see). On the Mac these live in `~/Coding/remesh-bench`, the merge tree in
  `/tmp/merge-try`.
- `bench_remesh_timer.hpp` – timer header copied into `src/mesh/` by `apply_patch.py`.
  Prints one `REMESH_TIMING` line per remesh: per-span times (max over ranks), block counts
  (moved/refined/derefined/cross-rank), and `msgs_<span>`/`bytes_<span>` (summed over ranks)
  from a PMPI `MPI_Isend` shim. Spans: `d_*` develop only (restrict, send, recv), `p_*` PR only
  (nbrs, halo, reset, tagmap, bufs, cinfo, post, send, alloc, recv, unpack, sendwait,
  counters),
  `f_*` shared finalize, plus lb, build, prolong, wait. `bench.py` groups `d_*`/`p_*` as
  `xfer` and `f_*` as `fin`.
- `build-develop/`, `build-pr/`, `build-merge/` – Release, MPI, Kokkos Serial, no HDF5.
- `kt-install/` – Kokkos Tools (`libkp_space_time_stack`, `libkp_memory_events`, `libkp_hwm_mpi`, ...).
  space-time-stack rebuilt with its hard-coded `output_threshold` lowered from 0.1 to 0.001
  (`kokkos-tools/profiling/space-time-stack/kp_space_time_stack.cpp:35`) so small remesh
  kernels show up.
- `parthinput.remesh_bench` – 3D hard-sphere advection deck; `bench.py` overrides it.
- `bench.py` – guarded sweep, writes `results/results.csv`.
- `probe.py` – one-off guarded run: `./probe.py <develop|pr> <np> <cap_gb> <timeout_s> key=val ...`.

## Safety
Every run is watched: summed RSS of the whole mpirun tree is sampled every 0.2 s (`mem.csv`
in the run dir); over `--mem-limit-gb` (default 8) or `--timeout` it kills every process
in the tree (Open MPI ranks are not in mpirun's process group, so killpg is not enough).
A memory kill stops the sweep unless `--keep-going`.

## Environment knobs (read at run time, no rebuild)
- `BENCH_BARRIER=1` (`bench.py --barrier`): MPI_Barrier after every mark; waiting goes to `wait`.
- `BENCH_PER_RANK=1`: rank 0 also prints a `REMESH_RANK` line per rank (gathered, not interleaved).
- `BENCH_WAIT_SENDS=1` (PR only, experiment; variant `pr_rw`): `ExchangeBoundaryBuffers`
  spins until its remesh sends (plain and coalesced) complete, like develop's
  `MPI_Waitall`. The merge branch does this unconditionally.
- `PARTHENON_MPI_PROGRESS_US`, `PARTHENON_MPI_PROGRESS_CPU` (merge only): progress thread,
  see `src/utils/mpi_progress.hpp`. Cray MPICH also needs `MPICH_MAX_THREAD_SAFETY=multiple`
  (the merge_pt and merge_ap variants set it).

## Results on the Mac (2026-11-23, 1 rep each, all `hst_identical`)
- `results-barrier/` barrier sweep; `results/` no barrier; `results-waitsends/` no barrier +
  `BENCH_WAIT_SENDS=1`; `results-waitsends-nocoal/` same but the wait ignored coalesced sends.
- Without the send wait the PR stalls ~5-12 ms in `p_recv`/`p_counters` on 4 ranks whenever
  messages exceed the eager limit: sends are left in flight while the sender does MPI-free
  host work (SetMeshBlockNeighbors etc.), so rendezvous transfers don't progress.
- With it, the remaining PR cost is mostly `p_unpack` (SetBounds general path) plus pack;
  `p_reset` is cost moved from `f_bufs`, not extra.

## Merge experiment: remesh-bvals-channels + faster-coalesced-comms (2026-11-23)
- The merge commit of this branch (`origin/lroberts36/faster-coalesced-comms` 4653525c4 into
  the PR). Besides the conflict resolution it gives remesh MeshData real
  `BlockListPartition`s (`partition = 0` silently became nullptr), makes
  `coalesced_wait_for_sends` skip remesh transfers (send and receive MeshData differ), and
  makes `ExchangeBoundaryBuffers` wait for its sends.
- `bench.py --variants ...` picks from `VARIANTS` (develop always runs as the reference):
  pr_rw, merge, merge_ws (+ coalesced_wait_for_sends), merge_pt
  (PARTHENON_MPI_PROGRESS_US=10), merge_ap (MPICH_ASYNC_PROGRESS=1). Mac results in
  `results-merge/` predate the built-in send wait: their merge_rw is today's merge, their
  merge is the merge without the wait.
- Findings: SetBounds fast path cuts unpack ~5x; with the remesh send wait the 4-rank
  stalls are gone and what remains of the PR's remesh xfer is p_reset (recovered in f_bufs).
  The fast path also speeds ordinary halo exchange, so merged totals beat develop.

## Usage
    ./bench.py --dry-run
    ./bench.py --machine mac [--barrier] [--outdir DIR]
    KOKKOS_TOOLS_LIBS=$PWD/kt-install/lib/libkp_space_time_stack.dylib ./probe.py pr 4 6 120

Rebuilding after a branch moves:
    git -C src-pr checkout --detach <sha> -- . && ./apply_patch.py src-pr   # or re-create worktree
    cmake --build build-pr --target advection-example -j 10

## Open MPI btl/sm memory growth on macOS (both branches) — fixed by eager limit
`bench.py`/`probe.py` pass `--mca btl_sm_eager_limit 32768` on the Mac (`MPI_ARGS["mac"]`;
`btl_vader_*` is an alias). With the default 4096, `Advection/num_vars > 1` 4-rank AMR runs
grew host memory without bound (~0.6 GB/cycle; `opal_free_list_grow_st` <-
`mca_btl_sm_frag_alloc` from `MPI_Iprobe`/`MPI_Test` in halo exchange). This is what took the
Mac to ~300 GB. With the flag, np4 nv4 vs4 peaks at ~3.5 GB. `PROBE_MPI_ARGS` overrides.

## Deferred allocation (bench branch, 2026-11-23)
- New blocks are made with `MeshBlock::Make(..., defer_allocation=true)` (variables but no
  data, no application data) and allocated by `AllocateDeferred` from a callback in
  `ExchangeBoundaryBuffers` after the remesh sends are posted (span `p_alloc`; `build` is
  now just block metadata). Coalesced receives are posted (`TryReceiveAny`) right after
  `StartReceiveBoundBufs`.
- The counters exchange wraps the field data exchange (posted before it, waited after), so
  `p_counters` is only the Waitall and unpack.
- Non-coalesced sends send the packed size (`BndInfo::size()`), not the pool capacity.
- The send wait is now `parthenon/mesh/remesh_wait_for_sends` (default false, like
  develop); variant `merge_rw` sets it. On the Mac (Open MPI sm) merge without it stalls
  ~5-10 ms in `p_recv` as before; `results-deferred/` has the np4 matrix, all histories
  identical to develop.
