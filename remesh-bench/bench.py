#!/usr/bin/env python3
"""Guarded remesh benchmark sweep: develop vs bvals-remesh branch.

Every run is watched: if the summed RSS of the mpirun process tree exceeds --mem-limit-gb
or the run exceeds --timeout seconds, the whole process group is killed and the case is
marked as such. Results go to <outdir>/results.csv and a summary table on stdout.

  ./bench.py --machine mac                 # default small Mac matrix
  ./bench.py --machine mac --dry-run       # list cases only
  ./bench.py --machine mac --only b8 --stack   # add Kokkos space-time-stack output
  ./bench.py --machine rocinante ...       # inside a Slurm allocation, see rocinante/

With the srun launcher only the timeout is enforced (ranks on other nodes are not
visible to the RSS watchdog); memory is left to Slurm.
"""
import math
import argparse
import csv
import itertools
import os
import re
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DECK = HERE / "parthinput.remesh_bench"
# Variants: name -> (build dir, extra input overrides, extra environment). Every variant's
# history output is compared against develop's.
# Cray MPICH only provides MPI_THREAD_MULTIPLE when asked to; harmless elsewhere
MT = {"MPICH_MAX_THREAD_SAFETY": "multiple"}
VARIANTS = {
    "develop": (HERE / "build-develop", {}, {}),
    "pr": (HERE / "build-pr", {}, {}),
    # BENCH_WAIT_SENDS: ExchangeBoundaryBuffers waits for its remesh sends
    "pr_rw": (HERE / "build-pr", {}, {"BENCH_WAIT_SENDS": "1"}),
    # remesh-bvals-channels merged with faster-coalesced-comms (always waits for its
    # remesh sends, so it corresponds to pr_rw)
    "merge": (HERE / "build-merge", {}, {}),
    # ... plus halo exchange receive tasks wait for this rank's coalesced sends
    "merge_ws": (HERE / "build-merge", {"parthenon/mesh/coalesced_wait_for_sends": "true"},
                 {}),
    # ... with the MPI progress thread instead
    "merge_pt": (HERE / "build-merge", {}, {"PARTHENON_MPI_PROGRESS_US": "10", **MT}),
    # ... with Cray MPICH's own progress thread instead
    "merge_ap": (HERE / "build-merge", {}, {"MPICH_ASYNC_PROGRESS": "1", **MT}),
}
EXE = "example/advection/advection-example"
STACK_LIB = HERE / ("kt-install/lib/libkp_space_time_stack."
                    + ("dylib" if sys.platform == "darwin" else "so"))



def cube(nx, nb, levels=2):
    return {**{f"parthenon/mesh/nx{d}": nx for d in (1, 2, 3)},
            **{f"parthenon/meshblock/nx{d}": nb for d in (1, 2, 3)},
            "parthenon/mesh/numlevel": levels}


# Problem shapes: name -> input overrides. b8/b16 are kept small on purpose for the Mac.
# Block counts are the peak over a 100-cycle run; larger meshes are extrapolated (root
# blocks scale with volume, refined ones with the sphere's surface).
SHAPES = {
    "b8": {"parthenon/mesh/nx1": 32, "parthenon/mesh/nx2": 32, "parthenon/mesh/nx3": 32,
           "parthenon/meshblock/nx1": 8, "parthenon/meshblock/nx2": 8,
           "parthenon/meshblock/nx3": 8, "parthenon/mesh/numlevel": 2},
    "b16": {"parthenon/mesh/nx1": 32, "parthenon/mesh/nx2": 32, "parthenon/mesh/nx3": 32,
            "parthenon/meshblock/nx1": 16, "parthenon/meshblock/nx2": 16,
            "parthenon/meshblock/nx3": 16, "parthenon/mesh/numlevel": 3},
    # Production-like block size. A 64^3 mesh (2 root blocks per direction) never changes
    # its refinement pattern, hence 128^3. On the Mac only one variable fits in memory
    # (num_vars=4 reaches ~7 GB on 4 ranks), see the mac32 matrix.
    "b32": {"parthenon/mesh/nx1": 128, "parthenon/mesh/nx2": 128, "parthenon/mesh/nx3": 128,
            "parthenon/meshblock/nx1": 32, "parthenon/meshblock/nx2": 32,
            "parthenon/meshblock/nx3": 32, "parthenon/mesh/numlevel": 2},  # 421 blocks
    "b32m": cube(256, 32),  # ~2000 blocks
    "b32l": cube(512, 32),  # ~10000 blocks
    "b64": cube(256, 64),  # ~420 blocks
    "b64m": cube(512, 64),  # ~2000 blocks
}
# On the Mac, Open MPI's btl/sm default eager limit (4 KiB) makes fragment free lists grow
# without bound under many outstanding small messages (see README); 32 KiB keeps memory
# bounded.
MAC_MPI_ARGS = ["--mca", "btl_sm_eager_limit", "32768"]
# Machines: matrix (cases are the product of pairs x coalesced x num_vars x vec_size x
# nlim; pairs are (shape, ranks)), launcher and its extra arguments, default variants
# and timeout.
MACHINES = {
    "mac": dict(pairs=[(s, r) for s in ("b8", "b16") for r in (1, 4)],
                coalesced=[False, True], num_vars=[1, 4], vec_size=[1, 4], nlim=[40],
                launcher="mpirun", mpi_args=MAC_MPI_ARGS, variants=["develop", "pr"],
                timeout=180.0),
    "mac32": dict(pairs=[("b32", 1), ("b32", 4)], coalesced=[False, True], num_vars=[1],
                  vec_size=[1], nlim=[100], launcher="mpirun", mpi_args=MAC_MPI_ARGS,
                  variants=["develop", "pr"], timeout=180.0),
    # 2 x 56-core Sapphire Rapids per node, hyperthreading on. Each rank gets one core
    # (both hyperthreads, -c 2) so the progress thread variants can pin their thread to
    # the sibling. Cases that need more nodes than the allocation has are skipped.
    "rocinante": dict(pairs=[("b32", 1), ("b32", 8), ("b32m", 56), ("b32m", 112),
                             ("b32m", 224), ("b32l", 448), ("b32l", 896), ("b64", 8),
                             ("b64", 56), ("b64m", 448), ("b64m", 896)],
                      coalesced=[False, True], num_vars=[1, 4], vec_size=[1], nlim=[200],
                      launcher="srun", ranks_per_node=112,
                      mpi_args=["-c", "2", "--cpu-bind=cores", "--kill-on-bad-exit=1"],
                      variants=["develop", "pr_rw", "merge", "merge_pt"], timeout=900.0),
}
# Span groups in the REMESH_TIMING line (see bench_remesh_timer.hpp). Group times are sums
# of per-span max-over-ranks.
GROUPS = {"xfer": ("d_", "p_"), "fin": ("f_",)}
TIMING_RX = re.compile(r"^REMESH_TIMING (.*)$", re.M)


def process_tree(root_pid):
    """{pid: rss_kB} for root_pid and all of its descendants."""
    out = subprocess.run(["ps", "-eo", "pid=,ppid=,rss="], capture_output=True,
                         text=True).stdout
    children, rss = {}, {}
    for line in out.splitlines():
        pid, ppid, r = (int(x) for x in line.split())
        children.setdefault(ppid, []).append(pid)
        rss[pid] = r
    tree, stack = {}, [root_pid]
    while stack:
        p = stack.pop()
        if p in rss:
            tree[p] = rss[p]
        stack.extend(children.get(p, []))
    return tree


def kill_tree(root_pid, exe):
    """SIGKILL every process in the tree. Open MPI puts ranks in their own process group,
    so killpg on mpirun alone orphans them; also sweep for anything running exe."""
    for _ in range(3):
        for pid in process_tree(root_pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        time.sleep(0.2)
    subprocess.run(["pkill", "-9", "-f", str(exe)])


def stop_srun(p):
    """srun forwards SIGTERM to the tasks; give the step time to wind down first."""
    p.terminate()
    try:
        p.wait(timeout=30)
    except subprocess.TimeoutExpired:
        p.kill()


def guarded_run(cmd, exe, cwd, env, mem_limit_kb, timeout, launcher="mpirun"):
    """Run cmd, killing its whole process tree on memory or time overrun. With srun
    only the local srun process is visible, so only the timeout applies."""
    with open(cwd / "out.log", "w") as out, open(cwd / "mem.csv", "w") as trace:
        trace.write("t_s,total_mb,max_proc_mb,nproc\n")
        p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=out, stderr=subprocess.STDOUT,
                             start_new_session=True)
        t0, peak, status = time.time(), 0, None
        try:
            while p.poll() is None:
                tree = process_tree(p.pid)
                total = sum(tree.values())
                peak = max(peak, total)
                trace.write(f"{time.time() - t0:.2f},{total / 1024:.1f},"
                            f"{max(tree.values(), default=0) / 1024:.1f},{len(tree)}\n")
                if launcher != "srun" and peak > mem_limit_kb:
                    status = "killed_mem"
                elif time.time() - t0 > timeout:
                    status = "killed_time"
                if status:
                    break
                time.sleep(0.2)
        finally:
            # Also covers Ctrl-C and exceptions in this script
            if p.poll() is None or status:
                if launcher == "srun":
                    stop_srun(p)
                else:
                    kill_tree(p.pid, exe)
                p.wait()
        status = status or ("ok" if p.returncode == 0 else f"exit_{p.returncode}")
    return status, peak / 1024.0, time.time() - t0


def parse_log(text):
    events = []
    for m in TIMING_RX.finditer(text):
        events.append({k: float(v) for k, v in
                       (kv.split("=") for kv in m.group(1).split())})
    zc = re.findall(r"zone-cycles/wallsecond = ([0-9.e+-]+)", text)
    wall = re.findall(r"walltime used = ([0-9.e+-]+)", text)
    return events, (float(zc[-1]) if zc else None), (float(wall[-1]) if wall else None)


def span_names(events):
    skip = {"event", "nb_old", "nb_new", "n_move", "n_ref", "n_deref", "n_xrank", "barrier"}
    return [k for k in events[0] if k not in skip and not k.startswith(("msgs_", "bytes_"))]


def summarize(events):
    """First event separately (includes pool allocation), median over the rest. Message
    and byte counts are summed over all events."""
    row = {"n_events": len(events)}
    if not events:
        return row
    spans = span_names(events)
    for e in events:
        for g, prefixes in GROUPS.items():
            e[g] = sum(v for k, v in e.items() if k in spans and k.startswith(prefixes))
            for kind in ("msgs", "bytes"):
                e[f"{kind}_{g}"] = sum(v for k, v in e.items()
                                       if k.startswith(tuple(f"{kind}_{p}" for p in prefixes)))
    row["first_total"] = events[0]["total"]
    rest = events[1:] or events
    for s in spans + list(GROUPS):
        row[f"med_{s}"] = statistics.median(e[s] for e in rest)
    row["sum_total"] = sum(e["total"] for e in events)
    for k in sorted({k for e in events for k in e if k.startswith(("msgs_", "bytes_"))}):
        row[f"sum_{k}"] = int(sum(e.get(k, 0) for e in events))
    for k in ("n_move", "n_ref", "n_deref"):
        row[f"sum_{k}"] = int(sum(e[k] for e in events))
    row["max_blocks"] = int(max(e["nb_new"] for e in events))
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--machine", default="mac", choices=MACHINES)
    ap.add_argument("--outdir", default=str(HERE / "results"))
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--mem-limit-gb", type=float, default=8.0)
    ap.add_argument("--timeout", type=float, default=None,
                    help="per run, seconds (default per machine)")
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these shapes")
    ap.add_argument("--ranks", nargs="*", type=int, default=None,
                    help="restrict to these rank counts")
    ap.add_argument("--variants", nargs="*", default=None, choices=VARIANTS,
                    help="default per machine; develop is always run first as the "
                    "reference")
    ap.add_argument("--num-vars", nargs="*", type=int, default=None,
                    help="restrict to these Advection/num_vars")
    ap.add_argument("--coalesced", nargs="*", type=int, default=None, choices=[0, 1],
                    help="restrict to these do_coalesced_comms settings")
    ap.add_argument("--nodes", type=int, default=None,
                    help="srun: nodes available (default $SLURM_NNODES)")
    ap.add_argument("--stack", action="store_true", help="enable space-time-stack")
    ap.add_argument("--barrier", action="store_true",
                    help="BENCH_BARRIER=1: barrier after every timer mark")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep-going", action="store_true",
                    help="continue the sweep after a memory kill")
    ap.add_argument("--launcher", default=None, help="mpirun or srun executable")
    args = ap.parse_args()

    m = MACHINES[args.machine]
    srun = m["launcher"] == "srun"
    launcher = args.launcher or m["launcher"]
    timeout = args.timeout or m["timeout"]
    nodes_avail = args.nodes or int(os.environ.get("SLURM_NNODES", "1"))
    pairs = [(s, r) for s, r in m["pairs"]
             if (args.only is None or s in args.only)
             and (args.ranks is None or r in args.ranks)]
    if srun:
        skipped = [p for p in pairs if math.ceil(p[1] / m["ranks_per_node"]) > nodes_avail]
        pairs = [p for p in pairs if p not in skipped]
        if skipped:
            print(f"skipping {skipped}: need more than {nodes_avail} nodes")
    coalesced = [c for c in m["coalesced"]
                 if args.coalesced is None or int(c) in args.coalesced]
    num_vars = [n for n in m["num_vars"] if args.num_vars is None or n in args.num_vars]
    cases = [(s, r, *rest) for (s, r), *rest in
             itertools.product(pairs, coalesced, num_vars, m["vec_size"], m["nlim"],
                               range(args.reps))]
    variants = ["develop"] + [v for v in args.variants or m["variants"] if v != "develop"]
    missing = [v for v in variants if not (VARIANTS[v][0] / EXE).exists()]
    print(f"{len(cases)} cases x {len(variants)} variants ({' '.join(variants)}), "
          + ("" if srun else f"mem limit {args.mem_limit_gb} GB, ")
          + f"timeout {timeout} s")
    if args.dry_run:
        for c in cases:
            print("  ", c)
        if missing:
            print(f"not built: {' '.join(missing)}")
        return
    if missing:
        sys.exit(f"not built: {' '.join(missing)}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    if args.stack:
        env["KOKKOS_TOOLS_LIBS"] = str(STACK_LIB)
    if args.barrier:
        env["BENCH_BARRIER"] = "1"
    rows = []
    for shape, ranks, coal, nv, vs, nlim, rep in cases:
        case = f"{shape}_np{ranks}_coal{int(coal)}_nv{nv}_vs{vs}_n{nlim}_r{rep}"
        hst = {}
        for branch in variants:
            build, extra_overrides, extra_env = VARIANTS[branch]
            rundir = outdir / case / branch
            rundir.mkdir(parents=True, exist_ok=True)
            overrides = dict(SHAPES[shape])
            overrides.update({"parthenon/mesh/do_coalesced_comms": str(coal).lower(),
                              "Advection/num_vars": nv, "Advection/vec_size": vs,
                              "parthenon/time/nlim": nlim})
            overrides.update(extra_overrides)
            exe = build / EXE
            if srun:
                nodes = math.ceil(ranks / m["ranks_per_node"])
                cmd = [launcher, "-N", str(nodes), "-n", str(ranks), *m["mpi_args"]]
            else:
                cmd = [launcher, *m["mpi_args"], "-np", str(ranks)]
            cmd += [str(exe), "-i", str(DECK)]
            cmd += [f"{k}={v}" for k, v in overrides.items()]
            status, peak_mb, elapsed = guarded_run(
                cmd, exe, rundir, {**env, **extra_env}, args.mem_limit_gb * 1024**2, timeout,
                m["launcher"])
            if srun:
                peak_mb = float("nan")  # only srun itself is visible
            events, zc, wall = parse_log((rundir / "out.log").read_text())
            hst_file = rundir / "remesh_bench.out0.hst"
            hst[branch] = hst_file.read_bytes() if hst_file.exists() else None
            row = dict(case=case, branch=branch, shape=shape, ranks=ranks,
                       coalesced=int(coal), num_vars=nv, vec_size=vs, nlim=nlim, rep=rep,
                       barrier=int(args.barrier), machine=args.machine,
                       status=status, peak_rss_mb=round(peak_mb, 1),
                       elapsed_s=round(elapsed, 2), walltime_s=wall,
                       zone_cycles_per_s=zc,
                       hst_identical=int(hst[branch] is not None
                                         and hst[branch] == hst.get("develop")),
                       **summarize(events))
            rows.append(row)
            nan = float("nan")
            print(f"{case:38s} {branch:9s} {status:11s} ev={row['n_events']:3d} "
                  f"total={row.get('med_total', nan):.2e} "
                  f"xfer={row.get('med_xfer', nan):.2e} fin={row.get('med_fin', nan):.2e} "
                  f"msgs={row.get('sum_msgs_xfer', 0):6d} "
                  f"MB={row.get('sum_bytes_xfer', 0) / 2**20:7.1f} "
                  f"peak={peak_mb:6.0f}MB wall={wall} zc/s={zc}", flush=True)
            if status != "ok":
                print(f"  !! {status}: see {rundir / 'out.log'}")
                if status == "killed_mem" and not args.keep_going:
                    sys.exit("stopping sweep after memory kill (use --keep-going)")
                break
        differ = [b for b in hst if hst[b] is None or hst[b] != hst.get("develop")]
        if differ:
            print(f"  !! history differs from develop for {case}: {' '.join(differ)}")

    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(outdir / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {outdir / 'results.csv'}")


if __name__ == "__main__":
    sys.exit(main())
