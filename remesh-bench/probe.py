#!/usr/bin/env python3
"""One-off capped probe: probe.py <branch> <np> <cap_gb> <timeout_s> [key=val ...]"""
import pathlib, sys, tempfile
import bench
branch, np_, cap, tmo = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
exe = bench.HERE / f"build-{branch}" / bench.EXE
ov = dict(bench.SHAPES["b8"]); ov["parthenon/time/nlim"] = 40
ov.update(dict(kv.split("=", 1) for kv in sys.argv[5:]))
import os
# Mac default (btl/sm eager limit); PROBE_MPI_ARGS replaces it
extra = os.environ.get("PROBE_MPI_ARGS", " ".join(bench.MAC_MPI_ARGS)).split()
cmd = ["mpirun", *extra, "-np", np_, str(exe), "-i", str(bench.DECK)] + [f"{k}={v}" for k, v in ov.items()]
d = pathlib.Path(tempfile.mkdtemp(prefix="probe-"))
status, peak, el = bench.guarded_run(cmd, exe, d, None, cap * 1024**2, tmo)
ev, zc, wall = bench.parse_log((d / "out.log").read_text())
print(f"{branch} np{np_} {' '.join(sys.argv[5:]):60s} {status:11s} peak={peak:7.0f} MB "
      f"t={el:5.1f}s remeshes={len(ev)} done={wall is not None}  ({d})")
