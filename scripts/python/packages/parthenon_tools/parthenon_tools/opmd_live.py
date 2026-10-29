#!/usr/bin/env python3
# =========================================================================================
# Parthenon performance portable AMR framework
# Copyright(C) 2026 The Parthenon collaboration
# Licensed under the 3-clause BSD License, see LICENSE file for details
# =========================================================================================
# This file was made in part with generative AI.

"""Live viewer for streamed openPMD outputs (``stream_engine = sst | bp5``).

Each received step is plotted (2D data directly, 3D data as a slice through the middle of
the outermost axis). For outputs with ``on_request = true`` the next step is requested by
creating the ``<problem_id>.<id>.request`` file after the current step has been received,
so that the simulation only writes data that is actually being looked at.
"""

import argparse
import os
import pathlib
import subprocess
import sys
import time

# Keeps requesting until the first step arrives. Required as SST only admits new readers
# at step boundaries. Runs in a separate process because the blocking open holds the GIL.
BOOTSTRAP = "import pathlib, sys, time\nwhile True:\n pathlib.Path(sys.argv[1]).touch()\n time.sleep(0.2)"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stream", help="stream to read, e.g., advection.live.sst")
    ap.add_argument("record", help="mesh record to plot, e.g., advected_lvl0")
    ap.add_argument("--component", help="record component (default: first one)")
    ap.add_argument("--cmap", default="viridis", help="matplotlib colormap")
    ap.add_argument("--vmin", type=float, help="lower limit of the color scale")
    ap.add_argument("--vmax", type=float, help="upper limit of the color scale")
    ap.add_argument(
        "--outdir", help="save frames as png to this dir (instead of a window)"
    )
    ap.add_argument("--max-frames", type=int, help="stop after this number of frames")
    ap.add_argument(
        "--min-interval",
        type=float,
        default=0.0,
        help="minimum time in seconds between requests (limits the simulation overhead)",
    )
    args = ap.parse_args()

    import matplotlib

    if args.outdir:
        matplotlib.use("Agg")
        os.makedirs(args.outdir, exist_ok=True)
    import matplotlib.pyplot as plt
    import numpy as np
    import openpmd_api as io

    request = pathlib.Path(os.path.splitext(args.stream)[0] + ".request")
    boot = subprocess.Popen([sys.executable, "-c", BOOTSTRAP, str(request)])
    try:  # the stream is only created by the simulation on the first request
        while not os.path.exists(args.stream):
            time.sleep(0.1)
        series = io.Series(args.stream, io.Access.read_linear)
    except BaseException:  # e.g., KeyboardInterrupt while waiting
        boot.kill()
        request.unlink(missing_ok=True)
        raise
    fig, ax, im = None, None, None
    t_req = -float("inf")  # time of the last request
    try:
        for nframe, (idx, it) in enumerate(series.snapshots().items()):
            if args.record not in it.meshes:  # e.g., fully refined level with AMR
                print(f"Skipping iteration {idx} as it does not contain {args.record}")
                it.close()
                time.sleep(max(t_req + args.min_interval - time.monotonic(), 0))
                request.touch()
                t_req = time.monotonic()
                continue
            mesh = it.meshes[args.record]
            comps = dict(mesh.items())
            comp = comps[args.component or next(iter(comps))]
            # only load the written chunks (with AMR, finer levels are only partially
            # covered); the remaining (invalid) data is masked
            chunks = [(c.offset, c.extent) for c in comp.available_chunks()]
            chunks = [(o, e, comp.load_chunk(o, e)) for o, e in chunks]
            spacing, offset = list(mesh.grid_spacing), list(mesh.grid_global_offset)
            t = it.time
            series.flush()
            it.close()
            data = np.full(comp.shape, np.nan)
            for o, e, chunk in chunks:
                data[tuple(slice(i, i + n) for i, n in zip(o, e))] = chunk
            data = np.ma.masked_invalid(data)
            if boot is not None:
                boot.kill()
                boot = None
            last = args.max_frames is not None and nframe + 1 >= args.max_frames
            delay = not last and time.monotonic() < t_req + args.min_interval
            if not last and not delay:
                request.touch()  # simulation can produce the next step while we plot
                t_req = time.monotonic()

            while data.ndim > 2:  # slice through the middle of the outermost axis
                data, spacing, offset = (
                    data[data.shape[0] // 2],
                    spacing[1:],
                    offset[1:],
                )
            # data is in C order, i.e., the last axis is x
            extent = [
                offset[-1],
                offset[-1] + spacing[-1] * data.shape[-1],
                offset[0],
                offset[0] + spacing[0] * data.shape[0],
            ]
            if im is None:
                fig, ax = plt.subplots()
                im = ax.imshow(
                    data,
                    origin="lower",
                    extent=extent,
                    cmap=args.cmap,
                    vmin=args.vmin,
                    vmax=args.vmax,
                )
                fig.colorbar(im, ax=ax, label=args.record)
                if not args.outdir:
                    plt.show(block=False)
            else:
                im.set_data(data)
                if args.vmin is None and args.vmax is None:
                    im.autoscale()
            ax.set_title(f"iteration {idx}, time {t:.4g}")
            if args.outdir:
                fig.savefig(os.path.join(args.outdir, f"{args.record}.{idx:05d}.png"))
            else:
                plt.pause(0.01)
                if not plt.fignum_exists(fig.number):
                    break
            if last:
                break
            if delay:  # wait (keeping the window responsive) before requesting
                wait = max(t_req + args.min_interval - time.monotonic(), 1e-3)
                if args.outdir:
                    time.sleep(wait)
                else:
                    plt.pause(wait)
                request.touch()
                t_req = time.monotonic()
    finally:
        if boot is not None:
            boot.kill()
        request.unlink(missing_ok=True)
        series.close()


if __name__ == "__main__":
    main()
