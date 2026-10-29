#!/usr/bin/env python3
# =========================================================================================
# Parthenon performance portable AMR framework
# Copyright(C) 2026 The Parthenon collaboration
# Licensed under the 3-clause BSD License, see LICENSE file for details
# =========================================================================================
# This file was made in part with generative AI.

"""Live viewer for streamed openPMD outputs (``stream_engine = sst | bp5``).

All refinement levels of a variable are combined on a uniform float32 grid of a given
level. Each received step is plotted (2D data directly, 3D data as a slice through the
middle of the outermost axis or as a volume rendering). For outputs with
``on_request = true`` the next step is requested by creating the
``<problem_id>.<id>.request`` file after the current step has been received, so that the
simulation only writes data that is actually being looked at.
"""

import argparse
import contextlib
import os
import pathlib
import subprocess
import sys
import time

# Keeps requesting until the first step arrives. Required as SST only admits new readers
# at step boundaries. Runs in a separate process because the blocking open holds the GIL.
BOOTSTRAP = "import pathlib, sys, time\nwhile True:\n pathlib.Path(sys.argv[1]).touch()\n time.sleep(0.2)"


def scale(idx, s):
    """Scales the indices idx by 2^s."""
    r = 2 ** abs(s)
    if s < 0 and any(i % r for i in idx):
        raise ValueError(f"Cannot coarsen {idx} by {r}, use a finer target level.")
    return [i * r if s >= 0 else i // r for i in idx]


def composite(series, it, var, level=0, component=None):
    """Combines all (cell-centered) records var_lvl<N> of iteration it on a uniform
    float32 grid of the given refinement level (negative levels are coarser than the root
    grid). Finer data is averaged, coarser data is repeated.
    Returns the data (C order, i.e., the last axis is x), cell spacing and offset."""
    import numpy as np

    chunks, geom = [], None
    for name, mesh in it.meshes.items():
        base, _, lvl = name.rpartition("_lvl")
        if base != var or not lvl.isdigit():
            continue
        comps = dict(mesh.items())
        comp = comps[component or next(iter(comps))]
        s = level - int(lvl)  # log2 of the change in resolution
        if geom is None:
            geom = (scale(comp.shape, s), [d * 2.0**-s for d in mesh.grid_spacing])
            offset = list(mesh.grid_global_offset)
        for c in comp.available_chunks():
            chunks.append((s, c.offset, comp.load_chunk(c.offset, c.extent)))
    if geom is None:
        raise KeyError(f"No records {var}_lvl<N> in iteration.")
    series.flush()
    data = np.full(geom[0], np.nan, dtype=np.float32)
    for s, o, chunk in chunks:
        o = scale(o, s)
        if s < 0:  # average 2^-s cells in each direction
            r = 2**-s
            shape = [n for e in scale(chunk.shape, s) for n in (e, r)]
            axes = tuple(range(1, 2 * chunk.ndim, 2))
            chunk = chunk.reshape(shape).mean(axis=axes, dtype=np.float32)
        for ax in range(chunk.ndim if s > 0 else 0):
            chunk = chunk.repeat(2**s, axis=ax)
        data[tuple(slice(i, i + n) for i, n in zip(o, chunk.shape))] = chunk
    return data, geom[1], offset


def frames(
    stream,
    var,
    level=0,
    component=None,
    min_interval=0.0,
    max_frames=None,
    pause=time.sleep,
):
    """Yields (iteration, time, data, spacing, offset) of each received step, see
    composite(). The next step is requested once a step has been received (but not more
    often than every min_interval seconds, waiting with pause(seconds) after the step has
    been processed), so that the simulation can produce it while the step is processed.
    """
    import openpmd_api as io

    request = pathlib.Path(os.path.splitext(stream)[0] + ".request")
    boot = subprocess.Popen([sys.executable, "-c", BOOTSTRAP, str(request)])
    series = None
    try:
        # the stream is only created by the simulation on the first request
        while not os.path.exists(stream):
            time.sleep(0.1)
        series = io.Series(stream, io.Access.read_linear)
        t_req = -float("inf")  # time of the last request
        for nframe, (idx, it) in enumerate(series.snapshots().items()):
            frame = (idx, it.time, *composite(series, it, var, level, component))
            it.close()
            boot.kill()
            if max_frames is not None and nframe + 1 >= max_frames:
                yield frame
                return
            delay = time.monotonic() < t_req + min_interval
            if not delay:
                request.touch()
                t_req = time.monotonic()
            yield frame
            if delay:
                pause(max(t_req + min_interval - time.monotonic(), 1e-3))
                request.touch()
                t_req = time.monotonic()
    finally:
        boot.kill()
        request.unlink(missing_ok=True)
        if series is not None:
            series.close()


class SlicePlot:
    """Plots 2D data or a slice through the middle of the outermost axis of 3D data."""

    def __init__(self, args):
        import matplotlib

        if args.outdir:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        self.plt, self.args, self.im = plt, args, None
        self.pause = time.sleep if args.outdir else plt.pause

    def __call__(self, idx, t, data, spacing, offset):
        import numpy as np

        args, plt = self.args, self.plt
        while data.ndim > 2:
            data, spacing, offset = data[data.shape[0] // 2], spacing[1:], offset[1:]
        data = np.ma.masked_invalid(data)
        if self.im is None:
            self.fig, self.ax = plt.subplots()
            extent = [offset[-1], offset[-1] + spacing[-1] * data.shape[-1]]
            extent += [offset[0], offset[0] + spacing[0] * data.shape[0]]
            self.im = self.ax.imshow(
                data,
                origin="lower",
                extent=extent,
                cmap=args.cmap,
                vmin=args.vmin,
                vmax=args.vmax,
            )
            self.fig.colorbar(self.im, ax=self.ax, label=args.var)
            if not args.outdir:
                plt.show(block=False)
        else:
            self.im.set_data(data)
            if args.vmin is None and args.vmax is None:
                self.im.autoscale()
        self.ax.set_title(f"iteration {idx}, time {t:.4g}")
        if args.outdir:
            self.fig.savefig(os.path.join(args.outdir, f"{args.var}.{idx:05d}.png"))
            return True
        plt.pause(0.01)
        return plt.fignum_exists(self.fig.number)


class VolumePlot:
    """Volume rendering of 3D data with PyVista (off-screen if frames are saved).
    The volume is only set up for the first frame (which also sets the color scale
    unless given) and its data is updated in place for later frames."""

    def __init__(self, args):
        import pyvista as pv

        self.pv, self.args, self.grid = pv, args, None
        self.pl = pv.Plotter(off_screen=bool(args.outdir))

    def pause(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:  # keep the window responsive
            if self.grid is not None and not self.args.outdir:
                self.pl.update(stime=10)
            else:
                time.sleep(0.01)

    def __call__(self, idx, t, data, spacing, offset):
        import numpy as np

        args, pl = self.args, self.pl
        if data.ndim != 3:
            raise ValueError("Volume rendering requires 3D data.")
        first = self.grid is None
        if first:
            # data at cell centers, reversed to (x, y, z) order of VTK
            spacing, offset = spacing[::-1], offset[::-1]
            grid = self.pv.ImageData(
                dimensions=data.shape[::-1],
                spacing=spacing,
                origin=[o + 0.5 * d for o, d in zip(offset, spacing)],
            )
            grid.point_data[args.var] = data.ravel()
            clim = [
                np.nanmin(data) if args.vmin is None else args.vmin,
                np.nanmax(data) if args.vmax is None else args.vmax,
            ]
            vol = pl.add_volume(
                grid, scalars=args.var, cmap=args.cmap, clim=clim, opacity=args.opacity
            )
            # the dataset actually used by the mapper (in case it has been copied)
            self.grid = self.pv.wrap(vol.mapper.GetDataSetInput())
            pl.add_mesh(grid.outline(), color="gray")  # for orientation
            pl.show_axes()
        else:
            self.grid.point_data[args.var][:] = data.ravel()
            self.grid.GetPointData().GetArray(args.var).Modified()
        # render=False as add_text would render (incl. the data upload) again otherwise
        pl.add_text(f"iteration {idx}, time {t:.4g}", name="title", render=False)
        if args.outdir:
            pl.render()  # screenshot() only renders before the first frame
            pl.screenshot(os.path.join(args.outdir, f"{args.var}.{idx:05d}.png"))
        elif first:
            pl.show(interactive_update=True)
        else:
            pl.update()
        return pl.render_window is not None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stream", help="stream to read, e.g., advection.live.sst")
    ap.add_argument("var", help="variable to plot, e.g., advected")
    ap.add_argument(
        "--level",
        type=int,
        default=0,
        help="refinement level of the plotted (uniform) grid, negative values coarsen "
        "below the root grid (default: 0)",
    )
    ap.add_argument("--component", help="record component (default: first one)")
    ap.add_argument("--volume", action="store_true", help="volume rendering (PyVista)")
    ap.add_argument(
        "--opacity", default="sigmoid", help="PyVista opacity transfer function"
    )
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
    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)

    plot = VolumePlot(args) if args.volume else SlicePlot(args)
    steps = frames(
        args.stream,
        args.var,
        args.level,
        args.component,
        args.min_interval,
        args.max_frames,
        plot.pause,
    )
    with contextlib.closing(steps):
        for frame in steps:
            if not plot(*frame):
                break


if __name__ == "__main__":
    main()
