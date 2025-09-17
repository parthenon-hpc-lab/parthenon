#!/usr/bin/env python
# ========================================================================================
# Parthenon performance portable AMR framework
# Copyright(C) 2020-2025 The Parthenon collaboration
# Licensed under the 3-clause BSD License, see LICENSE file for details
# =========================================================================================
# (C) (or copyright) 2020-2025. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001 for Los
# Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
# for the U.S. Department of Energy/National Nuclear Security Administration. All rights
# in the program are reserved by Triad National Security, LLC, and the U.S. Department
# of Energy/National Nuclear Security Administration. The Government is granted for
# itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
# license in this material to reproduce, prepare derivative works, distribute copies to
# the public, perform publicly and display publicly, and to permit others to do so.
# =========================================================================================

from __future__ import print_function

import numpy as np
from scipy import interpolate
from functools import partial

try:
    from phdf import phdf
except ModuleNotFoundError:
    from parthenon_tools.phdf import phdf

from argparse import ArgumentParser
from multiprocessing import Pool
import matplotlib.pyplot as plt


parser = ArgumentParser(prog="contour1d", description="Make a spacetime contour plot")
parser.add_argument(
    "--components",
    dest="tc",
    type=int,
    nargs="+",
    default=None,
    help="Components of field to plot.",
)
parser.add_argument(
    "--colorbar", type=str, default=None, help="Add a colorbar with the specified label"
)
parser.add_argument(
    "--logcolor", action="store_true", help="Colorbar is on a log scale"
)
parser.add_argument("--colormap", type=str, default="plasma", help="Colormap to use")
parser.add_argument(
    "--colorpoints",
    type=float,
    default=[],
    nargs="+",
    help="Points used on colormap for contours",
)
parser.add_argument(
    "--workers",
    "-w",
    help="Number of parallel workers to use (default: 10)",
    type=int,
    metavar="COUNT",
    default=10,
)
parser.add_argument(
    "--savename",
    help="Name to save file as",
    default="plot.png",
)
parser.add_argument(
    "--xlim",
    type=float,
    help="x bounds. Defaults to whole domain.",
    nargs=2,
    default=None,
)
parser.add_argument("--xlabel", type=str, help="Label for x axis", default="Time")
parser.add_argument(
    "--ylim",
    type=float,
    help="y bounds. Defaults to whole domain.",
    nargs=2,
    default=None,
)
parser.add_argument("--ylabel", type=str, help="Label for y axis", default="radius")
parser.add_argument(
    "-o", "--spline-order", type=int, default=0, help="Order of spline interpolation"
)
parser.add_argument("field", type=str, help="field to plot")
parser.add_argument("--overlay", type=str, default=None, help="field to overlay")
parser.add_argument(
    "--overlay-color",
    default="lightgreen",
    type=str,
    help="Color for overlaid contours",
)
parser.add_argument(
    "--overlay-points",
    type=float,
    default=None,
    nargs="+",
    help="Points of field to overlay",
)
parser.add_argument(
    "--logoverlay", action="store_true", help="Overlay is on a log scale"
)
parser.add_argument(
    "--overlay-components", default=None, nargs="+", help="Components of overlay field"
)
parser.add_argument("files", type=str, nargs="+", help="files to plot")
parser.add_argument(
    "--xscale", type=float, default=1, help="Value to rescale x-axis by"
)
parser.add_argument(
    "--yscale", type=float, default=1, help="Value to rescale y-axis by"
)
parser.add_argument("-g", "--grid", action="store_true", help="Overlay grid")


def get_tensor_components(q, components):
    # get tensor components
    ntensors = len(q.shape[1:-3])
    if components:
        if len(components) != ntensors:
            raise ValueError(
                "Tensor rank not the same as number of specified components: {}, {}, {}".format(
                    len(components), ntensors, q.shape
                )
            )
        # The first index of q is block index. Here we walk through
        # the tensor components, slowest-first and, iteratively, fix
        # q's slowest moving non-block index to the fixed tensor
        # component. Then we move to the next index.
        for c in components:
            if c > (q.shape[1] - 1):
                raise ValueError(
                    "Component {} out of bounds. Shape = {}".format(c, q.shape)
                )
            q = q[:, c]
    # move to 1d
    q = q[..., 0, 0, :]
    return q


def flatten(xblock, q):
    # sort x by first value
    sort_idx = np.argsort(xblock[:, 0])
    xsorted = xblock[sort_idx]
    qsorted = q[sort_idx]
    return xsorted.flatten(), qsorted.flatten()


def get_slice(field, components, file_name):
    data = phdf(file_name)
    q = get_tensor_components(data.Get(field, False), components)
    x, q = flatten(data.x, q)
    return data.Time, x, q


def get_all_slices(files, field, components, workers):
    p = Pool(workers)
    results = p.map(partial(get_slice, field, components), files)
    ts = []
    xs = []
    qs = []
    for t, x, q in results:
        ts.append(t)
        xs.append(x)
        qs.append(q)
    return np.array(ts), xs, qs


def generate_covering_grid(xs):
    dxs = (x[1:] - x[:-1] for x in xs)
    xmin = min(x.min() for x in xs)
    xmax = max(x.max() for x in xs)
    dxmin = min(dx.min() for dx in dxs)
    covering_grid = np.arange(xmin, xmax + dxmin, dxmin)
    return covering_grid


def generate_spacetime_mesh(xs, qs, covering_grid, spline_order):
    fill_data = np.zeros((len(qs), len(covering_grid)))
    for i, q in enumerate(qs):
        xc = xs[i]
        spl = interpolate.make_interp_spline(xc, q, k=spline_order)
        fill_data[i] = spl(covering_grid)
    return fill_data


def main():
    args = parser.parse_args()

    components = []
    if args.tc is not None:
        components = args.tc
    overlay_components = []
    if args.overlay_components is not None:
        overlay_components = args.overlay_components

    data = phdf(args.files[0])
    print(data)

    ts, xs, qs = get_all_slices(args.files, args.field, components, args.workers)
    covering_grid = generate_covering_grid(xs)
    fill_data = generate_spacetime_mesh(xs, qs, covering_grid, args.spline_order)

    fig = plt.figure(figsize=(9, 5))
    fig.subplots_adjust(left=0.10, top=0.96, bottom=0.12, right=0.86)

    ax1 = fig.add_subplot(111)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    q = fill_data
    if args.logcolor:
        q = np.log10(q)

    if args.colorpoints:
        if len(args.colorpoints) == 1:
            colorpoints = np.linspace(q.min(), q.max(), int(args.colorpoints[0]))
        else:
            colorpoints = args.colorpoints
    else:
        colorpoints = np.linspace(q.min(), q.max(), 10)

    cplt = plt.contourf(
        args.xscale * ts,
        args.yscale * covering_grid,
        q.transpose(),
        colorpoints,
        extend="both",
        cmap=args.colormap,
    )

    if args.overlay is not None:
        _, _, os = get_all_slices(
            args.files, args.overlay, overlay_components, args.workers
        )
        overlay_data = generate_spacetime_mesh(xs, os, covering_grid, args.spline_order)
        os = overlay_data
        if args.logoverlay:
            os = np.log10(os)
        if args.overlay_points:
            if len(args.overlay_points) == 1:
                overlaypoints = np.linspace(
                    os.min(), os.max(), int(args.overlay_points[0])
                )
            else:
                overlaypoints = args.overlay_points
        else:
            overlaypoints = np.linspace(os.min(), os.max(), 4)

        plt.contour(
            args.xscale * ts,
            args.yscale * covering_grid,
            os.transpose(),
            overlaypoints,
            colors=[args.overlay_color],
            linestyles=["--"],
            linewidths=[1.5],
        )

    if args.grid:
        plt.grid()
    plt.xlabel(args.xlabel, fontsize=20)
    plt.ylabel(args.ylabel, fontsize=20)
    if args.xlim is not None:
        plt.xlim(args.xlim[0], args.xlim[1])
    if args.ylim is not None:
        plt.ylim(args.ylim[0], args.ylim[1])
    if args.colorbar is not None:
        pos = ax1.get_position()
        cbaxes = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.02, pos.y1 - pos.y0])
        cb = plt.colorbar(cplt, cax=cbaxes)
        cb.set_ticks(args.colorpoints)
        cb.ax.tick_params(labelsize=14)
        cb.set_label(args.colorbar, labelpad=22, fontsize=20, rotation=270)

    plt.savefig(args.savename, bbox_inches="tight")


if __name__ == "__main__":
    main()
