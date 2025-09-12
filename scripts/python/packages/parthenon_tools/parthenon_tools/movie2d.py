#!/usr/bin/env python
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

import re
import os
import logging
import numpy as np

try:
    from phdf import phdf
except ModuleNotFoundError:
    from parthenon_tools.phdf import phdf

from argparse import ArgumentParser
from pathlib import Path
from concurrent.futures import (
    ThreadPoolExecutor,
    ProcessPoolExecutor,
    wait,
    ALL_COMPLETED,
)

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import is_color_like
from matplotlib.colors import LogNorm


def maybe_float(string):
    try:
        return float(string)
    except:
        return string


logging.basicConfig(
    level=logging.CRITICAL, format="%(asctime)s [%(levelname)s]\t%(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

parser = ArgumentParser(
    prog="movie2d", description="Plot snapshots of 2d parthenon output"
)
parser.add_argument(
    "--vector-component",
    dest="vc",
    type=int,
    default=None,
    help="Vector component of field to plot. Mutually exclusive with --tensor-component.",
)
parser.add_argument(
    "--tensor-component",
    dest="tc",
    type=int,
    nargs=2,
    default=None,
    help="Tensor components of field to plot. Mutally exclusive with --vector-component.",
)
parser.add_argument(
    "--colorbar", type=str, default=None, help="Add a colorbar with the specified label"
)
parser.add_argument(
    "--logcolor", action="store_true", help="Colorbar is on a log scale"
)
parser.add_argument("--colormap", type=str, default="plasma", help="Colormap to use")
parser.add_argument(
    "--colorbounds", type=float, nargs=2, default=None, help="Bounds of colorbar"
)
parser.add_argument(
    "--symmetrizecolors", action="store_true", help="Force colorbar to be symmetric"
)
parser.add_argument(
    "--swarm",
    type=str,
    default=None,
    help="Optional particle swarm to overplot figure",
)
parser.add_argument(
    "--swarmcolor",
    type=str,
    default=None,
    help="Optional color of overplotted particle positions. Default is black. You may specify a scalar swarm variable as the color or a matplotlib color string.",
)
parser.add_argument(
    "--particlesize",
    type=maybe_float,
    default=mpl.rcParams["lines.markersize"] ** 2,
    help="Optional size of overplotted particles. Default is standard size chosen by matplotlib. You may specify either a scalar swarm variable or a float.",
)
parser.add_argument(
    "--maxparticlesize",
    dest="pscale",
    type=float,
    default=mpl.rcParams["lines.markersize"] ** 2,
    help="If --particlesize is set by swarm variable, rescales it to scale from 0 to this value. Default is default matplotlib markersize**2.",
)
parser.add_argument(
    "--maxparticles",
    metavar="N",
    type=int,
    default=100,
    help="Limit plot to only N particles at most. Default is 100.",
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
    "--worker-type",
    help="Type of worker to use (default: process)",
    choices=["process", "thread"],
    default="process",
)
parser.add_argument(
    "--output-directory",
    "-d",
    help=f"Output directory to save the images (default: {os.getcwd()})",
    type=Path,
    default=os.getcwd(),
    metavar="DIR",
)
parser.add_argument(
    "--prefix",
    help="Prefix for the file name to save",
    default="",
    metavar="PREFIX",
)
parser.add_argument(
    "--debug-plot",
    help="Make plots with an exploded grid and including ghost zones. (default: false)",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--log-level",
    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    default="INFO",
    help="Log level to set for the logger. (default: INFO)",
)
parser.add_argument(
    "--frame-rate",
    type=int,
    default=24,
    help="Movie framerate (default: 24)",
    metavar="FRAMERATE",
)
parser.add_argument(
    "--movie-format",
    choices=("gif", "mp4"),
    default="mp4",
    help="Movie output format. (default: mp4)",
)
parser.add_argument(
    "--movie-filename",
    help="Basename of the output movie file. (default: output.FORMAT)",
    default="output",
)
parser.add_argument(
    "--render",
    help="Generate the movie from the parsed output files (default: false)",
    default=False,
    action="store_true",
)
parser.add_argument(
    "--xlim",
    type=float,
    help="x bounds. Defaults to whole domain.",
    nargs=2,
    default=None,
)
parser.add_argument("--xlabel", type=str, help="Label for x axis", default=None)
parser.add_argument(
    "--ylim",
    type=float,
    help="y bounds. Defaults to whole domain.",
    nargs=2,
    default=None,
)
parser.add_argument("--ylabel", type=str, help="Label for y axis", default=None)
parser.add_argument("field", type=str, help="field to plot")
parser.add_argument("files", type=str, nargs="+", help="files to plot")


def report_find_fail(key, search_location, available, logger):
    logger.error(f"{key} not found in {search_location}. Further processing stopped.")
    logger.info(f"Available fields: {available}")
    return


def subsample(array, maxlen):
    "Subsample array with fixed stride to have a maximum length of maxlen"
    ratio = len(array) / maxlen
    if ratio >= 1:
        aout = array[:: int(ratio)]
    else:
        aout = array
    return aout[:maxlen]


def rescale(array, amax):
    "Makes all vars in array range from 0 to amax. Destructive operation"
    out = array.astype(float)
    out -= out.min()
    out *= np.array((amax / float(out.max())), dtype=float)
    return out


def plot_dump(
    xf,
    yf,
    q,
    time_title,
    output_file: Path,
    colormap="viridis",
    logcolor=False,
    colorbar=None,
    colorbounds=None,
    symmetrize_colors=False,
    with_mesh=False,
    block_ids=[],
    xi=None,
    yi=None,
    xe=None,
    ye=None,
    components=[0, 0],
    swarmx=None,
    swarmy=None,
    swarmcolor=None,
    particlesize=None,
    xlim=None,
    xlabel=None,
    ylim=None,
    ylabel=None,
):
    if xe is None:
        xe = xf
    if ye is None:
        ye = yf

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
    # move to 2d
    q = q[..., 0, :, :]

    fig, p = plt.subplots()
    p.set_aspect(1)
    if time_title is not None:
        p.set_title(f"t = {time_title}")

    if logcolor:
        q = np.abs(q)

    qm = np.ma.masked_where(np.isnan(q), q)
    qmin = qm.min()
    qmax = qm.max()

    # Removes a visual quirk where matplotlib plots things at totally
    # random colors if a field is exactly zero
    if qmin == qmax == 0:
        qmin = -1e-14
        qmax = 1e-14

    if colorbounds is not None:
        qmin, qmax = colorbounds

    if symmetrize_colors:
        qmin = min(qmin, -qmax)
        qmax = max(qmax, -qmin)

    if logcolor:
        if qmin <= 0 or qmax <= 0 or qmax < qmin:
            raise ValueError("Log plot must have strictly positive bounds")

    n_blocks = q.shape[0]
    for i in range(n_blocks):
        # Plot the actual data, should work if parthenon/output*/ghost_zones = true/false
        # but obviously no ghost data will be shown if ghost_zones = false
        if logcolor:
            pm = p.pcolormesh(
                xf[i, :],
                yf[i, :],
                q[i, :, :],
                cmap=colormap,
                norm=LogNorm(vmin=qmin, vmax=qmax),
            )
        else:
            pm = p.pcolormesh(
                xf[i, :], yf[i, :], q[i, :, :], cmap=colormap, vmin=qmin, vmax=qmax
            )

        # Print the block gid in the center of the block
        if len(block_ids) > 0:
            p.text(
                0.5 * (xf[i, 0] + xf[i, -1]),
                0.5 * (yf[i, 0] + yf[i, -1]),
                str(block_ids[i]),
                fontsize=8,
                color="w",
                ha="center",
                va="center",
            )

        # Plot the interior and exterior boundaries of the block
        if with_mesh:
            rect = mpatches.Rectangle(
                (xe[i, 0], ye[i, 0]),
                (xe[i, -1] - xe[i, 0]),
                (ye[i, -1] - ye[i, 0]),
                linewidth=0.225,
                edgecolor="k",
                facecolor="none",
                linestyle="solid",
            )
            p.add_patch(rect)
            if (xi is not None) and (yi is not None):
                rect = mpatches.Rectangle(
                    (xi[i, 0], yi[i, 0]),
                    (xi[i, -1] - xi[i, 0]),
                    (yi[i, -1] - yi[i, 0]),
                    linewidth=0.225,
                    edgecolor="k",
                    facecolor="none",
                    linestyle="dashed",
                )
                p.add_patch(rect)

    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])

    if swarmx is not None and swarmy is not None:
        p.scatter(swarmx, swarmy, s=particlesize, c=swarmcolor)
    if colorbar is not None:
        plt.colorbar(pm, label=colorbar, fraction=0.02, pad=0.04, ax=p)
    if xlabel is not None:
        p.set_xlabel(xlabel)
    if ylabel is not None:
        p.set_ylabel(ylabel)

    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig=fig)
    logger.debug(f"Saved {time_title}s time-step to {output_file}")


def main():
    ERROR_FLAG = False
    args = parser.parse_args()
    logger.setLevel(args.log_level)

    if args.tc and args.vc:
        raise ValueError(
            "Only one of --tensor-component and --vector-component should be set."
        )

    if args.workers > 1:
        logger.warning(
            "Matplotlib is not multi-thread friendly. Read this for more details https://matplotlib.org/stable/users/faq/howto_faq.html#work-with-threads"
        )
        logger.warning(
            "Try decreasing threads if you encounter any undefined behaviour"
        )
    # Create output director if does't exists
    args.output_directory.mkdir(0o755, True, True)
    logger.info(f"Total files to process: {len(args.files)}")

    components = []
    if args.tc is not None:
        components = args.tc
    if args.vc is not None:
        components = [args.vc]
    do_swarm = args.swarm is not None

    _x = ProcessPoolExecutor if args.worker_type == "process" else ThreadPoolExecutor
    with _x(max_workers=args.workers) as pool:
        futures = []
        for frame_id, file_name in enumerate(args.files):
            data = phdf(file_name)

            if args.field not in data.Variables:
                report_find_fail(args.field, file_name, data.Variables, logger)
                ERROR_FLAG = True
                break
            q = data.Get(args.field, False, not args.debug_plot)
            if do_swarm:
                if args.swarm not in data.Variables:
                    report_find_fail(args.swarm, file_name, data.Variables, logger)
                    ERROR_FLAG = True
                    break
                swarm = data.GetSwarm(args.swarm)
                swarmx = subsample(swarm.x, args.maxparticles)
                swarmy = subsample(swarm.y, args.maxparticles)
                if args.swarmcolor is not None:
                    if not is_color_like(args.swarmcolor):
                        if args.swarmcolor not in swarm.variables:
                            report_find_fail(
                                args.swarmcolor, args.swarm, swarm.variables, logger
                            )
                            ERROR_FLAG = True
                            break
                        swarmcolor = swarm[args.swarmcolor]
                        if len(swarmcolor.shape) > 1:
                            logger.error(
                                f"{args.swarmcolor} has nonzero tensor rank, which is not supported."
                            )
                            ERROR_FLAG = True
                            break
                        swarmcolor = subsample(swarmcolor, args.maxparticles)
                    else:
                        swarmcolor = args.swarmcolor
                else:
                    swarmcolor = "k"
                if not isinstance(args.particlesize, float):
                    if args.particlesize not in swarm.variables:
                        report_find_fail(
                            args.particlesize, args.swarm, swarm.variables, logger
                        )
                        ERROR_FLAG = True
                        break
                    particlesize = swarm[args.particlesize]
                    if len(particlesize.shape) > 1:
                        logger.error(
                            f"{args.particlesize} has nonzero tensor rank, which is not supported."
                        )
                        ERROR_FLAG = True
                        break
                    particlesize = subsample(particlesize, args.maxparticles)
                    particlesize = rescale(particlesize, args.pscale)
                else:
                    particlesize = args.particlesize
            else:
                swarm = None
                swarmx = None
                swarmy = None
                swarmcolor = None
                particlesize = None

            name = "{}{:04d}.png".format(args.prefix, frame_id).strip()
            output_file = args.output_directory / name

            # NOTE: After doing 5 test on different precision, keeping 2 looks more promising
            current_time = format(data.Time, ".2e")
            if args.debug_plot:
                futures.append(
                    pool.submit(
                        plot_dump,
                        data.xg,
                        data.yg,
                        q,
                        current_time,
                        output_file,
                        args.colormap,
                        args.logcolor,
                        args.colorbar,
                        args.colorbounds,
                        args.symmetrizecolors,
                        True,
                        data.gid,
                        data.xig,
                        data.yig,
                        data.xeg,
                        data.yeg,
                        components,
                        swarmx,
                        swarmy,
                        swarmcolor,
                        particlesize,
                        args.xlim,
                        args.xlabel,
                        args.ylim,
                        args.ylabel,
                    )
                )
            else:
                futures.append(
                    pool.submit(
                        plot_dump,
                        data.xng,
                        data.yng,
                        q,
                        current_time,
                        output_file,
                        args.colormap,
                        args.logcolor,
                        args.colorbar,
                        args.colorbounds,
                        args.symmetrizecolors,
                        True,
                        components=components,
                        swarmx=swarmx,
                        swarmy=swarmy,
                        swarmcolor=swarmcolor,
                        particlesize=particlesize,
                        xlim=args.xlim,
                        xlabel=args.xlabel,
                        ylim=args.ylim,
                        ylabel=args.ylabel,
                    )
                )
        wait(futures, return_when=ALL_COMPLETED)
        for future in futures:
            exception = future.exception()
            if exception:
                raise exception

    if not ERROR_FLAG:
        logger.info("All frames produced.")

        if args.render:
            logger.info(f"Generating {args.movie_format} movie")
            input_pattern = args.output_directory / "*.png"
            ffmpeg_cmd = f"ffmpeg -hide_banner -loglevel error -y -framerate {args.frame_rate} -pattern_type glob -i '{input_pattern}' "
            output_filename = (
                args.output_directory / f"{args.movie_filename}.{args.movie_format}"
            )

            if args.movie_format == "gif":
                ffmpeg_cmd += '-vf "scale=800:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" '
            elif args.movie_format == "mp4":
                ffmpeg_cmd += "-c:v libx264 -pix_fmt yuv420p "

            ffmpeg_cmd += f"{output_filename}"
            logger.debug(f"Executing ffmpeg command: {ffmpeg_cmd}")
            os.system(ffmpeg_cmd)
            logger.info(f"Movie saved to {output_filename}")


if __name__ == "__main__":
    main()
