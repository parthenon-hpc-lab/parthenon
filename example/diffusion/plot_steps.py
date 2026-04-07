#========================================================================================
# (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001 for Los
# Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
# for the U.S. Department of Energy/National Nuclear Security Administration. All rights
# in the program are reserved by Triad National Security, LLC, and the U.S. Department
# of Energy/National Nuclear Security Administration. The Government is granted for
# itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
# license in this material to reproduce, prepare derivative works, distribute copies to
# the public, perform publicly and display publicly, and to permit others to do so.
#========================================================================================

#!/usr/bin/env python3
"""
[This code was generated with the help of generative AI]
Plot zone-cycles/wsec_step and v-cycles versus step for one or more Parthenon log files.

Usage:
    python plot_steps.py run1.log run2.log
    python plot_steps.py *.out --output perf.png

Plot style:
    - Left y-axis:  zone-cycles/wsec_step   (solid)
    - Right y-axis: v-cycles                (dashed)
    - One color per run
    - Compact legend:
        * one legend for run colors
        * one legend for line styles
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams["axes.prop_cycle"] = plt.cycler(color=[
    "#0072B2", "#E69F00", "#009E73", "#CC79A7",
    "#56B4E9", "#D55E00", "#F0E442", "#000000"
])

CYCLE_RE = re.compile(
    r"""
    \bcycle=(?P<cycle>\d+)
    .*?
    \bzone-cycles/wsec_step=(?P<zc>[0-9.eE+\-]+)
    .*?
    \bv-cycles=(?P<vc>\d+)
    """,
    re.VERBOSE,
)


@dataclass
class RunData:
    label: str
    cycles: list[int]
    zc_per_sec: list[float]
    vcycles: list[int]


def parse_log(path: pathlib.Path, include_zero: bool = False) -> RunData:
    cycles: list[int] = []
    zc_per_sec: list[float] = []
    vcycles: list[int] = []

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = CYCLE_RE.search(line)
            if not m:
                continue

            cycle = int(m.group("cycle"))
            if cycle == 0 and not include_zero:
                continue

            cycles.append(cycle)
            zc_per_sec.append(float(m.group("zc")))
            vcycles.append(int(m.group("vc")))

    if not cycles:
        raise ValueError(f"No matching cycle lines found in {path}")

    return RunData(
        label=path.stem,
        cycles=cycles,
        zc_per_sec=zc_per_sec,
        vcycles=vcycles,
    )


def make_plot(runs: list[RunData], output: str | None = None, title: str | None = None) -> None:
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    run_handles: list[Line2D] = []
    run_labels: list[str] = []

    for i, run in enumerate(runs):
        color = colors[i % len(colors)]

        ax1.semilogy(
            run.cycles,
            run.zc_per_sec,
            color=color,
            linewidth=2.0,
            linestyle="-",
        )
        ax2.plot(
            run.cycles,
            run.vcycles,
            color=color,
            linewidth=1.8,
            linestyle="--",
        )

        run_handles.append(Line2D([0], [0], color=color, lw=2.5))
        run_labels.append(run.label)

    ax1.set_xlabel("Step")
    ax1.set_ylabel("zone-cycles / s")
    ax2.set_ylabel("v-cycles")

    if title:
        ax1.set_title(title)
    else:
        ax1.set_title("Performance vs. step")

    ax1.grid(True, which="major", alpha=0.3)

    # Legend 1: colors = runs
    legend_runs = ax1.legend(
        run_handles,
        run_labels,
        title="Runs",
        loc="upper left",
        fontsize=9,
        title_fontsize=10,
    )
    ax1.add_artist(legend_runs)

    # Legend 2: line style meaning
    style_handles = [
        Line2D([0], [0], color="black", lw=2.0, linestyle="-"),
        Line2D([0], [0], color="black", lw=1.8, linestyle="--"),
    ]
    style_labels = [
        "zone-cycles / wsec_step",
        "v-cycles",
    ]
    ax1.legend(
        style_handles,
        style_labels,
        title="Line style",
        loc="upper right",
        fontsize=9,
        title_fontsize=10,
    )

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=200, bbox_inches="tight")
        print(f"Wrote {output}")
    else:
        plt.show()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", help="Log files to read")
    parser.add_argument("--output", "-o", help="Write figure to this file instead of showing it")
    parser.add_argument("--title", help="Custom plot title")
    parser.add_argument(
        "--include-zero",
        action="store_true",
        help="Include cycle=0 in the plot",
    )
    args = parser.parse_args()

    runs: list[RunData] = []
    for log in args.logs:
        path = pathlib.Path(log)
        if not path.exists():
            print(f"File not found: {path}", file=sys.stderr)
            return 1
        try:
            runs.append(parse_log(path, include_zero=args.include_zero))
        except Exception as e:
            print(f"Failed to parse {path}: {e}", file=sys.stderr)
            return 1

    make_plot(runs, output=args.output, title=args.title)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())