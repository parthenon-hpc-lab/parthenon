#!/usr/bin/env python

# © 2022-2025. Triad National Security, LLC. All rights reserved.  This
# program was produced under U.S. Government contract
# 89233218CNA000001 for Los Alamos National Laboratory (LANL), which
# is operated by Triad National Security, LLC for the U.S.  Department
# of Energy/National Nuclear Security Administration. All rights in
# the program are reserved by Triad National Security, LLC, and the
# U.S. Department of Energy/National Nuclear Security
# Administration. The Government is granted for itself and others
# acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
# license in this material to reproduce, prepare derivative works,
# distribute copies to the public, perform publicly and display
# publicly, and to permit others to do so.

"""This is a python script for plotting 1d simulations that leverages the coords output so
it always gets the geometry correct and plots in x coordinates.
"""

from argparse import ArgumentParser
import numpy as np
import sys
import os

import matplotlib
import matplotlib.pyplot as plt

# Assumes phdf in global python path
try:
    from parthenon_tools.phdf import phdf # type: ignore
except ModuleNotFoundError:
    from phdf import phdf # type: ignore

from multiprocessing import Pool


def analytic_solution(x: np.ndarray, t: float, D: float | np.ndarray, t0: float) -> np.ndarray:
    """analytic solution for the diffusion of a constant coefficient Gaussian"""
    xc = 0.0
    return np.sqrt(t0 / (t + t0)) * np.exp(-0.25 * (x - xc) ** 2 / (D * (t + t0)))


def plot_dump(
    filename: str,
    varname: str,
    savename: str,
) -> None:
    data = phdf(filename)

    if savename is not None:
        matplotlib.use("Agg")

    q = data.Get(varname, False)
    if q is None:
        print("ERROR: variable not found!")
        raise Exception(f"Variable {varname} not found!")
    if len(q[0,:,0,0]) != 1 or len(q[0,0,:,0]) != 1:
        raise Exception("This script only supports 1D plotting!")
    NB = q.shape[0]

    x = data.x

    t0 = 0.001
    D = 1.0
    t = data.Time
    fig, ax = plt.subplots()
    for i in range(NB):
        label_n = "Numeric" if i == 0 else ""
        xplt = x[i, :]
        ax.plot(xplt, q[i, 0, 0, :], color="#a78598", marker="x", ls=" ", label=label_n)
    x_sol = np.linspace(x.min(), x.max(), 1024)
    ax.plot(
            x_sol, analytic_solution(x_sol, t, D, t0), color="#011939", label="Analytic"
    )

    plt.title(f"Time = {data.Time}")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$\phi$")
    plt.ylim(0.0, 1.0)

    plt.legend(frameon=True)
    plt.savefig(savename, dpi=300, bbox_inches="tight")

    plt.clf()
    plt.cla()
    plt.close()

    return


if __name__ == "__main__":
    parser = ArgumentParser(description="Plot a 1d simulation snapshot.")
    parser.add_argument(
        "-s", "--saveprefix", type=str, default="", help="Prefix for file save names"
    )
    parser.add_argument(
        "-n", "--nprocs", type=int, default=1, help="Number of parallel threads"
    )
    parser.add_argument("--pdf", action="store_true", help="Save as pdf instead of png")
    parser.add_argument("varname", type=str, help="Variable to plot")
    parser.add_argument(
        "files", type=str, nargs="+", help="Files to plot"
    )
    args = parser.parse_args()

    postfix = ".pdf" if args.pdf else ".png"

    def make_frame(pair):
        i, f = pair
        savename = args.saveprefix + str(i).rjust(5, "0") + postfix
        print(savename)
        plot_dump(
            f,
            args.varname,
            savename,
        )
    
    try:
        p = Pool(processes=args.nprocs)
        p.map(make_frame, enumerate(args.files))
    except Exception as e:
        print(f"An error has occured: {e}")
        sys.exit(os.EX_SOFTWARE)
