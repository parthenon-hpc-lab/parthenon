# =========================================================================================
# (C) (or copyright) 2020-2021. Triad National Security, LLC. All rights reserved.
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

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def addPath():
    """add the vis/python directory to the pythonpath variable"""
    myPath = os.path.realpath(os.path.dirname(__file__))
    # sys.path.insert(0,myPath+'/../vis/python')
    # sys.path.insert(0,myPath+'/vis/python')


def read(filename, nGhost=0):
    """Read the parthenon hdf file"""
    from phdf import phdf

    f = phdf(filename)
    return f


def plot_dump(xf, yf, q, name, with_mesh=False):
    fig = plt.figure()
    p = fig.add_subplot(111, aspect=1)
    qmin = q.min()
    qmax = q.max()
    NumBlocks = q.shape[0]
    for i in range(NumBlocks):
        p.pcolormesh(xf[i, :], yf[i, :], q[i, 0, :, :], vmin=qmin, vmax=qmax)
        if with_mesh:
            rect = mpatches.Rectangle(
                (xf[i, 0], yf[i, 0]),
                (xf[i, -1] - xf[i, 0]),
                (yf[i, -1] - yf[i, 0]),
                linewidth=0.225,
                edgecolor="k",
                facecolor="none",
            )
            p.add_patch(rect)
    plt.savefig(name, dpi=300)
    plt.close()


if __name__ == "__main__":
    addPath()
    field = sys.argv[1]
    files = sys.argv[2:]
    dump_id = 0
    for f in files:
        data = read(f)
        print(data)
        xf = data.xf
        yf = data.yf
        q = data.Get(field, False)
        name = str(dump_id).rjust(4, "0") + ".png"
        plot_dump(xf, yf, q, name, True)
        dump_id += 1
