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


def plot_dump(xf, yf, q, name, with_mesh=False, block_ids=[], xi = None, yi = None, xe = None, ye = None): 
    
    if xe is None: xe = xf 
    if ye is None: ye = yf

    fig = plt.figure()
    p = fig.add_subplot(111, aspect=1)
    qm = np.ma.masked_where(np.isnan(q), q) 
    qmin = qm.min()
    qmax = qm.max()
    NumBlocks = q.shape[0]
    for i in range(NumBlocks):
        # Plot the actual data, should work if parthenon/output*/ghost_zones = true or false
        # but obviously no ghost data will be shown if ghost_zones = false 
        p.pcolormesh(xf[i, :], yf[i, :], q[i, 0, :, :], vmin=qmin, vmax=qmax)
        
        # Print the block gid in the center of the block 
        if (len(block_ids) > 0):
          p.text(0.5*(xf[i, 0] + xf[i, -1]), 0.5*(yf[i, 0] + yf[i, -1]), 
             str(block_ids[i]), fontsize=8, color="w", ha='center', va='center')
        
        # Plot the interior and exterior boundaries of the block 
        if with_mesh:
            rect = mpatches.Rectangle(
                (xe[i, 0], ye[i, 0]),
                (xe[i, -1] - xe[i, 0]),
                (ye[i, -1] - ye[i, 0]),
                linewidth=0.225,
                edgecolor="k",
                facecolor="none",
                linestyle='solid'
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
                   linestyle="dashed"
               )
               p.add_patch(rect)

    plt.savefig(name, dpi=300)
    plt.close()


if __name__ == "__main__":
    addPath()
    field = sys.argv[1]
    files = sys.argv[2:]
    dump_id = 0
    debug_plot = False
    for f in files:
        data = read(f)
        print(data)
        q = data.Get(field, False, not debug_plot)
        name = str(dump_id).rjust(4, "0") + ".png"
        if debug_plot:
          plot_dump(data.xg, data.yg, q, name, True, data.gid, data.xig, data.yig, data.xeg, data.yeg)
        else:
          plot_dump(data.xng, data.yng, q, name, True)
        dump_id += 1
