# =========================================================================================
# (C) (or copyright) 2022. Triad National Security, LLC. All rights reserved.
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

from argparse import ArgumentParser
import numpy as np
import h5py
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

parser = ArgumentParser("Plot x3 slice of Parthenon mesh")
parser.add_argument('file', type=str,
                    help='rhdf or phdf file to use for mesh')
parser.add_argument('--slice', type=float, default=None,
                    help='Slice through X3 to plot. Default is middle.')
parser.add_argument('--save', type=str, default='mesh.png',
                    help='File to save figure as. Defaults to mesh.png')
args = parser.parse_args()

with h5py.File(args.file,'r') as f:
    domain = f['Info'].attrs['RootGridDomain'].reshape(3,3)
    NB = f['Info'].attrs['NumMeshBlocks']
    if args.slice is not None:
        zslice = args.slice
    else:
        zslice = 0.5(domain[2,0] + domain[2,1])
    fig,ax = plt.subplots()
    for b in range(NB):
        if f['Locations/z'][b,0] <= zslice <= f['Locations/z'][b,-1]:
            xb = (f['Locations/x'][b,0], f['Locations/x'][b,-1])
            yb = (f['Locations/y'][b,0], f['Locations/y'][b,-1])
            rect = mpatches.Rectangle((xb[0],yb[0]),
                                      xb[1]-xb[0],
                                      yb[1]-yb[0],
                                      linewidth=0.225,
                                      alpha=1,
                                      edgecolor='k',
                                      facecolor='none',
                                      linestyle='solid')
            ax.add_patch(rect)
    plt.xlim(domain[0,0],domain[0,1])
    plt.ylim(domain[1,0],domain[1,1])
    plt.savefig(args.save,bbox_inches='tight')

