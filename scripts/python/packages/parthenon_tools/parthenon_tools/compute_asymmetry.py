#!/usr/bin/env python
# ========================================================================================
#  (C) (or copyright) 2025. Triad National Security, LLC. All rights reserved.
#
#  This program was produced under U.S. Government contract 89233218CNA000001 for Los
#  Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
#  for the U.S. Department of Energy/National Nuclear Security Administration. All rights
#  in the program are reserved by Triad National Security, LLC, and the U.S. Department
#  of Energy/National Nuclear Security Administration. The Government is granted for
#  itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
#  license in this material to reproduce, prepare derivative works, distribute copies to
#  the public, perform publicly and display publicly, and to permit others to do so.
# ========================================================================================

import sys
import numpy as np
import h5py
from argparse import ArgumentParser


def compute_asymmetry(f, varname):
    "Computes the asymmetry of var with varname in hdf5 output file object f"
    xlocs = f["Locations/x"][:]
    ylocs = f["Locations/y"][:]
    iylocs = -np.flip(f["Locations/y"][:], axis=1)
    matches = np.zeros((ylocs.shape[0]), dtype=int)
    for b in range(ylocs.shape[0]):
        for bb in range(ylocs.shape[0]):
            if np.all(np.abs(ylocs[b] - iylocs[bb]) <= 1e-10) and np.all(
                np.abs(xlocs[b] - xlocs[bb]) <= 1e-10
            ):
                matches[b] = bb

    var = f[varname][:]
    var_diff = np.zeros_like(var)
    for b in range(ylocs.shape[0]):
        bb = matches[b]
        if np.any(ylocs[b] >= 0):
            if len(var_diff.shape) > 5:
                for d in range(var_diff.shape[1]):
                    sign1 = -1 if (var.shape[1] == 3) and d == 1 else 1
                    for dd in range(var_diff.shape[2]):
                        sign2 = -1 if (var.shape[2] == 3) and dd == 1 else 1
                        sign = sign1 * sign2
                        var_diff[b, d, dd] = var[b, d, dd] - sign * np.flip(var[bb, d, dd], axis=-2)
                        var_diff[bb, d, dd] = var[bb, d, dd] - sign * np.flip(var[b, d, dd], axis=-2)
            else:
                for d in range(var_diff.shape[1]):
                    sign = -1 if (var.shape[1] == 3) and d == 1 else 1
                    var_diff[b, d] = var[b, d] - sign * np.flip(var[bb, d], axis=-2)
                    var_diff[bb, d] = var[bb, d] - sign * np.flip(var[b, d], axis=-2)

    return var_diff


parser = ArgumentParser(
    prog="compute_asymmetry.py",
    description="compute asymmetry in X2 of a field and save it to the output file. "
    + "Assumes mesh is symmetric about 0 in X2. "
    + "Only works for cell- and node-centered data.",
)
parser.add_argument("field", type=str, help="Variable to compute")
parser.add_argument("files", type=str, nargs="+", help="Files to compute")

if __name__ == "__main__":
    args = parser.parse_args()
    for i, fname in enumerate(args.files):
        with h5py.File(fname, "a") as f:
            print(f"Computing asymmetry for {args.field} in {fname}...")
            var_diff = compute_asymmetry(f, args.field)
            savename = f"{args.field}_asymmetry"
            try:
                f.create_dataset(savename, data=var_diff)
            except ValueError:
                f[savename][:] = var_diff
