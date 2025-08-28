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


match_cache = {}


def compute_asymmetry(f, varname, filename=None):
    "Computes the asymmetry of var with varname in hdf5 output file object f"

    if filename is not None and (varname, filename) in match_cache.keys():
        matches = match_cache[(varname, filename)]
    else:
        xlocs = f["Locations/x"][:]
        ylocs = f["Locations/y"][:]
        iylocs = -np.flip(ylocs, axis=1)

        x_diff = np.abs(xlocs[:, np.newaxis, :] - xlocs[np.newaxis, :, :])
        y_diff = np.abs(ylocs[:, np.newaxis, :] - iylocs[np.newaxis, :, :])
        x_match = np.all(x_diff <= 1e-10, axis=2)
        y_match = np.all(y_diff <= 1e-10, axis=2)

        combined_match = x_match & y_match
        matches = np.argmax(combined_match, axis=1)

        if filename is not None:
            match_cache[(varname, filename)] = matches

    var = f[varname][:]
    var_diff = np.zeros_like(var)

    # indices for blocks with y > 0 and their partners with y < 0
    mask = np.any(ylocs >= 0, axis=1)
    blk_indices = np.nonzero(mask)[0]
    partner_indices = matches[blk_indices]
    # Used for flipping the sign for vector/tensor quantities... only
    # supports up to rank 2
    ndim_outer = var.shape[1]
    ndim_inner = var.shape[2] if len(var.shape) > 5 else 1

    for d in range(ndim_outer):
        sign1 = -1 if (var.shape[1] == 3) and d == 1 else 1
        for dd in range(ndim_inner):  # maybe trivial
            vlocb = var[blk_indices, d]
            vlocbb = var[partner_indices, d]
            sign2 = -1 if (var.shape[2] == 3) and (ndim_inner > 1) and (dd == 1) else 1
            sign = sign1 * sign2
            if ndim_inner > 1:
                vlocb = vlocb[:, dd]
                vlocbb = vlocbb[:, dd]
            diff_top = vlocb - sign * np.flip(vlocbb, axis=-2)
            diff_bottom = vlocbb - sign * np.flip(vlocb, axis=-2)
            if ndim_inner > 1:
                var_diff[blk_indices, d, dd] = diff_top
                var_diff[partner_indices, d, dd] = diff_bottom
            else:
                var_diff[blk_indices, d] = diff_top
                var_diff[partner_indices, d] = diff_bottom

    # exclude outermost faces for fluxes in trivial directions
    if "bnd_flux" in varname:
        var_diff[:, 0, ..., -1, :, :] = 0
        var_diff[:, 0, ..., :, -1, :] = 0
        var_diff[:, 1, ..., -1, :, :] = 0
        var_diff[:, 1, ..., :, :, -1] = 0
        var_diff[:, 2, ..., :, -1, :] = 0
        var_diff[:, 2, ..., :, :, -1] = 0

    return var_diff


parser = ArgumentParser(
    prog="compute_asymmetry.py",
    description="compute asymmetry in X2 of a field and save it to the output file. "
    + "Assumes mesh is symmetric about 0 in X2. "
    + "Only works for cell- and node-centered data.",
)
parser.add_argument("field", type=str, help="Variable to compute")
parser.add_argument("files", type=str, nargs="+", help="Files to compute")


def main():
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


if __name__ == "__main__":
    main()
