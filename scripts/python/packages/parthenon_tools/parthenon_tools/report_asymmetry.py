#!/usr/bin/env/python
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
from argparse import ArgumentParser

import os, re

import h5py

try:
    from compute_asymmetry import compute_asymmetry
except ModuleNotFoundError:
    from parthenon_tools.compute_asymmetry import compute_asymmetry

parser = ArgumentParser(
    prog="report_asymmetry.py",
    description="Report asymmetry in X2 of all fields and save it to the output file. "
    + "Assumes mesh is symmetric about 0 in X2. "
    + "Only works for cell- and node-centered data.",
)
parser.add_argument("files", type=str, nargs="+", help="Files to compute")
parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
parser.add_argument(
    "-e", "--exclude", type=str, nargs="*", default=[], help="Variables to exclude"
)
parser.add_argument(
    "-q", "--quiet", action="store_true", help="Print only a maximum asymmetry"
)

dset_type = h5py._hl.dataset.Dataset


def main():
    args = parser.parse_args()
    for fname in args.files:
        asym_fields = {}
        t = 0
        with h5py.File(fname, "r") as f:
            try:
                t = f["Info"].attrs["Time"]
            except:
                continue
            if args.verbose:
                print(f"Computing asymmetry in {fname} for vars...")
            for k, v in f.items():
                if (type(v) == dset_type) and not any(
                    re.search(pattern, k) for pattern in args.exclude
                ):
                    if len(v.shape) > 2:
                        if args.verbose:
                            print(f"\t...{k}")
                        try:
                            var_diff = compute_asymmetry(f, k, fname)
                            absdiff = np.max(np.abs(var_diff))
                            absval = np.max(np.abs(f[k]))
                            frac_diff = absdiff / (absval + 1e-100)
                            if frac_diff > 1e-12:
                                asym_fields[k] = (absdiff, absval, frac_diff)
                        except ValueError:
                            if args.verbose:
                                print("\t\tcorrupted!")
        if args.quiet:
            max_asym = 0
            for k, vals in asym_fields.items():
                absdiff, absval, frac_diff = vals
                max_asym = max(max_asym, frac_diff)
            print("{:<50} {:<50}".format(t, max_asym))
        else:
            print("The following fields had non-trivial asymmetry in {}:".format(fname))
            for k, vals in asym_fields.items():
                absdiff, absval, frac_diff = vals
                print(
                    "{:<50} {:14e} / {:14e} =\t{:14e}".format(
                        k, absdiff, absval, frac_diff
                    )
                )


if __name__ == "__main__":
    main()
