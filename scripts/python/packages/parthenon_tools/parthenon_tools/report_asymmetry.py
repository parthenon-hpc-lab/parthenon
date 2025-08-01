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

import os

import h5py
from compute_asymmetry import compute_asymmetry

parser = ArgumentParser(
    prog="report_asymmetry.py",
    description="Report asymmetry in X2 of all fields and save it to the output file. "
    + "Assumes mesh is symmetric about 0 in X2. "
    + "Only works for cell- and node-centered data.",
)
parser.add_argument("files", type=str, nargs="+", help="Files to compute")

dset_type = h5py._hl.dataset.Dataset
asym_fields = {}
if __name__ == "__main__":
    args = parser.parse_args()
    for fname in args.files:
        with h5py.File(fname, "r") as f:
            print(f"Computing asymmetry in {fname} for vars...")
            for k, v in f.items():
                if (type(v) == dset_type):
                    if len(v.shape) > 2:
                        print(f"\t...{k}:")
                        try:
                            var_diff = compute_asymmetry(f, k)
                            absdiff = np.max(np.abs(var_diff))
                            absval = np.max(np.abs(f[k]))
                            frac_diff = absdiff / (absval + 1e-100)
                            if frac_diff > 1e-12:
                                asym_fields[k] = (absdfiff, absval, frac_diff)
                        except ValueError:
                            print("\t\tcorrupted!")
    print("The following fields had non-trivial asymmetry:")
    for k, (absdiff,absval,frac_diff) in asym_fields:
        print(
            "{} : {:14e} / {:14e} : {:14e}".format(
                k, np.max(np.abs(var_diff)), np.max(np.abs(f[k])), frac_diff))
