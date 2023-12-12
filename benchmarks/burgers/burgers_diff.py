#!/usr/bin/env python
# ========================================================================================
# (C) (or copyright) 2023. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001 for Los
# Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
# for the U.S. Department of Energy/National Nuclear Security Administration. All rights
# in the program are reserved by Triad National Security, LLC, and the U.S. Department
# of Energy/National Nuclear Security Administration. The Government is granted for
# itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
# license in this material to reproduce, prepare derivative works, distribute copies to
# the public, perform publicly and display publicly, and to permit others to do so.
# ========================================================================================

import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser(
    prog="burgers_diff.py",
    description="Compute difference between two history solvers parthenon VIBE",
)
parser.add_argument("file1", type=str, help="First file in diff")
parser.add_argument("file2", type=str, help="Second fiel in diff")
parser.add_argument(
    "-t", "--tolerance", type=float, default=1e-8, help="Relative tolerance for diff"
)

def get_rel_diff(d1, d2):
    "Get relative difference between two numpy arrays"
    return 2 * np.abs(d1 - d2) / (d1 + d2 + 1e-20)


if __name__ == "__main__":
    args = parser.parse_args()
    d1 = np.loadtxt(args.file1)
    d2 = np.loadtxt(args.file2)
    diffs = get_rel_diff(d1, d2)
    mask = diffs > args.tolerance
    if np.any(mask):
        print("Diffs found!")
        indices = np.transpose(np.nonzero(mask))
        print("Diff locations (row, column) =", indices)
        print("Diffs =", diffs[mask])
    else:
        print("No diffs found!")
