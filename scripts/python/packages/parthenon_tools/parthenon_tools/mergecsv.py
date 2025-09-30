#!/usr/bin/env python3

# =========================================================================================
# Parthenon performance portable AMR framework
# Copyright(C) 2020-2024 The Parthenon collaboration
# Licensed under the 3-clause BSD License, see LICENSE file for details
# =========================================================================================
# (C) (or copyright) 2025. Triad National Security, LLC. All rights reserved.
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

# This script was generated with the help of Generative AI. The first
# draft was built with the help of ChatGPT5.

import sys
import csv
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Merge CSV files, keeping header from the first file, deduplicating, and sorting rows."
    )
    parser.add_argument("files", nargs="+", help="CSV files to merge")
    args = parser.parse_args()

    all_rows = set()
    header = None

    for i, filename in enumerate(args.files):
        with open(filename, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            file_header = next(reader)

            if i == 0:
                header = file_header
            else:
                if file_header != header:
                    print(
                        f"Warning: Header in {filename} does not match the first file",
                        file=sys.stderr,
                    )

            for row in reader:
                all_rows.add(tuple(row))  # tuples are hashable

    sorted_rows = sorted(all_rows)

    writer = csv.writer(sys.stdout)
    writer.writerow(header)
    writer.writerows(sorted_rows)


if __name__ == "__main__":
    main()
