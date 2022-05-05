# ========================================================================================
# Parthenon performance portable AMR framework
# Copyright(C) 2020 The Parthenon collaboration
# Licensed under the 3-clause BSD License, see LICENSE file for details
# ========================================================================================
# (C) (or copyright) 2020-2022. Triad National Security, LLC. All rights reserved.
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

# Modules
import math
import numpy as np
import matplotlib

matplotlib.use("agg")
import matplotlib.pylab as plt
import sys
import os
import utils.test_case

# To prevent littering up imported folders with .pyc files or __pycache_ folder
sys.dont_write_bytecode = True

mb_sizes = [256, 128, 64, 32]  # meshblock sizes


class TestCase(utils.test_case.TestCaseAbs):
    def Prepare(self, parameters, step):

        # splits integer n into the three largest factors
        def get_split(n):
            i = 2
            factors = []
            while i * i <= n:
                if n % i != 0:
                    i += 1
                else:
                    n //= i
                    factors.append(i)
            # always append remainder, could also be 1
            factors.append(n)

            # fill to 3 dims
            while len(factors) < 3:
                factors.append(1)
            # split to 3 dims
            while len(factors) > 3:
                factors = sorted(
                    factors[:-2] + [factors[-2] * factors[-1]], reverse=True
                )
            return sorted(factors, reverse=True)

        num_proc_x, num_proc_y, num_proc_z = get_split(parameters.num_ranks)

        parameters.driver_cmd_line_args = [
            "parthenon/mesh/nx1=%d" % (num_proc_x * 256),
            "parthenon/meshblock/nx1=%d" % mb_sizes[step - 1],
            "parthenon/mesh/nx2=%d" % (num_proc_y * 256),
            "parthenon/meshblock/nx2=%d" % mb_sizes[step - 1],
            "parthenon/mesh/nx3=%d" % (num_proc_z * 256),
            "parthenon/meshblock/nx3=%d" % mb_sizes[step - 1],
            "parthenon/sparse/enable_sparse=false"
            if parameters.sparse_disabled
            else "",
        ]

        return parameters

    def Analyse(self, parameters):

        perfs = []
        for output in parameters.stdouts:
            for line in output.decode("utf-8").split("\n"):
                print(line)
                if "zone-cycles/wallsecond" in line:
                    perfs.append(float(line.split(" ")[2]))

        perfs = np.array(perfs)

        # Save performance metrics to file
        with open("performance_metrics.txt", "w") as writer:
            writer.write("zone-cycles/s | Normalize Overhead | Meshblock Size\n")
            for ind, mb_size in enumerate(mb_sizes):
                writer.write(
                    "%.2e %.2e %d\n" % (perfs[ind], perfs[0] / perfs[ind], mb_size)
                )

        # Plot results
        fig, p = plt.subplots(2, 1, figsize=(4, 8), sharex=True)

        p[0].loglog(mb_sizes, perfs, label="$256^3$ Mesh")
        p[1].loglog(mb_sizes, perfs[0] / perfs)

        for i in range(2):
            p[i].grid()
        p[0].legend()
        p[0].set_ylabel("zone-cycles/s")
        p[1].set_ylabel("normalized overhead")
        p[1].set_xlabel("Meshblock size")
        fig.savefig(
            os.path.join(parameters.output_path, "performance.png"), bbox_inches="tight"
        )

        return True
