# ========================================================================================
# Parthenon performance portable AMR framework
# Copyright(C) 2026 The Parthenon collaboration
# Licensed under the 3-clause BSD License, see LICENSE file for details
# ========================================================================================
# (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
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

# This file was made in part with generative AI.

import numpy as np
import sys
import utils.test_case

sys.dont_write_bytecode = True

# 16 blocks of 100 tracers
NUM_TRACERS = 1600


class TestCase(utils.test_case.TestCaseAbs):
    def Prepare(self, parameters, step):
        # One MeshData per rank, then one per block
        if step == 1:
            parameters.driver_cmd_line_args = ["parthenon/job/problem_id=default"]
        else:
            parameters.driver_cmd_line_args = [
                "parthenon/job/problem_id=packed",
                "parthenon/mesh/pack_size=1",
            ]
        return parameters

    def Analyse(self, parameters):
        success = True
        for name in ["default", "packed"]:
            # Columns: time, dt, cycle, nbtotal, num_tracers, total_deposition,
            # num_iterations
            hst = np.genfromtxt(f"{name}.out0.hst")
            time, num_tracers = hst[:, 0], hst[:, 4]
            deposition, num_iterations = hst[:, 5], hst[:, 6]

            # Tracers cross several blocks per step, but are never lost on a periodic
            # mesh
            if not (num_tracers == NUM_TRACERS).all():
                print(f"TEST FAIL: {name}: Tracers were lost or duplicated:")
                print(num_tracers)
                success = False
            # Every tracer is pushed to the end of the step and deposited exactly once, in
            # a block of the MeshData holding it. The deposition is first computed in the
            # first cycle.
            if not (deposition[time > 0] == NUM_TRACERS).all():
                print(f"TEST FAIL: {name}: Tracers were not all deposited once:")
                print(deposition)
                success = False
            # Tracers only drop out of the push when they move to another MeshData, which
            # takes more than one iteration. With one MeshData per rank, that is another
            # rank.
            several = name == "packed" or parameters.num_ranks > 1
            expected = (lambda n: n > 1) if several else (lambda n: n == 1)
            if not all(expected(n) for n in num_iterations[time > 0]):
                print(
                    f"TEST FAIL: {name}: Unexpected iterations on "
                    f"{parameters.num_ranks} ranks:"
                )
                print(num_iterations)
                success = False
        return success
