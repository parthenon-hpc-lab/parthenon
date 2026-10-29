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

NUM_TRACERS = 1000


class TestCase(utils.test_case.TestCaseAbs):
    def Prepare(self, parameters, step):
        if step == 1:
            parameters.driver_cmd_line_args = ["parthenon/job/problem_id=default"]
        elif step == 2:
            # Field boundaries go through coalesced messages, next to the lukewarm ones,
            # with one block per MeshData so that every block boundary is a MeshData one
            parameters.driver_cmd_line_args = [
                "parthenon/job/problem_id=coalesced",
                "parthenon/mesh/do_coalesced_comms=true",
                "parthenon/mesh/pack_size=1",
            ]
        else:
            # The mesh refines and derefines as the advected profile moves, and tracers
            # move to the new MeshData. The deposition is not carried to new blocks, so
            # outputs come before the remesh.
            parameters.driver_cmd_line_args = [
                "parthenon/job/problem_id=amr",
                "parthenon/mesh/refinement=adaptive",
                "parthenon/mesh/numlevel=2",
                "parthenon/refinement0/method=derivative_order_1",
                "parthenon/refinement0/field=field.advected",
                "parthenon/refinement0/refine_tol=0.04",
                "parthenon/refinement0/derefine_tol=0.03",
                "parthenon/time/output_before_amr=true",
            ]
        return parameters

    def Analyse(self, parameters):
        success = True
        for problem_id in ["default", "coalesced", "amr"]:
            # Columns: time, dt, cycle, nbtotal, num_tracers, total_deposition
            hst = np.genfromtxt(problem_id + ".out0.hst")
            time, nbtotal = hst[:, 0], hst[:, 3]
            num_tracers, deposition = hst[:, 4], hst[:, 5]

            # Tracers move between blocks, MeshData, and ranks, but are never lost on a
            # periodic mesh. Each block of the initial mesh rounds its share of
            # NUM_TRACERS, so with AMR the total depends on the initial refinement.
            if problem_id != "amr" and num_tracers[0] != NUM_TRACERS:
                print(f"TEST FAIL ({problem_id}): Expected {NUM_TRACERS} tracers:")
                print(num_tracers)
                success = False
            if not (num_tracers == num_tracers[0]).all():
                print(f"TEST FAIL ({problem_id}): Tracers were lost or duplicated:")
                print(num_tracers)
                success = False
            # Every tracer is deposited exactly once, in a block of the MeshData holding
            # it. The deposition is first computed in the first cycle.
            if not (deposition[time > 0] == num_tracers[time > 0]).all():
                print(f"TEST FAIL ({problem_id}): Tracers were not all deposited once:")
                print(deposition)
                success = False
            if problem_id == "amr" and len(np.unique(nbtotal)) < 2:
                print(f"TEST FAIL ({problem_id}): The mesh never changed:")
                print(nbtotal)
                success = False
        return success
