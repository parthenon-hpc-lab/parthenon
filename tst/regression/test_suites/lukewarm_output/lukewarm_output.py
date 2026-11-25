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

import glob
import os
import numpy as np
import sys
import utils.test_case

sys.dont_write_bytecode = True

NUM_TRACERS = 1000


class TestCase(utils.test_case.TestCaseAbs):
    def Prepare(self, parameters, step):
        if step == 1:
            parameters.driver_cmd_line_args = ["parthenon/job/problem_id=default"]
        else:
            # Blocks of different sizes hold different numbers of tracers
            parameters.driver_cmd_line_args = [
                "parthenon/job/problem_id=amr",
                "parthenon/mesh/refinement=adaptive",
                "parthenon/mesh/numlevel=2",
                "parthenon/refinement0/method=derivative_order_1",
                "parthenon/refinement0/field=field.advected",
                "parthenon/refinement0/refine_tol=0.04",
                "parthenon/refinement0/derefine_tol=0.03",
            ]
        return parameters

    def Analyse(self, parameters):
        sys.path.insert(
            1,
            parameters.parthenon_path
            + "/scripts/python/packages/parthenon_tools/parthenon_tools",
        )
        from phdf import phdf

        success = True

        def Fail(msg):
            nonlocal success
            print("TEST FAIL (" + filename + "): " + msg)
            success = False

        for problem_id in ["default", "amr"]:
            filenames = sorted(glob.glob(problem_id + ".out0.*.phdf"))
            if len(filenames) < 2:
                print(f"TEST FAIL ({problem_id}): Too few outputs: {filenames}")
                success = False
            for filename in filenames:
                data = phdf(filename)
                tracers = data.GetLukewarm("tracers")
                counts, offsets = tracers.counts, tracers.offsets
                x, y, z, ids = tracers.x, tracers.y, tracers.z, tracers.id

                # Tracers are never lost on a periodic mesh. With AMR, each block of the
                # initial mesh rounds its share of NUM_TRACERS.
                if problem_id != "amr" and len(ids) != NUM_TRACERS:
                    Fail(f"Expected {NUM_TRACERS} tracers, found {len(ids)}")
                if len(np.unique(ids)) != len(ids):
                    Fail("Tracer ids are not unique")
                if len(counts) != data.NumBlocks or counts.sum() != len(ids):
                    Fail(f"Counts {counts} do not add up to {len(ids)} tracers")
                if not (offsets == np.cumsum(counts) - counts).all():
                    Fail(f"Offsets {offsets} do not follow counts {counts}")

                # Each block's tracers are inside it
                for b in range(data.NumBlocks):
                    xmin, xmax, ymin, ymax, zmin, zmax = data.BlockBounds[b]
                    s = slice(offsets[b], offsets[b] + counts[b])
                    if not (
                        (xmin <= x[s]).all()
                        and (x[s] <= xmax).all()
                        and (ymin <= y[s]).all()
                        and (y[s] <= ymax).all()
                    ):
                        Fail(f"Tracers listed under block {b} are outside it")
                        break

                # Positions for XDMF are the same positions, interleaved
                positions = tracers.Get("lukewarm_positions")
                if (
                    positions is None
                    or not (positions == np.stack((x, y, z), axis=1)).all()
                ):
                    Fail("lukewarm_positions does not match the tracer positions")
                xdmf = filename + ".lukewarm.xdmf"
                if not os.path.exists(xdmf):
                    Fail("No " + xdmf)
                elif (
                    ":/tracers/LukewarmVars/lukewarm_positions" not in open(xdmf).read()
                ):
                    Fail(xdmf + " does not refer to the tracer positions")
                # There are no swarms, so no swarm XDMF file
                if os.path.exists(filename + ".swarm.xdmf"):
                    Fail("Unexpected " + filename + ".swarm.xdmf")
                data.close()
        return success
