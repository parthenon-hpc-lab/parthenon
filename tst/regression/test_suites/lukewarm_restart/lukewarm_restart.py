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

AMR = [
    "parthenon/mesh/refinement=adaptive",
    "parthenon/mesh/numlevel=2",
    "parthenon/refinement0/method=derivative_order_1",
    "parthenon/refinement0/field=field.advected",
    "parthenon/refinement0/refine_tol=0.04",
    "parthenon/refinement0/derefine_tol=0.03",
]


class TestCase(utils.test_case.TestCaseAbs):
    def Prepare(self, parameters, step):
        parameters.coverage_status = "both"
        # Each restart runs on MeshData of a different size than the run it restarts
        if step == 1:
            parameters.driver_cmd_line_args = ["parthenon/job/problem_id=gold"]
        elif step == 2:
            parameters.driver_cmd_line_args = [
                "-r",
                "gold.out1.00002.rhdf",
                "parthenon/job/problem_id=silver",
                "parthenon/mesh/pack_size=2",
            ]
        elif step == 3:
            parameters.driver_cmd_line_args = [
                "parthenon/job/problem_id=amr_gold"
            ] + AMR
        else:
            parameters.driver_cmd_line_args = [
                "-r",
                "amr_gold.out1.00002.rhdf",
                "parthenon/job/problem_id=amr_silver",
                "parthenon/mesh/pack_size=2",
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
            print("TEST FAIL: " + msg)
            success = False

        for prefix in ["", "amr_"]:
            for name in ["00003", "final"]:
                for kind in ["out0", "out1"]:
                    files = [
                        f"{prefix}{run}.{kind}.{name}" for run in ["gold", "silver"]
                    ]
                    files = [
                        f + (".phdf" if kind == "out0" else ".rhdf") for f in files
                    ]
                    gold, silver = phdf(files[0]), phdf(files[1])
                    if gold.Time != silver.Time:
                        Fail(f"{files} are at times {gold.Time} and {silver.Time}")
                    tracers = [data.GetLukewarm("tracers") for data in [gold, silver]]
                    ids = [t.id for t in tracers]
                    order = [np.argsort(i) for i in ids]
                    if (
                        len(ids[0]) != len(ids[1])
                        or not (ids[0][order[0]] == ids[1][order[1]]).all()
                    ):
                        Fail(f"{files} hold different tracers")
                        continue
                    # Restarting changes nothing about any tracer
                    for x in ["x", "y", "z"]:
                        a, b = [getattr(t, x)[o] for t, o in zip(tracers, order)]
                        if not (a == b).all():
                            Fail(f"{files} differ in {x} by {np.abs(a - b).max()}")

                    # Each block's tracers are inside it after the restart
                    t = tracers[1]
                    counts, offsets = t.counts, t.offsets
                    if counts.sum() != len(t.id):
                        Fail(f"{files[1]} counts do not add up")
                    for b in range(silver.NumBlocks):
                        xmin, xmax, ymin, ymax, _, _ = silver.BlockBounds[b]
                        s = slice(offsets[b], offsets[b] + counts[b])
                        if not (
                            (xmin <= t.x[s]).all()
                            and (t.x[s] <= xmax).all()
                            and (ymin <= t.y[s]).all()
                            and (t.y[s] <= ymax).all()
                        ):
                            Fail(f"{files[1]}: tracers of block {b} are outside it")
                            break
                    gold.close()
                    silver.close()
        return success
