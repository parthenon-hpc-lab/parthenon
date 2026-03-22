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

# This file was made in part with generative AI

# Modules
import math
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured

import sys
import os
import utils.test_case

# To prevent littering up imported folders with .pyc files or __pycache_ folder
sys.dont_write_bytecode = True


class TestCase(utils.test_case.TestCaseAbs):
    def Prepare(self, parameters, step):

        return parameters

    def Analyse(self, parameters):
        sys.path.insert(
            1,
            parameters.parthenon_path
            + "/scripts/python/packages/parthenon_tools/parthenon_tools",
        )
        from phdf import phdf

        success = True

        data = phdf("particle_tracers_amr_source_sink.out0.final.phdf")
        swarm = data.GetSwarm("tracers")
        cohort = swarm.Get("cohort")
        if len(swarm.x) == 0:
            print("TEST FAIL: AMR source/sink swarm is empty.")
            success = False
        if len(cohort) != len(swarm.x):
            print("TEST FAIL: AMR source/sink cohort field length mismatch.")
            success = False
        if not np.any(cohort < 0):
            print("TEST FAIL: missing seeded particle cohort population.")
            success = False
        if not np.any(cohort >= 0):
            print("TEST FAIL: missing sourced particle cohort population.")
            success = False
        return success
