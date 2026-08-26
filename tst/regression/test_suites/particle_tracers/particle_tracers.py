# ========================================================================================
# Parthenon performance portable AMR framework
# Copyright(C) 2021 The Parthenon collaboration
# Licensed under the 3-clause BSD License, see LICENSE file for details
# ========================================================================================
# (C) (or copyright) 2024. Triad National Security, LLC. All rights reserved.
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

        data = phdf("particle_tracers.out0.final.phdf")
        swarm = data.GetSwarm("tracers")
        inds = np.argsort(swarm.Get("id"))
        final_data = np.vstack((swarm.x, swarm.y, swarm.z))
        final_data = final_data.transpose()[inds]
        final_data[np.abs(final_data) < 1e-12] = 0
        print(final_data)

        # Sourced by sampling in examples/particle_tracers/particle_tracers.cpp. Note that all
        # tracers should be in the +x region.
        ref_data = np.array(
            [
                [0.08444359, -0.3121717, -0.47703363],
                [0.25117872, -0.43392261, 0.17052895],
                [0.1887266, -0.21990098, -0.43307513],
                [0.27823534, -0.05306855, 0.36869215],
                [0.07034217, 0.17493737, -0.16868733],
                [0.27436085, 0.10335561, -0.4434526],
                [0.12410318, 0.44082736, 0.44992147],
                [0.31997097, 0.32557197, -0.09350522],
            ]
        )
        if ref_data.shape != final_data.shape:
            print("TEST FAIL: Mismatch between actual and reference data shape.")
            success = False
        if not (np.abs(final_data - ref_data) <= 1e-8).all():
            print(
                "Disagreement between reference and actual data:", ref_data, final_data
            )
            print(
                f"Important: If the numbers match but not the rows, then this is "
                f"likely related to a mismatch in the autogereated ids, which are implicitly tested here. "
                f"In this case the test here should be updated by ordering particles positions first "
                f"and then compare."
            )
            success = False
        return success
