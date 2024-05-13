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

        data = phdf("particle_tracers.out0.final.phdf")
        swarm = data.GetSwarm("tracers")
        inds = np.argsort(swarm["id"])
        final_data = np.vstack((swarm.x, swarm.y, swarm.z))
        final_data = final_data.transpose()[inds]
        final_data[np.abs(final_data) < 1e-12] = 0

        # see examples/particle_tracers/particle_tracers.cpp for reference data
        ref_data = np.array(
            [
                [0.08365301, -0.47193529, 0.16082123],
                [0.46202008, -0.29930838, -0.23241539],
                [0.07983044, -0.18021412, -0.27073458],
                [0.29650397, -0.19697732, 0.2891521],
                [0.08365301, 0.02806471, 0.16082123],
                [0.46202008, 0.20069162, -0.23241539],
                [0.07983044, 0.31978588, -0.27073458],
                [0.29650397, 0.30302268, 0.2891521],
            ]
        )
        if ref_data.shape != final_data.shape:
            print("TEST FAIL: Mismatch between actual and reference data shape.")
            return False
        return (np.abs(final_data - ref_data) <= 1e-8).all()
