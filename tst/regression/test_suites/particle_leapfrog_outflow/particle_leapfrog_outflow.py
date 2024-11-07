# ========================================================================================
# Parthenon performance portable AMR framework
# Copyright(C) 2021 The Parthenon collaboration
# Licensed under the 3-clause BSD License, see LICENSE file for details
# ========================================================================================
# (C) (or copyright) 2021. Triad National Security, LLC. All rights reserved.
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
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured

import sys
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

        data = phdf("particles.out0.final.phdf")
        swarm = data.GetSwarm("my_particles")
        inds = np.argsort(swarm["id"])
        final_data = np.vstack((swarm.x, swarm.y, swarm.z, swarm["v"]))
        final_data = final_data.transpose()[inds]

        # see examples/particle_leapfrog/particle_leapfrog.cpp for reference data
        # here, outflow boundaries are used such that some of the particles escape
        # this tests code paths not otherwise present (when blocks have >0 neighbors)
        ref_data = np.array(
            [
                [0.45, 0.2, 0.3, 1.0, 0.0, 0.0],
                [0.4, 0.45, 0.3, 0.0, 1.0, 0.0],
                [-0.1, 0.3, 0.475, 0.0, 0.0, 0.5],
            ]
        )
        if ref_data.shape != final_data.shape:
            print("TEST FAIL: Mismatch between actual and reference data shape.")
            return False
        return (np.abs(final_data - ref_data) <= 1e-10).all()
