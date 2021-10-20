# ========================================================================================
# Parthenon performance portable AMR framework
# Copyright(C) 2020-2021 The Parthenon collaboration
# Licensed under the 3-clause BSD License, see LICENSE file for details
# ========================================================================================
# (C) (or copyright) 2020-2021. Triad National Security, LLC. All rights reserved.
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
import h5py
import math
import numpy as np
import sys
import os
import utils.test_case


""" To prevent littering up imported folders with .pyc files or __pycache_ folder"""
sys.dont_write_bytecode = True


class TestCase(utils.test_case.TestCaseAbs):
    def Prepare(self, parameters, step):

        # enable coverage testing on pass where restart
        # files are both read and written
        parameters.coverage_status = "both"

        # run baseline (to the very end)
        if step == 1:
            parameters.driver_cmd_line_args = ["parthenon/job/problem_id=gold"]
        # restart from an early snapshot and run for two seconds
        elif step == 2:
            parameters.driver_cmd_line_args = [
                "-r",
                "gold.out0.00001.rhdf",
                "parthenon/job/problem_id=silver",
                "-t",
                "00:00:02",
            ]
        # now restart from the walltime based output
        else:
            parameters.driver_cmd_line_args = [
                "-r",
                "silver.out0.final.rhdf",
            ]

        return parameters

    def Analyse(self, parameters):
        success = True
        # spotcheck one variable
        goldFile = "gold.out0.00005.rhdf"
        silverFile = "silver.out0.00005.rhdf"
        varName = "advected"
        with h5py.File(goldFile, "r") as gold, h5py.File(silverFile, "r") as silver:
            goldData = np.zeros(gold[varName].shape, dtype=np.float64)
            gold[varName].read_direct(goldData)
            silverData = np.zeros(silver[varName].shape, dtype=np.float64)
            silver[varName].read_direct(silverData)

        maxdiff = np.abs(goldData - silverData).max()
        print("Variable: %s, diff=%g, N=%d" % (varName, maxdiff, len(goldData)))
        if maxdiff != 0.0:
            print("ERROR: Found difference between gold and silver output.")
            success = False

        found_line = False
        for line in parameters.stdouts[1].decode("utf-8").split("\n"):
            if "Terminating on wall-time limit" in line:
                found_line = True
        if not found_line:
            print("ERROR: wall-time limit based termination not triggered.")
            success = False

        return success
