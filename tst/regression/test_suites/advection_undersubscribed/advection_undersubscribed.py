# ========================================================================================
# Parthenon performance portable AMR framework
# Copyright(C) 2020-2025 The Parthenon collaboration
# Licensed under the 3-clause BSD License, see LICENSE file for details
# ========================================================================================
# (C) (or copyright) 2020-2025. Triad National Security, LLC. All rights reserved.
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
import sys
import utils.test_case

# To prevent littering up imported folders with .pyc files or __pycache_ folder
sys.dont_write_bytecode = True


class TestCase(utils.test_case.TestCaseAbs):
    def Prepare(self, parameters, step):
        parameters.coverage_status = "both"
        return parameters

    def Analyse(self, parameters):
        try:
            import h5py
        except ModuleNotFoundError:
            print("Couldn't find module h5py.")
            return False

        vol_sum = 0
        vol_sum_true = 0.07080078125
        try:
            with h5py.File("undersubscribed.out0.final.phdf", "r") as f:
                dx = (f["Locations/x"][:, 1:] - f["Locations/x"][:, :-1])[:, 0]
                dy = (f["Locations/y"][:, 1:] - f["Locations/y"][:, :-1])[:, 0]
                vol = dx * dy
                vol_sum = (vols[:, None, None] * f["advected"][:, 0, 0, ...]).sum()
        except:
            print("Couldn't open dump file or read all fields")
            return False

        if np.abs(vol_sum - vol_sum_true) > 1e-10:
            print(
                "Sum of advected field not correct. Measured = {:14e}, True = {:14e}".format(
                    vol_sum, vol_sum_true
                )
            )
            return False
        return True
