# ========================================================================================
# Parthenon performance portable AMR framework
# Copyright(C) 2020 The Parthenon collaboration
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
import math
import numpy as np
import sys
import os
import utils.test_case

# To prevent littering up imported folders with .pyc files or __pycache_ folder
sys.dont_write_bytecode = True


class TestCase(utils.test_case.TestCaseAbs):
    def Prepare(self, parameters, step):
        """
        Setup simulation parameters.
        """

        # TEST: 2D AMR
        if step == 1:
            # do nothing, keep defaults
            return parameters
        # TEST: 3D AMR
        elif step == 2:
            parameters.driver_cmd_line_args = [
                "parthenon/job/problem_id=advection_3d",  # change name for new outputs
                "parthenon/mesh/numlevel=2",  # reduce AMR depth for smaller sim
                "parthenon/mesh/nx1=32",
                "parthenon/meshblock/nx1=8",
                "parthenon/mesh/nx2=32",
                "parthenon/meshblock/nx2=8",
                "parthenon/mesh/nx3=32",
                "parthenon/meshblock/nx3=8",
                "parthenon/time/integrator=rk1",
                "Advection/cfl=0.3",
            ]
        # Same as step 1 but shortened for calculating coverage
        elif step == 3:
            parameters.coverage_status = "only-coverage"
            parameters.driver_cmd_line_args = [
                "parthenon/time/tlim=0.01",
            ]
        # Same as step 2 but shortened for calculating coverage
        elif step == 4:
            parameters.coverage_status = "only-coverage"
            parameters.driver_cmd_line_args = [
                "parthenon/job/problem_id=advection_3d",  # change name for new outputs
                "parthenon/mesh/numlevel=2",  # reduce AMR depth for smaller sim
                "parthenon/mesh/nx1=32",
                "parthenon/meshblock/nx1=8",
                "parthenon/mesh/nx2=32",
                "parthenon/meshblock/nx2=8",
                "parthenon/mesh/nx3=32",
                "parthenon/meshblock/nx3=8",
                "parthenon/time/integrator=rk1",
                "Advection/cfl=0.3",
                "parthenon/time/tlim=0.01",
            ]
        return parameters

    def Analyse(self, parameters):
        """
        Analyze the output and determine if the test passes.
        """
        analyze_status = True
        print(os.getcwd())

        sys.path.insert(
            1,
            parameters.parthenon_path
            + "/scripts/python/packages/parthenon_tools/parthenon_tools",
        )

        try:
            import phdf_diff
        except ModuleNotFoundError:
            print("Couldn't find module to compare Parthenon hdf5 files.")
            return False

        # TODO(pgrete) make sure this also works/doesn't fail for the user
        ret_2d = phdf_diff.compare(
            [
                "advection_2d.out0.final.phdf",
                parameters.parthenon_path
                + "/tst/regression/gold_standard/advection_2d.out0.final.phdf",
            ],
            one=True,
        )
        ret_3d = phdf_diff.compare(
            [
                "advection_3d.out0.final.phdf",
                parameters.parthenon_path
                + "/tst/regression/gold_standard/advection_3d.out0.final.phdf",
            ],
            one=True,
        )

        if ret_2d != 0 or ret_3d != 0:
            analyze_status = False

        hst_2d = np.genfromtxt("advection_2d.hst")
        hst_3d = np.genfromtxt("advection_3d.hst")
        ref_results = [
            ["time", 1.0, 1.0],
            ["dt", 1.75781e-03, 3.12500e-03],
            ["total", 7.06177e-02, 1.39160e-02],
            ["max", 9.43685e-01, 4.80914e-01],
            [
                "min",
                # 1.67180e-10 if parameters.sparse_disabled else 1.67171e-10,
                1.67180e-10,
                1.45889e-07,
            ],
        ]
        # check results in last row (at the final time of the sim)
        for i, val in enumerate(ref_results):
            if hst_2d[-1:, i] != val[1]:
                print(
                    "Wrong",
                    val[0],
                    "in hst output of 2D problem (",
                    hst_2d[-1:, i],
                    ", ",
                    val[1],
                    ")",
                )
                analyze_status = False
            if hst_3d[-1:, i] != val[2]:
                print(
                    "Wrong",
                    val[0],
                    "in hst output of 3D problem (",
                    hst_3d[-1:, i],
                    ", ",
                    val[2],
                    ")",
                )
                analyze_status = False

        return analyze_status
