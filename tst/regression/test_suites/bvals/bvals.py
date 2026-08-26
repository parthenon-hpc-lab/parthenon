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
import sys

import numpy as np
import utils.test_case

""" To prevent littering up imported folders with .pyc files or __pycache_ folder"""
sys.dont_write_bytecode = True


class TestCase(utils.test_case.TestCaseAbs):
    def Prepare(self, parameters, step):

        # Step 1 reflecting BC
        if step == 1:
            parameters.driver_cmd_line_args = [
                "parthenon/mesh/ix1_bc=reflecting",
                "parthenon/mesh/ox1_bc=reflecting",
                "parthenon/mesh/ix2_bc=reflecting",
                "parthenon/mesh/ox2_bc=reflecting",
                "parthenon/mesh/ix3_bc=reflecting",
                "parthenon/mesh/ox3_bc=reflecting",
                "parthenon/output0/id=reflecting",
            ]
        # Step 2: periodic BC
        if step == 2:
            parameters.driver_cmd_line_args = [
                "parthenon/mesh/ix1_bc=periodic",
                "parthenon/mesh/ox1_bc=periodic",
                "parthenon/mesh/ix2_bc=periodic",
                "parthenon/mesh/ox2_bc=periodic",
                "parthenon/mesh/ix3_bc=periodic",
                "parthenon/mesh/ox3_bc=periodic",
                "parthenon/output0/id=periodic",
            ]
        # Step 3: outflow BC
        if step == 3:
            parameters.driver_cmd_line_args = [
                "parthenon/mesh/ix1_bc=outflow",
                "parthenon/mesh/ox1_bc=outflow",
                "parthenon/mesh/ix2_bc=outflow",
                "parthenon/mesh/ox2_bc=outflow",
                "parthenon/mesh/ix3_bc=outflow",
                "parthenon/mesh/ox3_bc=outflow",
                "parthenon/output0/id=outflow",
            ]

        parameters.coverage_status = "both"
        return parameters

    def Analyse(self, parameters):

        sys.path.insert(
            1,
            parameters.parthenon_path
            + "/scripts/python/packages/parthenon_tools/parthenon_tools",
        )

        try:
            import phdf
            from phdf_diff import compare
        except ModuleNotFoundError:
            print("Couldn't find modules to load Parthenon hdf5.")
            return False

        all_pass = True

        # Reflection: First comparing initial condition to final state
        res = compare(
            ["advection.reflecting.00000.phdf", "advection.reflecting.final.phdf"],
            check_metadata=False,
            quiet=True,
        )
        if res != 0:
            print("Double reflection test failed: output start != end ")
            all_pass = False

        # Compare that data in between is actually different.
        res = compare(
            ["advection.reflecting.00000.phdf", "advection.reflecting.00002.phdf"],
            check_metadata=False,
            quiet=True,
        )
        if res == 0:
            print("Double reflection test failed: data identical during sim.")
            all_pass = False

        # Periodic: First comparing initial condition to final state
        res = compare(
            ["advection.periodic.00000.phdf", "advection.periodic.final.phdf"],
            check_metadata=False,
            quiet=True,
        )
        if res != 0:
            print("Fully periodic test failed: output start != end ")
            all_pass = False

        # Compare that data in between is actually different.
        res = compare(
            ["advection.periodic.00000.phdf", "advection.periodic.00001.phdf"],
            check_metadata=False,
            quiet=True,
        )
        if res == 0:
            print("Fully periodic test failed: data identical during sim.")
            all_pass = False

        # Outflow: Check there's sth in the beginning...
        outflow_data_initial = phdf.phdf("advection.outflow.00000.phdf")
        advected_initial = outflow_data_initial.Get("advected")
        if np.sum(advected_initial) != 60.0:
            print("Unexpected initial data for outflow test.")
            all_pass = False

        # ... and nothing at the end
        outflow_data_final = phdf.phdf("advection.outflow.final.phdf")
        advected_final = outflow_data_final.Get("advected")
        if np.sum(advected_final) != 0.0:
            print("Some 'advected' did not leave the box in outflow test.")
            all_pass = False

        return all_pass
