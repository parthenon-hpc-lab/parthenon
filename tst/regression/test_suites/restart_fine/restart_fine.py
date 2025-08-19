# ========================================================================================
# Parthenon performance portable AMR framework
# Copyright(C) 2020-2024 The Parthenon collaboration
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
import sys
import utils.test_case


# To prevent littering up imported folders with .pyc files or __pycache_ folder
sys.dont_write_bytecode = True


class TestCase(utils.test_case.TestCaseAbs):
    def Prepare(self, parameters, step):
        # enable coverage testing on pass where restart
        # files are both read and written
        parameters.coverage_status = "both"

        # run baseline (to the very end)
        if step == 1:
            parameters.driver_cmd_line_args = ["parthenon/job/problem_id=gold"]
        elif step == 2:
            parameters.driver_cmd_line_args = [
                "-r",
                "gold.out0.00004.rhdf",
                "parthenon/job/problem_id=silver",
            ]
        # check that we can dynamically enable outputs
        else:
            parameters.driver_cmd_line_args = [
                "-r",
                "gold.out0.00005.rhdf",
                "parthenon/job/problem_id=bronze",
                "parthenon/output1/file_type=hdf5",
                "parthenon/output1/dt=0.25",
                "parthenon/output1/last_time=0.25",
                "parthenon/output1/variables=advection.C",
            ]
        return parameters

    def Analyse(self, parameters):
        sys.path.insert(
            1,
            parameters.parthenon_path
            + "/scripts/python/packages/parthenon_tools/parthenon_tools",
        )

        try:
            from phdf_diff import compare
        except ModuleNotFoundError:
            print("Couldn't find module to compare Parthenon hdf5 files.")
            return False

        success = True

        def compare_files(name, base="silver"):
            delta = compare(
                [
                    "gold.out0.%s.rhdf" % name,
                    "{}.out0.{}.rhdf".format(base, name),
                ],
                one=True,
            )

            if delta != 0:
                print(
                    "ERROR: Found difference between gold and silver output '%s'."
                    % name
                )
                return False
            delta = compare(
                [
                    "bronze.out1.%s.phdf" % name,
                    "{}.out2.{}.phdf".format(base, name),
                ],
                # no need for metadata as the dynamically added output will cause
                # different metadata and we're just interested in the right data
                # being there.
                one=True,
                check_metadata=False,
            )

            if delta != 0:
                print(
                    "ERROR: Found difference between gold and bronze output '%s'."
                    % name
                )
                return False

            return True

        # comapre a few files throughout the simulations
        success &= compare_files("final")

        return success
