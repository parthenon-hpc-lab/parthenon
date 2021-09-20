# ========================================================================================
# Parthenon performance portable AMR framework
# Copyright(C) 2020 The Parthenon collaboration
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
import utils.test_case

""" To prevent littering up imported folders with .pyc files or __pycache_ folder"""
sys.dont_write_bytecode = True


class TestCase(utils.test_case.TestCaseAbs):
    def Prepare(self, parameters, step):

        parameters.coverage_status = "both"
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

        delta = compare(
            [
                "sparse.out0.00002.phdf",
                parameters.parthenon_path
                + "/tst/regression/gold_standard/sparse.out0.00002.phdf",
            ],
            # the gold file was made with true sparse but no MPI, so use tolerance 0 when running
            # this test without MPI, but use tolerance 1.0e-7 when running with MPI, since MPI
            # results differ from non-MPI results. The test are run with allocation threshold
            # 1.0e-5, so they agree with fake sparse output to a tolerance of 1.0e-5 with and
            # without MPI
            tol=0 if parameters.mpi_cmd == "" else 1.0e-7,
        )

        return delta == 0
