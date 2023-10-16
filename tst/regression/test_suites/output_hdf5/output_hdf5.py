# ========================================================================================
# Parthenon performance portable AMR framework
# Copyright(C) 2020-2023 The Parthenon collaboration
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
import numpy as np
import sys
import os
import utils.test_case
import h5py

# To prevent littering up imported folders with .pyc files or __pycache_ folder
sys.dont_write_bytecode = True


class TestCase(utils.test_case.TestCaseAbs):
    def Prepare(self, parameters, step):
        """
        Setup simulation parameters.
        """

        # TEST: 2D AMR
        if step == 1:
            # add param missing from input file to test cmdline override
            parameters.driver_cmd_line_args = [
                "parthenon/mesh/nx1=64",
            ]
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
                "parthenon/mesh/nx1=64",
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
            import phdf
        except ModuleNotFoundError:
            print("Couldn't find modules to read/compare Parthenon hdf5 files.")
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
                1.69755e-10,
                1.45889e-07,
            ],
        ]
        # check results in last row (at the final time of the sim)
        for i, val in enumerate(ref_results):
            if hst_2d[-1:, i] != val[1]:
                print(
                    "Wrong",
                    val[0],
                    "in hst output of 2D problem:",
                    hst_2d[-1:, i],
                    val[1],
                )
                analyze_status = False
            if hst_3d[-1:, i] != val[2]:
                print(
                    "Wrong",
                    val[0],
                    "in hst output of 3D problem:",
                    hst_3d[-1:, i],
                    val[2],
                )
                analyze_status = False

        # Parameter override warning form cmdline should be shown for each run
        for output in parameters.stdouts:
            warning_found = False
            for line in output.decode("utf-8").split("\n"):
                if ("nx1" in line) and ("will be added" in line):
                    warning_found = True
            if not warning_found:
                print(
                    f"\n\n!!!! TEST ERROR !!!\n"
                    f"Parameter override not triggered, but you should never be here "
                    f"because the simulation should not have started in the first place. "
                    f"Something is really wrong. Please open an issue on GitHub"
                )
                analyze_status = False

        # Checking Parthenon histograms versus numpy ones
        for dim in [2, 3]:
            # 1D histogram with binning of a variable with bins defined by a var
            data = phdf.phdf(f"advection_{dim}d.out0.final.phdf")
            advected = data.Get("advected")
            hist_np1d = np.histogram(
                advected, [1e-9, 1e-4, 1e-1, 2e-1, 5e-1, 1e0], weights=advected
            )
            with h5py.File(
                f"advection_{dim}d.out2.histograms.final.hdf", "r"
            ) as infile:
                hist_parth = infile["0/data"][:]
                all_close = np.allclose(hist_parth, hist_np1d[0])
                if not all_close:
                    print(f"1D variable-based hist for {dim}D setup don't match")
                    analyze_status = False

            # 2D histogram with binning of a variable with bins defined by one var and one coord
            omadvected = data.Get("one_minus_advected_sq")
            z, y, x = data.GetVolumeLocations()
            hist_np2d = np.histogram2d(
                x.flatten(),
                omadvected.flatten(),
                [[-0.5, -0.25, 0, 0.25, 0.5], [0, 0.5, 1]],
                weights=advected.flatten(),
            )
            with h5py.File(
                f"advection_{dim}d.out2.histograms.final.hdf", "r"
            ) as infile:
                hist_parth = infile["1/data"][:]
                # testing slices separately to ensure matching numpy convention
                all_close = np.allclose(hist_parth[:, 0], hist_np2d[0][:, 0])
                all_close &= np.allclose(hist_parth[:, 1], hist_np2d[0][:, 1])
                if not all_close:
                    print(f"2D hist for {dim}D setup don't match")
                    analyze_status = False

            # 1D histogram (simple sampling) with bins defined by a var
            hist_np1d = np.histogram(advected, [1e-9, 1e-4, 1e-1, 2e-1, 5e-1, 1e0])
            with h5py.File(
                f"advection_{dim}d.out2.histograms.final.hdf", "r"
            ) as infile:
                hist_parth = infile["2/data"][:]
                all_close = np.allclose(hist_parth, hist_np1d[0])
                if not all_close:
                    print(f"1D sampling-based hist for {dim}D setup don't match")
                    analyze_status = False

        return analyze_status
