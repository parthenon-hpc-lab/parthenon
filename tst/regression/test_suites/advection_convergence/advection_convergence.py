# ========================================================================================
# Athena++ astrophysical MHD code
# Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
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
import matplotlib

matplotlib.use("agg")
import matplotlib.pylab as plt
import sys
import os
import utils.test_case

# To prevent littering up imported folders with .pyc files or __pycache_ folder
sys.dont_write_bytecode = True

# if this is updated make sure to update the assert statements for the number of MPI ranks, too
lin_res = [32, 64, 128, 256, 512]  # resolution for linear convergence


class TestCase(utils.test_case.TestCaseAbs):
    def Prepare(self, parameters, step):
        """
        Any preprocessing that is needed before the drive is run can be done in
        this method

        This includes preparing files or any other pre processing steps that
        need to be implemented.  The method also provides access to the
        parameters object which controls which parameters are being used to run
        the driver.

        It is possible to append arguments to the driver_cmd_line_args if it is
        desired to  override the parthenon input file. Each element in the list
        is simply a string of the form '<block>/<field>=<value>', where the
        contents of the string are exactly what one would type on the command
        line run running a parthenon driver.

        As an example if the following block was uncommented it would overwrite
        any of the parameters that were specified in the parthenon input file
        parameters.driver_cmd_line_args = ['output1/file_type=vtk',
                'output1/variable=cons',
                'output1/dt=0.4',
                'time/tlim=0.4',
                'mesh/nx1=400']
        """

        n_res = len(lin_res)
        # make sure we can evenly distribute the MeshBlock sizes
        err_msg = "Num ranks must be multiples of 2 for convergence test."
        assert parameters.num_ranks == 1 or parameters.num_ranks % 2 == 0, err_msg
        # ensure a minimum block size of 4
        assert (
            lin_res[0] / parameters.num_ranks >= 4
        ), "Use <= 8 ranks for convergence test."

        # TEST: Advection only in x-direction
        # using nx2 = nx3 = 4 > 1 for identical errors between dimensions
        if step <= n_res:

            if lin_res[step - 1] == 32:
                parameters.coverage_status = "both"
            parameters.driver_cmd_line_args = [
                "parthenon/mesh/nx1=%d" % lin_res[step % n_res - 1],
                "parthenon/meshblock/nx1=%d"
                % (lin_res[step % n_res - 1] // parameters.num_ranks),
                "parthenon/mesh/nx2=1",
                "parthenon/meshblock/nx2=1",
                "parthenon/mesh/nx3=1",
                "parthenon/meshblock/nx3=1",
                "Advection/vy=0.0",
                "Advection/vz=0.0",
            ]
        # TEST: Advection only in y-direction
        elif step <= 2 * n_res:
            # Only run coverage for 32 case
            if lin_res[step % n_res - 1] == 32:
                parameters.coverage_status = "both"
            parameters.driver_cmd_line_args = [
                "parthenon/mesh/nx1=4",
                "parthenon/meshblock/nx1=4",
                "parthenon/mesh/nx2=%d" % lin_res[step % n_res - 1],
                "parthenon/meshblock/nx2=%d"
                % (lin_res[step % n_res - 1] // parameters.num_ranks),
                "parthenon/mesh/nx3=1",
                "parthenon/meshblock/nx3=1",
                "Advection/vx=0.0",
                "Advection/vz=0.0",
                "Advection/ang_3_vert=true",
            ]
        # TEST: Advection only in z-direction
        elif step <= 3 * n_res:
            if lin_res[step % n_res - 1] == 32:
                parameters.coverage_status = "both"
            parameters.driver_cmd_line_args = [
                "parthenon/mesh/nx1=4",
                "parthenon/meshblock/nx1=4",
                "parthenon/mesh/nx2=4",
                "parthenon/meshblock/nx2=4",
                "parthenon/mesh/nx3=%d" % lin_res[step % n_res - 1],
                "parthenon/meshblock/nx3=%d"
                % (lin_res[step % n_res - 1] // parameters.num_ranks),
                "Advection/vx=0.0",
                "Advection/vy=0.0",
                "Advection/ang_2_vert=true",
            ]
        # TEST: Advection only in x at highest res with identical params
        elif step == 3 * n_res + 1:
            parameters.driver_cmd_line_args = [
                "parthenon/mesh/nx1=%d" % lin_res[-1],
                "parthenon/meshblock/nx1=%d" % (lin_res[-1] // parameters.num_ranks),
                "parthenon/mesh/nx2=4",
                "parthenon/meshblock/nx2=4",
                "parthenon/mesh/nx3=4",
                "parthenon/meshblock/nx3=4",
                "parthenon/mesh/x1min=-0.5",
                "parthenon/mesh/x1max=0.5",
                "parthenon/mesh/x2min=-0.5",
                "parthenon/mesh/x2max=0.5",
                "parthenon/mesh/x3min=-0.5",
                "parthenon/mesh/x3max=0.5",
                "Advection/vx=1.0",
                "Advection/vy=0.0",
                "Advection/vz=0.0",
            ]
        # TEST: Advection only in y at highest res with identical params
        elif step == 3 * n_res + 2:
            parameters.driver_cmd_line_args = [
                "parthenon/mesh/nx1=4",
                "parthenon/meshblock/nx1=4",
                "parthenon/mesh/nx2=%d" % lin_res[-1],
                "parthenon/meshblock/nx2=%d" % (lin_res[-1] // parameters.num_ranks),
                "parthenon/mesh/nx3=4",
                "parthenon/meshblock/nx3=4",
                "parthenon/mesh/x1min=-0.5",
                "parthenon/mesh/x1max=0.5",
                "parthenon/mesh/x2min=-0.5",
                "parthenon/mesh/x2max=0.5",
                "parthenon/mesh/x3min=-0.5",
                "parthenon/mesh/x3max=0.5",
                "Advection/vx=0.0",
                "Advection/vy=1.0",
                "Advection/vz=0.0",
                "Advection/ang_3_vert=true",
            ]
        # TEST: Advection only in z at highest res with identical params
        elif step == 3 * n_res + 3:
            parameters.driver_cmd_line_args = [
                "parthenon/mesh/nx1=4",
                "parthenon/meshblock/nx1=4",
                "parthenon/mesh/nx2=4",
                "parthenon/meshblock/nx2=4",
                "parthenon/mesh/nx3=%d" % lin_res[-1],
                "parthenon/meshblock/nx3=%d" % (lin_res[-1] // parameters.num_ranks),
                "parthenon/mesh/x1min=-0.5",
                "parthenon/mesh/x1max=0.5",
                "parthenon/mesh/x2min=-0.5",
                "parthenon/mesh/x2max=0.5",
                "parthenon/mesh/x3min=-0.5",
                "parthenon/mesh/x3max=0.5",
                "Advection/vx=0.0",
                "Advection/vy=0.0",
                "Advection/vz=1.0",
                "Advection/ang_2_vert=true",
            ]
        # TEST: Advection at along diagonal with dx != dy != dz and Lx != Ly != Lz (half res)
        elif step == 3 * n_res + 4:
            parameters.driver_cmd_line_args = [
                "parthenon/mesh/nx1=32",
                "parthenon/meshblock/nx1=%d" % (32 // parameters.num_ranks),
                "parthenon/mesh/nx2=32",
                "parthenon/meshblock/nx2=32",
                "parthenon/mesh/nx3=32",
                "parthenon/meshblock/nx3=32",
                "Advection/ang_2=-999.9",
                "Advection/ang_3=-999.9",
            ]
        # TEST: Advection at along diagonal with dx != dy != dz and Lx != Ly != Lz (def res)
        elif step == 3 * n_res + 5:
            parameters.driver_cmd_line_args = [
                "parthenon/meshblock/nx1=%d" % (64 // parameters.num_ranks),
                "Advection/ang_2=-999.9",
                "Advection/ang_3=-999.9",
            ]
        # TEST: Advection at along diagonal with dx != dy != dz and Lx != Ly != Lz (def res)
        # using multiple MeshBlocks
        elif step == 3 * n_res + 6:
            parameters.driver_cmd_line_args = [
                "parthenon/meshblock/nx1=8",
                "parthenon/meshblock/nx2=16",
                "parthenon/meshblock/nx3=8",
                "Advection/ang_2=-999.9",
                "Advection/ang_3=-999.9",
            ]
        # TEST: AMR test - diagonal advection low res no AMR (low res baseline)
        # Lx != Ly != Lz and dx != dy != dz
        elif step == 3 * n_res + 7:
            parameters.driver_cmd_line_args = [
                "parthenon/mesh/nx1=40",
                "parthenon/meshblock/nx1=20",
                "parthenon/mesh/nx2=30",
                "parthenon/meshblock/nx2=15",
                "parthenon/mesh/nx3=36",
                "parthenon/meshblock/nx3=18",
                "parthenon/mesh/x1min=-1.5",
                "parthenon/mesh/x1max=1.5",
                "parthenon/mesh/x2min=-0.75",
                "parthenon/mesh/x2max=0.75",
                "parthenon/mesh/x3min=-1.0",
                "parthenon/mesh/x3max=1.0",
                "Advection/vx=3.0",
                "Advection/vy=1.5",
                "Advection/vz=2.0",
                "Advection/profile=smooth_gaussian",
                "Advection/amp=1.0",
            ]
        # TEST: AMR test - diagonal advection higher res no AMR
        # Lx != Ly != Lz and dx != dy != dz
        elif step == 3 * n_res + 8:
            parameters.driver_cmd_line_args = [
                "parthenon/mesh/nx1=80",
                "parthenon/meshblock/nx1=40",
                "parthenon/mesh/nx2=60",
                "parthenon/meshblock/nx2=30",
                "parthenon/mesh/nx3=72",
                "parthenon/meshblock/nx3=36",
                "parthenon/mesh/x1min=-1.5",
                "parthenon/mesh/x1max=1.5",
                "parthenon/mesh/x2min=-0.75",
                "parthenon/mesh/x2max=0.75",
                "parthenon/mesh/x3min=-1.0",
                "parthenon/mesh/x3max=1.0",
                "Advection/vx=3.0",
                "Advection/vy=1.5",
                "Advection/vz=2.0",
                "Advection/profile=smooth_gaussian",
                "Advection/amp=1.0",
            ]
        # TEST: AMR test - diagonal advection low res with AMR
        # Lx != Ly != Lz and dx != dy != dz
        elif step == 3 * n_res + 9:
            parameters.driver_cmd_line_args = [
                "parthenon/mesh/refinement=adaptive",
                "parthenon/mesh/nx1=40",
                "parthenon/meshblock/nx1=8",
                "parthenon/mesh/nx2=30",
                "parthenon/meshblock/nx2=6",
                "parthenon/mesh/nx3=36",
                "parthenon/meshblock/nx3=6",
                "parthenon/mesh/x1min=-1.5",
                "parthenon/mesh/x1max=1.5",
                "parthenon/mesh/x2min=-0.75",
                "parthenon/mesh/x2max=0.75",
                "parthenon/mesh/x3min=-1.0",
                "parthenon/mesh/x3max=1.0",
                "Advection/vx=3.0",
                "Advection/vy=1.5",
                "Advection/vz=2.0",
                "Advection/profile=smooth_gaussian",
                "Advection/amp=1.0",
            ]
        # TEST: AMR test - diagonal advection low res with AMR shortened time step
        # Lx != Ly != Lz and dx != dy != dz
        elif step == 3 * n_res + 10:
            parameters.coverage_status = "only-coverage"
            parameters.driver_cmd_line_args = [
                "parthenon/time/tlim=0.01",
                "parthenon/mesh/refinement=adaptive",
                "parthenon/mesh/nx1=40",
                "parthenon/meshblock/nx1=8",
                "parthenon/mesh/nx2=30",
                "parthenon/meshblock/nx2=6",
                "parthenon/mesh/nx3=36",
                "parthenon/meshblock/nx3=6",
                "parthenon/mesh/x1min=-1.5",
                "parthenon/mesh/x1max=1.5",
                "parthenon/mesh/x2min=-0.75",
                "parthenon/mesh/x2max=0.75",
                "parthenon/mesh/x3min=-1.0",
                "parthenon/mesh/x3max=1.0",
                "Advection/vx=3.0",
                "Advection/vy=1.5",
                "Advection/vz=2.0",
                "Advection/profile=smooth_gaussian",
                "Advection/amp=1.0",
            ]
        return parameters

    def Analyse(self, parameters):
        """
        Analyze the output and determine if the test passes.

        This function is called after the driver has been executed. It is
        responsible for reading whatever data it needs and making a judgment
        about whether or not the test passes. It takes no inputs. Output should
        be True (test passes) or False (test fails).

        The parameters that are passed in provide the paths to relevant
        locations and commands. Of particular importance is the path to the
        output folder. All files from a drivers run should appear in and output
        folder located in
        parthenon/tst/regression/test_suites/test_name/output.

        It is possible in this function to read any of the output files such as
        hdf5 output and compare them to expected quantities.

        """

        try:
            f = open(os.path.join(parameters.output_path, "advection-errors.dat"), "r")
            lines = f.readlines()

            f.close()
        except IOError:
            print("Advection error file not accessible")

        analyze_status = True

        if len(lines) != 25:
            print("Missing lines in output file. Expected 25, but got ", len(lines))
            print(
                "CAREFUL!!! All following logs may be misleading (tests have fixed indices)."
            )
            analyze_status = False

        # ensure errors in all three directions are identical
        n_res = len(lin_res)

        for i in range(n_res):
            # sample line: 128  4  4  427  3.258335e-08   1.570405e+00  5.116905e-08
            line_x = lines[i + 0 * n_res + 1].split()
            line_y = lines[i + 1 * n_res + 1].split()
            line_z = lines[i + 2 * n_res + 1].split()
            # num iterations must be identical
            for j in [3]:
                if line_x[j] != line_y[j]:
                    print("Cycle mismatch between X and Y", line_x, line_y)
                    analyze_status = False
                if line_x[j] != line_z[j]:
                    print("Cycle mismatch between X and Z", line_x, line_z)
                    analyze_status = False
            # absolute errors should be close
            for j in [4, 6]:
                if not np.isclose(
                    2 * float(line_x[j]), float(line_y[j]), rtol=1e-6, atol=0.0
                ):
                    print("Mismatch between rel. X and Y", line_x, line_y)
                    analyze_status = False
                if not np.isclose(
                    3 * float(line_x[j]), float(line_z[j]), rtol=1e-6, atol=0.0
                ):
                    print("Mismatch between rel. X and Z", line_x, line_z)
                    analyze_status = False

        # test for error tolerance
        err_512 = lines[5].split()

        # double checking that this is the correct line
        if int(err_512[0]) != 512:
            print("Mismtach in line. Expected 512 in nx1 but got:", lines[5])
            analyze_status = False
        if float(err_512[4]) >= 1.7e-8:
            print("Error too large. Expected < 1.7e-8 but got:", err_512[4])
            analyze_status = False

        offset = 3 * n_res + 1
        # make sure errors are identical in different dims for identical params
        if lines[offset][11:] != lines[offset + 1][11:]:
            print("X and Y dim don't match: ", lines[offset], lines[offset + 1])
            analyze_status = False
        if lines[offset][11:] != lines[offset + 2][11:]:
            print("X and Z dim don't match: ", lines[offset], lines[offset + 2])
            analyze_status = False

        offset += 3
        # check convergence and error for diagnoal advection
        if float(lines[offset].split()[4]) / float(lines[offset + 1].split()[4]) < 1.9:
            print(
                "Advection across diagnonal did not converge: ",
                lines[offset],
                lines[offset + 1],
            )
            analyze_status = False
        if float(lines[offset + 1].split()[4]) >= 2.15e-7:
            print(
                "Error too large in diagnoal advection. Expected < 2.15e-7 but got:",
                lines[offset + 1].split()[4],
            )
            analyze_status = False

        # ensure that using a single meshblock for the entire mesh and multiple give same result
        if lines[offset + 2] != lines[offset + 1]:
            print(
                "Single meshblock error:",
                lines[offset + 1],
                "is different from usingle multiple: ",
                lines[offset + 2],
            )
            analyze_status = False

        offset += 3
        # AMR test
        # make sure error got smaller when using higher res and compare absolute error val
        if not np.isclose(
            float(lines[offset].split()[4]) / 1.089750e-03, 1.0, atol=0.0, rtol=1e-6
        ):
            print(
                "Mismatch in error for low res AMR baseline: ",
                lines[offset],
                "but expected: 1.089750e-03",
            )
            analyze_status = False
        if not np.isclose(
            float(lines[offset + 1].split()[4]) / 9.749603e-04, 1.0, atol=0.0, rtol=1e-6
        ):
            print(
                "Mismatch in error for higher res AMR baseline: ",
                lines[offset + 1],
                "but expected: 9.749603e-04",
            )
            analyze_status = False

        # ensure that higher res static run and lower res AMR run provide similar errors
        ratio = float(lines[offset + 1].split()[4]) / float(
            lines[offset + 2].split()[4]
        )
        if ratio > 1.0:
            print(
                "AMR run is more accurate than static grid run at higher res:",
                lines[offset + 2],
                lines[offset + 1],
            )
            analyze_status = False
        if ratio < 0.999:
            print(
                "AMR error too large compared to static grid run (threshold used 0.999):",
                lines[offset + 2],
                lines[offset + 1],
            )
            analyze_status = False

        # Plot results
        data = np.genfromtxt(
            os.path.join(parameters.output_path, "advection-errors.dat")
        )

        n_res = 5
        sym = "xo+"

        for i, x in enumerate("xyz"):
            plt.plot(
                data[i * n_res : (i + 1) * n_res, 0 + i],
                data[i * n_res : (i + 1) * n_res, 4],
                marker=sym[i],
                label=x + "-dir (vary res)",
            )
            plt.plot(
                data[3 * n_res + i, 0 + i],
                data[3 * n_res + i, 4],
                lw=0,
                marker=sym[i],
                label=x + "-dir (same res)",
                alpha=0.5,
            )

        plt.plot(
            data[3 * n_res + 3 : 3 * n_res + 3 + 2, 0],
            data[3 * n_res + 3 : 3 * n_res + 3 + 2, 4],
            marker="^",
            label="oblique",
        )

        plt.plot([32, 512], [3e-7, 3e-7 / (512 / 32)], "--", label="first order")

        plt.legend()
        plt.xscale("log")
        plt.yscale("log")
        plt.ylabel("L1 err")
        plt.xlabel("Linear resolution")
        plt.savefig(
            os.path.join(parameters.output_path, "advection-errors.png"),
            bbox_inches="tight",
        )

        return analyze_status
