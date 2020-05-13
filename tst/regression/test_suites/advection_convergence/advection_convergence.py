#========================================================================================
# Athena++ astrophysical MHD code
# Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
# Licensed under the 3-clause BSD License, see LICENSE file for details
#========================================================================================
# (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001 for Los
# Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
# for the U.S. Department of Energy/National Nuclear Security Administration. All rights
# in the program are reserved by Triad National Security, LLC, and the U.S. Department
# of Energy/National Nuclear Security Administration. The Government is granted for
# itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
# license in this material to reproduce, prepare derivative works, distribute copies to
# the public, perform publicly and display publicly, and to permit others to do so.
#========================================================================================

# Modules
import math
import numpy as np
import sys
import os
import utils.test_case

""" To prevent littering up imported folders with .pyc files or __pycache_ folder"""
sys.dont_write_bytecode = True

lin_res = [32, 64, 128, 256, 512] # resolution for linear convergence

class TestCase(utils.test_case.TestCaseAbs):
    def Prepare(self,parameters, step):
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

        # TEST: Advection only in x-direction 
        # using nx2 = nx3 = 4 > 1 for identical errors between dimensions
        if step <= n_res:
            parameters.driver_cmd_line_args = [
                'parthenon/mesh/nx1=%d' % lin_res[step % n_res -1],
                'parthenon/meshblock/nx1=%d' % lin_res[step % n_res -1],
                'parthenon/mesh/nx2=4',
                'parthenon/meshblock/nx2=4',
                'parthenon/mesh/nx3=4',
                'parthenon/meshblock/nx3=4',
                'Advection/vy=0.0',
                'Advection/vz=0.0',
                ]
        # TEST: Advection only in y-direction
        elif step <= 2*n_res:
            parameters.driver_cmd_line_args = [
                'parthenon/mesh/nx1=4',
                'parthenon/meshblock/nx1=4',
                'parthenon/mesh/nx2=%d' % lin_res[step % n_res -1],
                'parthenon/meshblock/nx2=%d' % lin_res[step % n_res -1],
                'parthenon/mesh/nx3=4',
                'parthenon/meshblock/nx3=4',
                'Advection/vx=0.0',
                'Advection/vz=0.0',
                ]
        # TEST: Advection only in z-direction
        elif step <= 3*n_res:
            parameters.driver_cmd_line_args = [
                'parthenon/mesh/nx1=4',
                'parthenon/meshblock/nx1=4',
                'parthenon/mesh/nx2=4',
                'parthenon/meshblock/nx2=4',
                'parthenon/mesh/nx3=%d' % lin_res[step % n_res -1],
                'parthenon/meshblock/nx3=%d' % lin_res[step % n_res -1],
                'Advection/vx=0.0',
                'Advection/vy=0.0',
                ]
        # TEST: Advection at along diagonal with dx != dy != dz and Lx != Ly != Lz (half res)
        elif step == 16:
            parameters.driver_cmd_line_args = [
                'parthenon/mesh/nx1=32',
                'parthenon/meshblock/nx1=32',
                'parthenon/mesh/nx2=32',
                'parthenon/meshblock/nx2=32',
                'parthenon/mesh/nx3=32',
                'parthenon/meshblock/nx3=32',
                ]
        # TEST: Advection at along diagonal with dx != dy != dz and Lx != Ly != Lz (def res)
        elif step == 17:
            parameters.driver_cmd_line_args = [
                ]
        # TEST: Advection at along diagonal with dx != dy != dz and Lx != Ly != Lz (def res)
        # using multiple MeshBlocks
        elif step == 18:
            parameters.driver_cmd_line_args = [
                'parthenon/meshblock/nx1=8',
                'parthenon/meshblock/nx2=16',
                'parthenon/meshblock/nx3=8',
                ]

        return parameters

    def Analyse(self,parameters):
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
            f = open(os.path.join(parameters.output_path, "advection-errors.dat"),"r")
            lines = f.readlines()

            f.close()
        except IOError:
            print("Advection error file not accessible")

        analyze_status = True

        # ensure errors in all three directions are identical
        n_res = len(lin_res)

        for i in range(n_res):
            # sample line: 128  4  4  427  3.258335e-08   1.570405e+00  5.116905e-08
            line_x = lines[i+0*n_res+1].split()
            line_y = lines[i+1*n_res+1].split()
            line_z = lines[i+2*n_res+1].split()
            for j in range(3,7):
                if line_x[j] != line_y[j]:
                    print("Mismatch between ", line_x, line_y)
                    analyze_status = False
                if line_x[j] != line_z[j]:
                    print("Mismatch between ", line_x, line_z)
                    analyze_status = False

        # test for error tolerance
        err_512 = lines[5].split()

        # double checking that this is the correct line
        if int(err_512[0]) != 512:
            print("Mismtach in line. Expected 512 in nx1 but got:", lines[5])
            analyze_status = False
        if float(err_512[4]) >= 8.5e-9:
            print("Error too large. Expected < 8.5e-9 but got:", err_512[4])
            analyze_status = False

        # ensure that using a single meshblock for the entire mesh and multiple give same result
        if lines[-1] != lines[-2]:
            print("Single meshblock error:", lines[-2],
                  "is different from usingle multiple: ", lines[-1])
            analyze_status = False

        return analyze_status
