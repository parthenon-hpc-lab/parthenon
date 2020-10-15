#========================================================================================
# Parthenon performance portable AMR framework
# Copyright(C) 2020 The Parthenon collaboration
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


class TestCase(utils.test_case.TestCaseAbs):
    def Prepare(self,parameters, step):
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
                'parthenon/job/problem_id=advection_3d', # change name for new outputs
                'parthenon/mesh/numlevel=2', # reduce AMR depth for smaller sim
                'parthenon/mesh/nx1=%d' % (32 * parameters.num_ranks ),
                'parthenon/meshblock/nx1=8',
                'parthenon/mesh/nx2=32',
                'parthenon/meshblock/nx2=8',
                'parthenon/mesh/nx3=32',
                'parthenon/meshblock/nx3=8',
                'parthenon/time/integrator=rk1',
                'Advection/cfl=0.3',
                ]
        # Same as step 1 but shortened for calculating coverage
        elif step == 3:
            parameters.coverage_status = "only-coverage"
            parameters.driver_cmd_line_args = [
                'parthenon/time/tlim=0.01',
                ]
        # Same as step 2 but shortened for calculating coverage
        elif step == 4:
            parameters.coverage_status = "only-coverage"
            parameters.driver_cmd_line_args = [
                'parthenon/job/problem_id=advection_3d', # change name for new outputs
                'parthenon/mesh/numlevel=2', # reduce AMR depth for smaller sim
                'parthenon/mesh/nx1=%d' % (32 * parameters.num_ranks),
                'parthenon/meshblock/nx1=8',
                'parthenon/mesh/nx2=32',
                'parthenon/meshblock/nx2=8',
                'parthenon/mesh/nx3=32',
                'parthenon/meshblock/nx3=8',
                'parthenon/time/integrator=rk1',
                'Advection/cfl=0.3',
                'parthenon/time/tlim=0.01',
                ]
        return parameters

    def Analyse(self,parameters):
        """
        Analyze the output and determine if the test passes.
        """
        analyze_status = True
        print(os.getcwd())

        # Determine path to parthenon installation
        # Fallback to relative path on failure
        try:
            parthenonPath = os.path.realpath(__file__)
            idx = parthenonPath.rindex('/parthenon/')
            parthenonPath = os.path.join(parthenonPath[:idx],'parthenon')
        except ValueError:
            baseDir = os.path.dirname(__file__)
            parthenonPath = baseDir + '/../../../..'
        sys.path.insert(1, parthenonPath+'/scripts/python')

        try:
            import phdf_diff 
        except ModuleNotFoundError:
            print("Couldn't find module to compare Parthenon hdf5 files.")
            return False

        # TODO(pgrete) make sure this also works/doesn't fail for the user
        ret_2d = phdf_diff.compare([
            'advection_2d.out0.00001.phdf',
            parthenonPath+'/tst/regression/gold_standard/advection_2d.out0.00001.phdf'])
        ret_3d = phdf_diff.compare([
            'advection_3d.out0.00001.phdf',
            parthenonPath+'/tst/regression/gold_standard/advection_3d.out0.00001.phdf'])
        
        if ret_2d != 0 or ret_3d != 0:
            analyze_status = False

        return analyze_status
