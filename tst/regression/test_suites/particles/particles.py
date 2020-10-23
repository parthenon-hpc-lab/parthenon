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
import matplotlib
matplotlib.use('agg')
import matplotlib.pylab as plt
import sys
import os
import utils.test_case

""" To prevent littering up imported folders with .pyc files or __pycache__ folder"""
sys.dont_write_bytecode = True

class TestCase(utils.test_case.TestCaseAbs):
    def Prepare(self, parameters, step):
        err_msg = "Number of ranks must be 1 for particles at the moment"
        assert parameters.num_ranks == 1, err_msg
        return parameters

    def Analyse(self, parameters):
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

        ret = phdf_diff.compare([
            'particles.out0.00001.phdf',
            parthenonPath+'/tst/regression/gold_standard/particles.out0.00001.phdf'])

        if ret != 0:
            analyze_status = False

        return analyze_status
