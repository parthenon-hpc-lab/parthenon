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
import subprocess
import utils.test_case
""" To prevent littering up imported folders with .pyc files or __pycache_ folder"""
sys.dont_write_bytecode = True

class TestCase(utils.test_case.TestCaseAbs):
    def Prepare(self, parameters, step):
            
        # enable coverage testing on pass where restart
        # files are both read and written
        parameters.coverage_status = "both"
        
        if step == 1:
            parameters.driver_cmd_line_args = ['parthenon/job/problem_id=gold']
        else:
            parameters.driver_cmd_line_args = [
                '-r',
                'gold.out0.00001.rhdf',
                'parthenon/job/problem_id=silver'
            ]
            
        return parameters

    def Analyse(self, parameters):
        # HACK: On some systems, including LANL Darwin Power9, importing the
        # h5py module causes future uses of `mpiexec` to fail - there appears to
        # be some buried call to `MPI_Init` in the import of `h5py` that causes
        # this. Therefore, we run the script in a child process for hygenic
        # purposes.
        #
        # For more information, see:
        # https://github.com/lanl/parthenon/issues/312
        script = """
import h5py
import sys

# spotcheck one variable
goldFile = 'gold.out0.00002.rhdf'
silverFile = 'silver.out0.00002.rhdf'

gold = h5py.File(goldFile,'r')
silver = h5py.File(silverFile,'r')

varName = "/advected"
goldData = gold[varName][:].flatten()
silverData = gold[varName][:].flatten()

# spot check on one variable
maxdiff = max(abs(goldData-silverData))
print('Variable: %s, diff=%g, N=%d'%(varName,maxdiff,len(goldData)))

if maxdiff == 0.0:
    exit = 0
else:
    exit = 1

sys.exit(exit)
"""
        proc = subprocess.run([sys.executable, "-c", script])

        return (proc.returncode == 0)
