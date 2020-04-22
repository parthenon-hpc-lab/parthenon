# Functions for interfacing with Athena++ during testing
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
import os
import sys
import subprocess

""" To prevent littering up imported folders with .pyc files or __pycache_ folder"""
sys.dont_write_bytecode = True
# Functions for running Parthenon
def run(parameters):
   
    run_command = []
    run_command.append(parameters.mpi_cmd)
    for opt in parameters.mpi_opts:
        run_command.extend(opt.split()) 
    #run_command.extend(parameters.mpi_opts)
    run_command.append(parameters.driver_path)  
    run_command.append('-i')
    run_command.append(parameters.driver_input_path)
    print("Command to execute driver")
    print(run_command)
    try:
        subprocess.check_call(run_command)
    except subprocess.CalledProcessError as err:
        raise ParthenonError('\nReturn code {0} from command \'{1}\''
                          .format(err.returncode, ' '.join(err.cmd)))

# General exception class for these functions
class ParthenonError(RuntimeError):
    pass
