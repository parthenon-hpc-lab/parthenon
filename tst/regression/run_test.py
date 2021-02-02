#!/usr/bin/env python3
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

# Python modules
import argparse
import os
import sys

try:
    # Configure mpi4py to not initialize MPI automatically
    # See: https://github.com/lanl/parthenon/pull/320
    import mpi4py
    mpi4py.rc(initialize=False)
except ImportError: ()

""" To prevent littering up imported folders with .pyc files"""
sys.dont_write_bytecode = True

# Parthenon modules
import utils.test_case as tc 

def checkRunScriptLocation(run_test_py_path):
  
    """ Check that run_test is in the correct folder """
    if not os.path.normpath(run_test_py_path).endswith(os.path.normpath("tst/regression")):
        error_msg = "Cannot run run_test.py, it is not in the correct directory, must be "
        error_msg += "kept in tst/regression"
        raise TestError(error_msg)

    """ Check that test_suites folder exists """
    if not os.path.isdir(os.path.join(run_test_py_path,'test_suites')): 
        raise TestError("Cannot run run_test.py, the test_suites folder is missing.")

# Main function
def main(**kwargs):

    print('\n')
    print('\n'.join(['{}={!r}'.format(k, v) for k, v in kwargs.items()]))
    if hasattr(kwargs,'mpirun_opts'):
        if kwargs.mpirun == "":
            raise TestError("Cannot provide --mpirun_opts without specifying --mpirun")

    print("*****************************************************************")
    print("Beginning Python regression testing script")
    print("*****************************************************************\n")

    run_test_py_path = os.path.dirname(os.path.realpath(__file__))
    checkRunScriptLocation(run_test_py_path) 

    print("Initializing Test Case")

    test_manager = tc.TestManager(run_test_py_path,**kwargs)

    print("Make output folder in test if does not exist")

    test_manager.MakeOutputFolder()

    for step in range(1,kwargs['num_steps'] + 1):
        test_manager.Prepare(step)

        test_manager.Run()

    test_result = test_manager.Analyse()

    if test_result == True:
        return 0
    else:
        raise TestError("Test " + test_manager.test + " failed")


# Exception for unexpected behavior by individual tests
class TestError(RuntimeError):
    pass

# Execute main function
if __name__ == '__main__':

    parser = argparse.ArgumentParser("run_test.py a regression testing script")
    
    desc = ('name of the test directory, relative to test_suites/, '
               'excluding .py.')
   
    parser.add_argument('--test_dir','-t',
                        type=str,
                        nargs=1,
                        required=True,
                        help=desc)

    parser.add_argument('--output_dir','-o',
                        type=str,
                        default="",
                        help="path to simulation outputs. " +
                             "Defaults to individual \"output\" folders in regression src.")

    parser.add_argument("--driver", "-dr",
                        type=str,
                        default=None,
                        nargs=1,
                        required=True,
                        help='path to the driver to test, where the driver is a binary executable')

    parser.add_argument("--driver_input", "-dr_in",
                        type=str,
                        default=None,
                        nargs=1,
                        required=True,
                        help='path to input file, to pass to driver')

    parser.add_argument("--kokkos_args", "-k_a",
                        default=[],
                        action='append',
                        help='kokkos arguments to pass to driver')

    parser.add_argument("--num_steps", "-n",
                        type=int,
                        default=1,
                        required=False,
                        help='Number of steps in test. Default: 1')

    parser.add_argument('--mpirun',
                        default='',
                        # 2x MPI, Slurm, PBS/Torque, LSF, Cray ALPS
                        nargs=1,
                        help='change MPI run wrapper command (e.g. for job schedulers)')

    parser.add_argument('--mpirun_ranks_flag',
                        default=None,
                        type=str,
                        help='Flag for the number of ranks')

    parser.add_argument('--mpirun_ranks_num',
                        default=1,
                        type=int,
                        help='Number of ranks')

    parser.add_argument('--mpirun_opts',
                        default=[],
                        action='append',
                        help='add options to mpirun command')

    parser.add_argument('--coverage','-c',
                        action='store_true',
                        help='Run test cases where coverage has been enabled')

    args = parser.parse_args()
    try:
        main(**vars(args))
    except Exception:
        raise
