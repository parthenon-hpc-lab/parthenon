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

""" To prevent littering up imported folders with .pyc files"""
sys.dont_write_bytecode = True

# Parthenon modules
import utils.test_case as tc 

""" Split a path into folders """
def SplitPathIntoFolders(path):
    folders = []
    while 1:
        path, folder = os.path.split(path)
        if folder != "":
            folders.append(folder)
        else:
            if path != "":
                folders.append(path)
            break
    return folders

def checkRunScriptLocation(run_test_py_path):
  
    """ Check that run_test is in the correct folder """
    folders = SplitPathIntoFolders(run_test_py_path)
    if( os.path.join(folders[2], folders[1], folders[0]) != "parthenon/tst/regression" ): 
        error_msg = "Cannot run run_test.py, it is not in the correct directory, must be "
        error_msg += "kept in parthenon/tst/regression"
        raise TestError(error_msg)

    """ Check that test_suites folder exists """
    if( not os.path.isdir(os.path.join(run_test_py_path,'test_suites')) ): 
        raise TestError("Cannot run run_test.py, the test_suites folder is missing.")

# Main function
def main(**kwargs):

    print("*****************************************************************")
    print("Beginning Python regression testing script")
    print("*****************************************************************")

    run_test_py_path = os.path.dirname(os.path.realpath(__file__))
    checkRunScriptLocation(run_test_py_path) 

    print("Initializing Test Case")

    test_case = tc.TestCase(run_test_py_path,**kwargs)

    print("Cleaning output folder in test")

    test_case.CleanOutputFolder()

    test_case.Run()

    test_result = test_case.Analyze()

    if (test_result == True):
        return 0
    else:
        raise TestError("Test " + test_case.test + " failed")


# Exception for unexpected behavior by individual tests
class TestError(RuntimeError):
    pass

# Execute main function
if __name__ == '__main__':
    help_msg = ('name of the test directory, relative to test_suites/, '
               'excluding .py.')
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir','-t',
                        type=str,
                        default=None,
                        nargs=1,
                        required=True,
                        help=help_msg)

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

    parser.add_argument('--mpirun',
                        default='mpirun',
                        # 2x MPI, Slurm, PBS/Torque, LSF, Cray ALPS
                        choices=['mpirun', 'mpiexec', 'srun', 'qsub', 'lsrun', 'aprun'],
                        help='change MPI run wrapper command (e.g. for job schedulers)')

    parser.add_argument('--mpirun_opts',
                        default=[],
                        action='append',
                        help='add options to mpirun command')

    args = parser.parse_args()

    try:
        main(**vars(args))
    except Exception:
        raise
