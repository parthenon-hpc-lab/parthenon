#!/usr/bin/env python
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

"""
Regression test script.

Usage: From this directory, call this script with python:
      python run_test.py

Notes:
  - Requires Python 2.7+. (compliant with Python 3)
  - This file should not be modified when adding new scripts.
  - To add a new script, create a new .py file in scripts/tests/ subdirectory.
  - See scripts/tests/example.py for an example.
    - Example can be forced to run, but does not run by default in full test.
  - For more information, check online regression test documentation.
"""

# Python modules
from __future__ import print_function
import argparse
import os
from shutil import rmtree
#from collections import OrderedDict
from pkgutil import iter_modules
from timeit import default_timer as timer

# Prevent generation of .pyc files
# This should be set before importing any user modules
import sys
sys.dont_write_bytecode = True

# Parthenon modules
import utils.parthenon as parthenon  # noqa


# Main function
def main(**kwargs):

    # Make list of tests to run
    test_dir = kwargs.pop('test')
    parthenon_driver = kwargs.pop('driver')
    parthenon_driver_input = kwargs.pop('driver_input')
    
    starting_dir = os.getcwd()

    if not os.path.isdir(test_dir[0]) :
        raise TestError("Missing regression test folder " + test_dir[0] + "." )

    abs_path_test_dir = os.path.abspath(test_dir[0])
    test = os.path.basename(os.path.normpath(abs_path_test_dir))

    if not os.path.isfile(test_dir[0] +  "/" + test + ".py") :
        raise TestError("Missing regression test python file " + test_dir[0] +"/" + test + ".py ." )

    if not os.path.isfile(parthenon_driver[0]):
        raise TestError("Unable to locate driver "+parthenon_driver[0])

    abs_path_driver = os.path.abspath(parthenon_driver[0])
    abs_path_driver_input = os.path.abspath(parthenon_driver_input[0])
    
    test_times = []
    test_errors = []
    test_result = 1
    
    t0 = timer()
    try:
        test_module = 'test_suites.' + test + '.' + test

        module = __import__(test_module, globals(), locals(),
                fromlist=['prepare', 'run', 'analyze'])


        # Move to regression test folder and run test there 
        os.chdir(abs_path_test_dir)
        # Check if output folder exists if it does delete it and create a fresh folder
        CleanOutputFolder()

        try:
            module.prepare(**kwargs)
        except Exception:
            raise TestError(test_module.replace('.','/')+'.py')

        try:
            run_ret = module.run(abs_path_driver,abs_path_driver_input)
        except Exception:
            raise TestError(test_module.replace('.','/')+'.py') 

        print("Run success")
        try:
            print("Running Analyze")
            test_result = module.analyze()
        except: 
            raise TestError(test_module.replace('.','/')+'.py')

        os.chdir(starting_dir)
    except TestError as err:
        test_times.append(None)
    else:
        test_times.append(timer() - t0)
        msg = 'Test {0} took {1:.3g} seconds to complete.'
        msg = msg.format(test, test_times[-1])
        test_errors.append(None)

    if (test_result == True):
        return 0
    else:
        raise TestError("Test " + test + " failed")


def CleanOutputFolder():
    if os.path.isdir("output"):
            rmtree("output")

    os.mkdir("output")
    os.chdir("output")


# Exception for unexpected behavior by individual tests
class TestError(RuntimeError):
    pass

# Execute main function
if __name__ == '__main__':
    help_msg = ('name of test to run, relative to modules/,'
                'excluding .py')
    parser = argparse.ArgumentParser()
    parser.add_argument('--test','-t',
                        type=str,
                        default=None,
                        nargs=1,
                        help=help_msg)

    parser.add_argument("--driver", "-dr",
                        type=str,
                        default=None,
                        nargs=1,
                        help='name of driver to test')

    parser.add_argument("--driver_input", "-drin",
                        type=str,
                        default=None,
                        nargs=1,
                        help='input file to pass to driver')

    args = parser.parse_args()

    try:
        main(**vars(args))
    except Exception:
        raise
