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
#import logging
import os
import subprocess
from timeit import default_timer as timer

# Functions for running Parthenon
def run(driver, input_filename, arguments, lcov_test_suffix=None):
    
    #try:
    run_command = [driver, '-i', input_filename]
    try:
        cmd = run_command #+ arguments + global_run_args
        print("run cmd is")
        print(cmd)
        print("current working dir")
        print(os.getcwd())
        #logging.getLogger('athena.run').debug('Executing: ' + ' '.join(cmd))
        #subprocess.check_call(cmd, stdout=out_log)
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as err:
        raise ParthenonError('Return code {0} from command \'{1}\''
                          .format(err.returncode, ' '.join(err.cmd)))
        #else:
            #os.chdir(current_dir)
            # (optional) if execution completes without error, and a lcov_test_suffix is
            # explicitly passed, process Lcov tracefile immediately after run_command
            #analyze_code_coverage(global_test_name, lcov_test_suffix)
    #finally:
        #out_log.close()
        #os.chdir(current_dir)
#
#
#def mpirun(mpirun_cmd, mpirun_opts, nproc, input_filename, arguments,
#           lcov_test_suffix=None):
#    current_dir = os.getcwd()
#    os.chdir('bin')
#    out_log = LogPipe('athena.run', logging.INFO)
#    try:
#        input_filename_full = '../' + athena_rel_path + 'inputs/' + \
#                              input_filename
#        run_command = [mpirun_cmd] + mpirun_opts + ['-n', str(nproc), './athena', '-i',
#                                                    input_filename_full]
#        run_command = list(filter(None, run_command))  # remove any empty strings
#        try:
#            cmd = run_command + arguments + global_run_args
#            logging.getLogger('athena.run').debug('Executing (mpirun): ' + ' '.join(cmd))
#            subprocess.check_call(cmd, stdout=out_log)
#        except subprocess.CalledProcessError as err:
#            raise AthenaError('Return code {0} from command \'{1}\''
#                              .format(err.returncode, ' '.join(err.cmd)))
#        else:
#            os.chdir(current_dir)
#            # (optional) if execution completes without error, and a lcov_test_suffix is
#            # explicitly passed, process Lcov tracefile immediately after run_command
#            analyze_code_coverage(global_test_name, lcov_test_suffix)
#    finally:
#        out_log.close()
#        os.chdir(current_dir)
#


# General exception class for these functions
class ParthenonError(RuntimeError):
    pass
