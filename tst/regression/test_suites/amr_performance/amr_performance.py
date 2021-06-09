#========================================================================================
# Parthenon performance portable AMR framework
# Copyright(C) 2021 The Parthenon collaboration
# Licensed under the 3-clause BSD License, see LICENSE file for details
#========================================================================================
# (C) (or copyright) 2021. Triad National Security, LLC. All rights reserved.
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

""" To prevent littering up imported folders with .pyc files or __pycache_ folder"""
sys.dont_write_bytecode = True


class TestCase(utils.test_case.TestCaseAbs):
    def Prepare(self, parameters, step):
        # making sure env variable is set to produce json output from kokkos space-time-stack
        os.environ['KOKKOS_PROFILE_EXPORT_JSON'] = 'ON'
        
        msg = "Missing Kokkos profiling lib.\n"
        msg += "Please set KOKKOS_PROFILE_LIBRARY to kp_space_time_stack.so"
        assert parameters.kokkos_profiling_lib is not None, msg

        # Profiling libraries introduce fences etc.
        # Thus, we run once with and once without.
        if step % 2 != 0:
            os.unsetenv('KOKKOS_PROFILE_LIBRARY')

        if step == 1 or step == 2:
            parameters.driver_cmd_line_args = [
                'Advection/num_vars=9',
                'Advection/vec_size=1'
            ]
        elif step == 3 or step == 4:
            parameters.driver_cmd_line_args = [
                'Advection/num_vars=3',
                'Advection/vec_size=3'
            ]
        elif step == 5 or step == 6:
            parameters.driver_cmd_line_args = [
                'Advection/num_vars=1',
                'Advection/vec_size=9'
            ]

        return parameters

    def Cleanup(self, parameters, step):
        # move json output to unique location for later analysis
        if step % 2 == 0:
            os.rename("noname.json", "step_%d.json" % step)

        with open("step_%d.out" % step,"wb") as outfile:
            outfile.write(parameters.stdouts[-1])
        
        # Reset to orig environment value
        if parameters.kokkos_profiling_lib is not None:
            os.environ['KOKKOS_PROFILE_LIBRARY'] = parameters.kokkos_profiling_lib



    def Analyse(self, parameters):

        fig, p = plt.subplots(4,1, sharex=True, figsize=(4,10))

        for (step, lbl) in [(1, "9 var. with 1 component "),
                            (3, "3 var. with 3 components"),
                            (5, "1 var. with 9 components"),
                           ]:
            cycle_all = []
            for line in parameters.stdouts[step - 1].decode("utf-8").split('\n'):
                # sample output:
                # cycle=3 time=2.6367187499999997e-03 dt=8.7890624999999991e-04 zone-cycles/wsec_step=9.16e+07 wsec_total=4.00e-01 wsec_step=8.30e-02 zone-cycles/wsec=9.16e+07 wsec_AMR=3.60e-06
                if "cycle=" == line[:6]:
                    cycle_current = []
                    for vals in line.split(" "):
                        cycle_current.append(float(vals.split("=")[-1]))
                    cycle_all.append(cycle_current)

            # convert to array and skip cycle cycle 0
            cycle_all = np.array(cycle_all)[1:,:]

            label = lbl + ": total wtime %.2f" % cycle_all[-1,4]
            p[0].plot(cycle_all[:,0], cycle_all[:,3], label=label)
            p[1].plot(cycle_all[:,0], cycle_all[:,5])
            p[2].plot(cycle_all[:,0], cycle_all[:,6])
            p[3].plot(cycle_all[:,0], cycle_all[:,7])

        p[0].legend(loc="lower right", bbox_to_anchor=(1,1))
        p[0].set_ylabel("zone-cycles/wsec_step")
        p[1].set_ylabel("wsec_step")
        p[2].set_ylabel("zone-cycles/wsec")
        p[3].set_ylabel("wsec_AMR")
        p[-1].set_xlabel("cycle #")

        for i in range(4):
            p[i].grid()

        fig.tight_layout()

        fig.savefig(os.path.join(parameters.output_path, "amr_performance.png"),
                    bbox_inches='tight')

        return True
