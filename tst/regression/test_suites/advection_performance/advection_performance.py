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
import matplotlib
matplotlib.use('agg')
import matplotlib.pylab as plt
import sys
import os
import utils.test_case

""" To prevent littering up imported folders with .pyc files or __pycache_ folder"""
sys.dont_write_bytecode = True

n_threads_arr = [1, 4, 8]
n_streams_arr = [1, 4, 8, 16]
mb_sizes = [256, 128, 64, 32] # meshblock sizes

def get_test_matrix():
    test_configs = []
    for n_threads in n_threads_arr:
        for n_streams in n_streams_arr:
            if n_streams < n_threads:
                continue
            if n_streams > 2*n_threads:
                continue
            for mb in mb_sizes:
                test_configs.append((n_threads, n_streams, mb))
    return test_configs

class TestCase(utils.test_case.TestCaseAbs):
    def Prepare(self, parameters, step):

        test_configs = get_test_matrix()
        num_threads, num_streams, mb = test_configs[step - 1]

        parameters.driver_cmd_line_args = [
            'parthenon/mesh/nx1=256',
            'parthenon/meshblock/nx1=%d' % mb,
            'parthenon/mesh/nx2=256',
            'parthenon/meshblock/nx2=%d' % mb,
            'parthenon/mesh/nx3=256',
            'parthenon/meshblock/nx3=%d' % mb,
            'parthenon/mesh/num_threads=%d' % num_threads,
            'parthenon/mesh/num_streams=%d' % num_streams,
        ]
        return parameters

    def Analyse(self, parameters):

        perfs = []
        for output in parameters.stdouts:
            this_perf = []
            for line in output.decode("utf-8").split('\n'):
                print(line)
                if 'cycle=5' in line or 'cycle=6' in line or 'cycle=7' in line or 'cycle=8' in line or 'cycle=9' in line:
                    this_perf.append(float(line.split(' ')[5]))
            perfs.append(this_perf)

        perfs = np.array(perfs)
        test_configs = get_test_matrix()

        if len(perfs) != len(test_configs):
            print("ERROR: mismatch between expected and reported performance counters")
            return False

        # Plot results
        fig, p = plt.subplots(2, 1, figsize = (4,8), sharex=True)

        num_envs = len(test_configs) // len(mb_sizes)

        for i in range(num_envs):
            num_threads, num_streams, unused = test_configs[i*len(mb_sizes)]
            label = "$256^3$ Mesh; $N_T$ = %d; $N_S$ = %d" % (num_threads, num_streams)
            perf = perfs[i*len(mb_sizes):(i+1)*len(mb_sizes)]
            p[0].errorbar(mb_sizes, np.median(perf,axis=1),
                yerr=(np.max(perf,axis=1) - np.median(perf,axis=1),
                      np.median(perf,axis=1) - np.min(perf,axis=1)), label=label)

            p[1].loglog(mb_sizes, np.median(perf,axis=1)[0]/np.median(perf,axis=1))

        for i in range(2):
            p[i].grid()
            p[i].set_xscale('log')
            p[i].set_yscale('log')
        p[0].legend(fontsize=8)
        p[0].set_ylabel("zone-cycles/s")
        p[1].set_ylabel("normalized overhead")
        p[1].set_xlabel("Meshblock size")
        fig.savefig(os.path.join(parameters.output_path, "performance.png"),
                    bbox_inches='tight')

        return True
