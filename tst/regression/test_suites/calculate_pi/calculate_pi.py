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

# Modules
import math
import numpy as np
import sys
import os
import utils.parthenon as parthenon          

""" To prevent littering up imported folders with .pyc files or __pycache_ folder"""
sys.dont_write_bytecode = True

def run(parameters):
    """
    Run the executable.

    This function is called second. It is responsible for calling the Athena++ binary in
    such a way as to produce testable output. It takes no inputs and produces no outputs.
    """
    parthenon.run(parameters)

def analyze(parameters):
    """
    Analyze the output and determine if the test passes.

    This function is called third; nothing from this file is called after it. It is
    responsible for reading whatever data it needs and making a judgment about whether or
    not the test passes. It takes no inputs. Output should be True (test passes) or False
    (test fails).
    """
    line = ""
    try:
        f = open(os.path.join(parameters.output_path, "summary.txt"),"r")
        # Do something with the file
        line = f.readline()

        f.close()
    except IOError:
        print("Summary file not accessible")

    words = line.split()
    pi_val = float(words[2])

    error_abs_e = math.fabs( math.pi - pi_val ) / math.pi  

    analyze_status = True
    if (error_abs_e > 0.001 ) or np.isnan(error_abs_e):
        analyze_status = False
    
    return analyze_status
