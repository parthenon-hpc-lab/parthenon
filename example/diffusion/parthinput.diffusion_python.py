#!/usr/bin/env python3
#========================================================================================
# (C) (or copyright) 2020-2026. Triad National Security, LLC. All rights reserved.
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
Python input file for diffusion example with Python-based field initialization.

Demonstrates using Python to initialize field values programmatically.
The initialization function is in a separate file (diffusion_init.py).
"""

import parthenon
import os

# Get ParameterInput object from C++
pi = parthenon.get_parameter_input()

# Mesh configuration
pi.add_int("parthenon/mesh", "nx1", 64)
pi.add_int("parthenon/mesh", "nx2", 64)
pi.add_int("parthenon/mesh", "nx3", 1)

pi.add_real("parthenon/mesh", "x1min", -1.0)
pi.add_real("parthenon/mesh", "x1max", 1.0)
pi.add_real("parthenon/mesh", "x2min", -1.0)
pi.add_real("parthenon/mesh", "x2max", 1.0)
pi.add_real("parthenon/mesh", "x3min", 0.0)
pi.add_real("parthenon/mesh", "x3max", 1.0)

pi.add_int("parthenon/meshblock", "nx1", 16)
pi.add_int("parthenon/meshblock", "nx2", 16)
pi.add_int("parthenon/meshblock", "nx3", 1)

# Time configuration
pi.add_real("parthenon/time", "tlim", 0.01)
pi.add_int("parthenon/time", "nlim", 100)

# Diffusion parameters
pi.add_real("diffusion", "dt", 1.0)
pi.add_bool("diffusion", "constant_coefficient", True)

# Python initialization configuration
# Point to separate init file in the same directory
init_file = os.path.join(os.path.dirname(__file__), "diffusion_init.py")
pi.add_string("diffusion/python_init", "u_function", "init_gaussian")
pi.add_string("diffusion/python_init", "u_file", init_file)

# Output
pi.add_string("parthenon/output0", "file_type", "hdf5")
pi.add_real("parthenon/output0", "dt", 0.01)
pi.add_string_vector("parthenon/output0", "variables", ["diffusion.u"])
