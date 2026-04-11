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

Demonstrates idiomatic Python input configuration with init function defined inline.
"""

from parthenon_input import InputFile

def init_gaussian(x, y, z, component, data):
    """Initialize field with Gaussian profile.

    Args:
        x, y, z: Coordinate arrays (numpy arrays if available, else lists)
        component: tuple of component indices (empty for scalar)
        data: Data array to write (numpy array if available, else list)
    """
    # Try numpy for performance (vectorized operations)
    try:
        import numpy as np
        x = np.asarray(x)
        y = np.asarray(y)
        z = np.asarray(z)

        # Gaussian profile parameters
        x0, y0, z0 = 0.0, 0.0, 0.0
        t0 = 0.001
        D = 1.0

        # Vectorized computation
        r2 = (x - x0)**2 + (y - y0)**2 + (z - z0)**2
        data[:] = np.exp(-r2 / (4.0 * D * t0))

        print(f"Initialized {len(data)} cells with Gaussian profile (numpy)")
        print(f"  Value range: [{data.min():.6f}, {data.max():.6f}]")

    except ImportError:
        # Fallback to pure Python (slower but no dependencies)
        import math
        x0, y0, z0 = 0.0, 0.0, 0.0
        t0 = 0.001
        D = 1.0

        for i in range(len(data)):
            r2 = (x[i] - x0)**2 + (y[i] - y0)**2 + (z[i] - z0)**2
            data[i] = math.exp(-r2 / (4.0 * D * t0))

        print(f"Initialized {len(data)} cells with Gaussian profile (pure Python)")
        min_val = min(data)
        max_val = max(data)
        print(f"  Value range: [{min_val:.6f}, {max_val:.6f}]")

# Create input file
inp = InputFile()

# Mesh configuration
inp.block("parthenon/mesh",
          nx1=64, nx2=64, nx3=1,
          x1min=-1.0, x1max=1.0,
          x2min=-1.0, x2max=1.0,
          x3min=0.0, x3max=1.0)

inp.block("parthenon/meshblock", nx1=16, nx2=16, nx3=1)

# Time configuration
inp.block("parthenon/time", tlim=0.01, nlim=100)

# Diffusion parameters
inp.block("diffusion", dt=1.0, constant_coefficient=True)

# Register Python initialization function
inp.block("diffusion/python_init",
          u_function="init_gaussian",
          u_file=__file__)

# Output configuration
out = inp.block("parthenon/output0")
out.set(file_type="hdf5", dt=0.01, variables=["diffusion.u"])

# Transfer to C++ (only if running as input file, not during field init)
import parthenon
try:
    pi = parthenon.get_parameter_input()
    inp.to_parameter_input(pi)
except KeyError:
    # Being loaded for field initialization, not input parsing
    # The init_gaussian function is already defined above
    pass
