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
Python field initialization functions for diffusion example.
"""

import math

def init_gaussian(x, y, z, component, data):
    """Initialize field with Gaussian profile.

    Args:
        x, y, z: 1D arrays of coordinates (flattened)
        component: tuple of component indices (empty for scalar)
        data: 1D array to write to (flattened, same length as x/y/z)
    """
    # Center of Gaussian
    x0, y0, z0 = 0.0, 0.0, 0.0

    # Initial time for diffusion kernel
    t0 = 0.001
    D = 1.0  # diffusion coefficient

    # Initialize each cell
    for i in range(len(data)):
        r2 = (x[i] - x0)**2 + (y[i] - y0)**2 + (z[i] - z0)**2
        exponent = -r2 / (4.0 * D * t0)
        data[i] = math.exp(exponent)

    print(f"Initialized {len(data)} cells with Gaussian profile")
    min_val = min(data)
    max_val = max(data)
    print(f"  Value range: [{min_val:.6f}, {max_val:.6f}]")
