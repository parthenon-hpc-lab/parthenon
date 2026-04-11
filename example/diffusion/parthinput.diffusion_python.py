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

Demonstrates Python field initialization with parameter-driven configuration.
"""

from parthenon_input import InputFile


def init_gaussian(x, y, z, component, data, pin):
    """Initialize field with Gaussian profile.

    Parameters are read from the input file, allowing flexible configuration
    without modifying the initialization function.

    Args:
        x, y, z: Coordinate arrays (numpy arrays, zero-copy views)
        component: tuple of component indices (empty for scalar)
        data: Data array to write (numpy array, zero-copy view)
        pin: ParameterInput object for reading parameters

    Note:
        NumPy is required for field initialization.
    """
    import numpy as np

    # Read parameters from input file
    x0 = pin.get_real("diffusion", "x0")
    y0 = pin.get_real("diffusion", "y0")
    z0 = pin.get_real("diffusion", "z0")
    t0 = pin.get_real("diffusion", "t0")
    D = pin.get_real("diffusion", "D")

    # Vectorized computation
    r2 = (x - x0)**2 + (y - y0)**2 + (z - z0)**2
    data[:] = np.exp(-r2 / (4.0 * D * t0))

    print(f"Initialized {len(data)} cells with Gaussian profile")
    print(f"  Center: ({x0}, {y0}, {z0}), t0={t0}, D={D}")
    print(f"  Value range: [{data.min():.6f}, {data.max():.6f}]")


def parthenon_init_parameters(pin):
    """Configure parameters for diffusion example.

    Args:
        pin: ParameterInput object to configure
    """
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
    inp.block("parthenon/time", tlim=0.01, nlim=100, integrator="rk1")

    # Diffusion parameters (including Gaussian profile parameters)
    inp.block("diffusion",
              dt=1.0,
              constant_coefficient=True,
              # Gaussian profile parameters for field initialization
              x0=0.0, y0=0.0, z0=0.0,  # Center of Gaussian
              t0=0.001,                  # Initial diffusion time
              D=1.0)                     # Diffusion coefficient

    # Register Python initialization function
    inp.block("diffusion/python_init",
              u_function="init_gaussian",
              u_file=__file__)

    # Output configuration
    out = inp.block("parthenon/output0")
    out.set(file_type="hdf5", dt=0.01, variables=["diffusion.u"])

    # Transfer to C++
    inp.to_parameter_input(pin)
