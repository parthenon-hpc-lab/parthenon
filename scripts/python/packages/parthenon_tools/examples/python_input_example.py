#!/usr/bin/env python3
# =========================================================================================
# (C) (or copyright) 2020-2025. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001 for Los
# Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
# for the U.S. Department of Energy/National Nuclear Security Administration. All rights
# in the program are reserved by Triad National Security, LLC, and the U.S. Department
# of Energy/National Nuclear Security Administration. The Government is granted for
# itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
# license in this material to reproduce, prepare derivative works, distribute copies to
# the public, perform publicly and display publicly, and to permit others to do so.
# =========================================================================================

"""
Example demonstrating Python-based typed parameter input.

This shows how to build parameter inputs in Python with full mutability,
then transfer to C++ ParameterInput with type preservation (no string parsing).

To use this, Parthenon must be built with -DPARTHENON_ENABLE_PYTHON_BINDINGS=ON
"""

from parthenon_tools import InputFile

# Stage 1: Build mutable parameter structure in Python
print("Building parameter structure...")
inp = InputFile()

# Add blocks with initial parameters
mesh = inp.block(
    "parthenon/mesh",
    nx1=64,
    nx2=64,
    nx3=64,
    x1min=0.0,
    x1max=1.0,
    x2min=0.0,
    x2max=1.0,
    x3min=0.0,
    x3max=1.0,
)

# Can modify parameters after creation
print("Modifying mesh parameters...")
mesh.params["nx1"] = 128  # Change resolution
mesh.params["nx2"] = 128
mesh.params["nx3"] = 128

time = inp.block("parthenon/time", tlim=1.0, nlim=100, integrator="rk2")

# Multiple output blocks
output1 = inp.block("parthenon/output1", file_type="hdf5", dt=0.1, variables=["cons"])

output2 = inp.block("parthenon/output2", file_type="hist", dt=0.01)

# Problem-specific parameters with various types
problem = inp.block(
    "problem",
    velocity=[1.0, 0.5, 0.0],  # vector
    periodic=True,  # bool
    num_modes=3,  # int
    amplitude=0.01,
)  # float

print("\nParameter structure built:")
for block in inp.blocks:
    print(f"  Block: {block.name}")
    for key, value in block.params.items():
        print(f"    {key} = {value} (type: {type(value).__name__})")

# Stage 2: Transfer to typed C++ ParameterInput
print("\nTransferring to C++ ParameterInput...")
try:
    pi = inp.to_parameter_input()
    print("Success! Parameters transferred with full type preservation.")

    # Can query parameters from C++ side
    print(f"\nVerifying parameters:")
    print(f"  mesh/nx1 = {pi.get_int('parthenon/mesh', 'nx1')}")
    print(f"  mesh/x1min = {pi.get_real('parthenon/mesh', 'x1min')}")
    print(f"  problem/periodic = {pi.get_bool('problem', 'periodic')}")
    print(f"  problem/velocity = {pi.get_real_vector('problem', 'velocity')}")

except ImportError as e:
    print(f"\nWarning: {e}")
    print("\nPython bindings not available. To enable:")
    print("  1. Rebuild Parthenon with -DPARTHENON_ENABLE_PYTHON_BINDINGS=ON")
    print("  2. Add build directory to PYTHONPATH")
    print("\nFalling back to text output for debugging:")
    print(inp)

    # Can still write text file for manual use
    inp.write("example_fallback.pin")
    print("Written to example_fallback.pin")
