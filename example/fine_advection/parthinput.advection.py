#!/usr/bin/env python3
# ========================================================================================
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
# ========================================================================================
# This file was made in part with generative AI.

# Example Python input file for fine_advection example
# Demonstrates advantages over text input files:
#  - Single file works for 1D, 2D, or 3D (just change ndim)
#  - Command line arguments for easy parameter studies
#  - Use variables and calculations
#  - Compute derived quantities automatically
#  - Cleaner than duplicating parameters across dimensions

import argparse
from parthenon_input import InputFile, mpi_print


def parthenon_init_parameters(pin):
    """Configure parameters for fine advection example.

    Args:
        pin: ParameterInput object to configure
    """
    parser = argparse.ArgumentParser(description="Fine advection example")
    parser.add_argument(
        "--ndim", type=int, default=2, help="Number of dimensions (1, 2, or 3)"
    )
    parser.add_argument(
        "--nx", type=int, default=64, help="Base mesh resolution (in active dimensions)"
    )
    parser.add_argument(
        "--meshblock-size",
        type=int,
        default=16,
        help="Meshblock size (in active dimensions)",
    )
    parser.add_argument("--num-levels", type=int, default=3, help="Number of AMR levels")
    parser.add_argument("--cfl", type=float, default=0.45, help="CFL number")
    args, unknown = parser.parse_known_args()
    
    # ======================================================================================
    # PROBLEM CONFIGURATION
    # ======================================================================================
    ndim = args.ndim
    nx_base = args.nx
    meshblock_size = args.meshblock_size
    num_amr_levels = args.num_levels
    cfl = args.cfl
    
    # Fixed parameters
    domain_min = -0.5  # Domain bounds
    domain_max = 0.5
    velocity = 1.0  # Advection velocity (in active dimensions)
    refine_tol = 0.3  # AMR refinement tolerance
    output_dt = 0.05  # Output cadence
    
    # ======================================================================================
    # BUILD CONFIGURATION (automatic based on ndim)
    # ======================================================================================
    inp = InputFile()
    
    inp.block("parthenon/job", problem_id="advection")
    
    # Mesh configuration - automatically set dimensions based on ndim
    inp.block(
        "parthenon/mesh",
        refinement="adaptive",
        numlevel=num_amr_levels,
        # Dimension 1 (always active)
        nx1=nx_base,
        x1min=domain_min,
        x1max=domain_max,
        ix1_bc="periodic",
        ox1_bc="periodic",
        # Dimension 2 (active if ndim >= 2)
        nx2=nx_base if ndim >= 2 else 1,
        x2min=domain_min,
        x2max=domain_max,
        ix2_bc="periodic",
        ox2_bc="periodic",
        # Dimension 3 (active if ndim >= 3)
        nx3=nx_base if ndim >= 3 else 1,
        x3min=domain_min,
        x3max=domain_max,
        ix3_bc="periodic",
        ox3_bc="periodic",
    )
    
    # Meshblock configuration - automatically sized based on ndim
    inp.block(
        "parthenon/meshblock",
        nx1=meshblock_size,
        nx2=meshblock_size if ndim >= 2 else 1,
        nx3=meshblock_size if ndim >= 3 else 1,
    )
    
    inp.block("parthenon/time", nlim=-1, tlim=1.0, integrator="rk2", ncycle_out_mesh=-10000)
    
    # Advection parameters - velocities set based on ndim
    inp.block(
        "Advection",
        cfl=cfl,
        vx=velocity,
        vy=velocity if ndim >= 2 else 0.0,
        vz=velocity if ndim >= 3 else 0.0,
        profile="hard_sphere",
        # Automatically compute derefine tolerance
        refine_tol=refine_tol,
        derefine_tol=refine_tol / 10.0,
        # Feature flags
        do_regular_advection=True,
        do_fine_advection=True,
        do_CT_advection=True,
    )
    
    # Restart output
    inp.block("parthenon/output1", file_type="rst", dt=output_dt)
    
    # HDF5 output
    inp.block(
        "parthenon/output0",
        file_type="hdf5",
        dt=output_dt,
        variables=["advection.scalar", "advection.scalar_fine_restricted"],
    )
    
    # Transfer all configuration to C++ ParameterInput
    inp.to_parameter_input(pin)
    
    # Print summary only from rank 0 using mpi_print helper
    mpi_print(f"Configured {ndim}D advection problem:")
    mpi_print(f"  Resolution: {nx_base}^{ndim}")
    mpi_print(f"  Meshblock size: {meshblock_size}")
    mpi_print(f"  AMR levels: {num_amr_levels}")
    mpi_print(f"  CFL: {cfl}")
