#!/usr/bin/env python3
# =========================================================================================
# (C) (or copyright) 2020-2021. Triad National Security, LLC. All rights reserved.
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

from .input_generator import InputFile, Block


def mpi_print(*args, **kwargs):
    """Print only from MPI rank 0.

    Behaves exactly like built-in print(), but only outputs from rank 0.
    Useful in Python input files to avoid duplicated output.

    Example:
        from parthenon_tools import mpi_print
        mpi_print(f"Configured {ndim}D problem with resolution {nx}")
    """
    try:
        import parthenon

        if parthenon.my_rank == 0:
            print(*args, **kwargs)
    except (ImportError, AttributeError):
        # If parthenon module not available or my_rank not set, just print
        # (e.g., when running outside of embedded context)
        print(*args, **kwargs)


__all__ = ["InputFile", "Block", "mpi_print"]
