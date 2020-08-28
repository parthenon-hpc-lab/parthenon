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

message(STATUS "Loading machine configuration for default CI machine."
  "Supported MACHINE_VARIANT includes 'cuda', 'mpi', and 'cuda-mpi'")

# common options
set(Kokkos_ARCH_WSM ON CACHE BOOL "CPU architecture")
set(HDF5_ROOT /usr/local/hdf5/parallel CACHE STRING "HDF5 path")

# variants
if (${MACHINE_VARIANT} MATCHES "cuda")
  set(Kokkos_ARCH_PASCAL61 ON CACHE BOOL "GPU architecture")
  set(Kokkos_ENABLE_CUDA ON CACHE BOOL "Enable Cuda")
  set(CMAKE_CXX_COMPILER ${CMAKE_CURRENT_SOURCE_DIR}/external/Kokkos/bin/nvcc_wrapper CACHE STRING "Use nvcc_wrapper")
endif()

#set(MACHINE_MPIEXEC srun CACHE STRING "Command to launch MPI applications")
