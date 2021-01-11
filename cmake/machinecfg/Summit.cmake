#========================================================================================
# Parthenon performance portable AMR framework
# Copyright(C) 2021 The Parthenon collaboration
# Licensed under the 3-clause BSD License, see LICENSE file for details
#========================================================================================
# (C) (or copyright) 2021. Triad National Security, LLC. All rights reserved.
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

message(STATUS "Loading machine configuration for OLCF's Summit.\n"
  "Supported MACHINE_VARIANT includes 'cuda', 'mpi', and 'cuda-mpi'\n"
  "This configuration has been tested using the following modules: "
  "module load cuda gcc cmake/3.14.2 python hdf5\n")

# common options
set(Kokkos_ARCH_POWER9 ON CACHE BOOL "CPU architecture")

# variants
if (${MACHINE_VARIANT} MATCHES "cuda")
  set(Kokkos_ARCH_VOLTA70 ON CACHE BOOL "GPU architecture")
  set(Kokkos_ENABLE_CUDA ON CACHE BOOL "Enable Cuda")
  set(CMAKE_CXX_COMPILER ${CMAKE_CURRENT_SOURCE_DIR}/external/Kokkos/bin/nvcc_wrapper CACHE STRING "Use nvcc_wrapper")
endif()

if (${MACHINE_VARIANT} MATCHES "mpi")
  set(TEST_MPIEXEC jsrun CACHE STRING "Command to launch MPI applications")
  set(TEST_NUMPROC_FLAG "-a" CACHE STRING "Flag to set number of processes")
  list(APPEND TEST_MPIOPTS "-n" "1" "-g" "6" "-c" "42" "-r" "1" "-d" "packed" "-b" "packed:7" "--smpiargs='-gpu'")
endif()
