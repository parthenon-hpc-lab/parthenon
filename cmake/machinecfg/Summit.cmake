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

message(STATUS "Loading machine configuration for OLCF's Summit.\n"
  "Supported MACHINE_VARIANT includes 'cuda', 'mpi', and 'cuda-mpi'\n"
  "This configuration has been tested using the following modules: \n"
  "  $ module load cuda/11.5.2 gcc cmake python hdf5\n"
  "Last tested: 2022-10-07\n\n")

# common options
set(Kokkos_ARCH_POWER9 ON CACHE BOOL "CPU architecture")
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Default release build")
set(MACHINE_VARIANT "cuda-mpi" CACHE STRING "Default build for CUDA and MPI")

# variants
if (${MACHINE_VARIANT} MATCHES "cuda")
  set(Kokkos_ARCH_VOLTA70 ON CACHE BOOL "GPU architecture")
  set(Kokkos_ENABLE_CUDA ON CACHE BOOL "Enable Cuda")
  set(CMAKE_CXX_COMPILER ${CMAKE_CURRENT_SOURCE_DIR}/external/Kokkos/bin/nvcc_wrapper CACHE STRING "Use nvcc_wrapper")
else()
  set(CMAKE_CXX_COMPILER g++ CACHE STRING "Use g++")
  set(CMAKE_CXX_FLAGS "-fopenmp-simd -ffast-math -fprefetch-loop-arrays" CACHE STRING "Default opt flags")
endif()

# Setting launcher options independent of parallel or serial test as the launcher always
# needs to be called from the batch node (so that the tests are actually run on the
# compute nodes.
set(TEST_MPIEXEC jsrun CACHE STRING "Command to launch MPI applications")
set(TEST_NUMPROC_FLAG "-a" CACHE STRING "Flag to set number of processes")
set(NUM_GPU_DEVICES_PER_NODE "6" CACHE STRING "6x V100 per node")
set(PARTHENON_ENABLE_GPU_MPI_CHECKS OFF CACHE BOOL "Disable check by default")

if (${MACHINE_VARIANT} MATCHES "mpi")
  # Use a single resource set on a node that includes all cores and GPUs.
  # GPUs are automatically assigned round robin when run with more than one rank.
  list(APPEND TEST_MPIOPTS "-n" "1" "-g" "6" "-c" "42" "-r" "1" "-d" "packed" "-b" "packed:7" "--smpiargs='-gpu'")
else()
  set(PARTHENON_DISABLE_MPI ON CACHE BOOL "Disable MPI")
endif()
