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

message(STATUS "Loading machine configuration for default CI machine. "
  "Supported MACHINE_VARIANT includes 'cuda', 'mpi', and 'cuda-mpi'")

# common options
set(Kokkos_ARCH_ZEN2 ON CACHE BOOL "CPU architecture")
set(NUM_MPI_PROC_TESTING "4" CACHE STRING "CI runs tests with 4 MPI ranks")
# variants
if (${MACHINE_VARIANT} MATCHES "cuda")
  set(Kokkos_ARCH_AMPERE80 ON CACHE BOOL "GPU architecture")
  set(Kokkos_ENABLE_CUDA ON CACHE BOOL "Enable Cuda")
  set(CMAKE_CXX_COMPILER ${CMAKE_CURRENT_SOURCE_DIR}/external/Kokkos/bin/nvcc_wrapper CACHE STRING "Use nvcc_wrapper")
else()
  set(CMAKE_CXX_FLAGS "-fopenmp-simd" CACHE STRING "Default opt flags")
endif()

if (${MACHINE_VARIANT} MATCHES "mpi")
  # not using the following as the default is determined correctly
  #set(TEST_MPIEXEC mpiexec CACHE STRING "Command to launch MPI applications")
  set(HDF5_ROOT /usr/local/hdf5/parallel CACHE STRING "HDF5 path")
else()
  set(HDF5_ROOT /usr/local/hdf5/serial CACHE STRING "HDF5 path")
  set(PARTHENON_DISABLE_MPI ON CACHE BOOL "Disable MPI")
endif()
