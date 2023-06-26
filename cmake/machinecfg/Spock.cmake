#========================================================================================
# Parthenon performance portable AMR framework
# Copyright(C) 2022 The Parthenon collaboration
# Licensed under the 3-clause BSD License, see LICENSE file for details
#========================================================================================
# (C) (or copyright) 2022. Triad National Security, LLC. All rights reserved.
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

message(STATUS "Loading machine configuration for OLCF's Spock.\n"
  "Supported MACHINE_VARIANT includes 'hip', 'mpi', and 'hip-mpi'\n"
  "This configuration has been tested (on 2022-03-24) using the following modules: \n"
  "module load PrgEnv-amd craype-accel-amd-gfx908 cmake hdf5 cray-python\n"
  "and environment variables:\n"
  "export MPIR_CVAR_GPU_EAGER_DEVICE_MEM=0\n"
  "export MPICH_GPU_SUPPORT_ENABLED=1\n"
  "export MPICH_SMP_SINGLE_COPY_MODE=CMA\n")

# common options
set(Kokkos_ARCH_ZEN2 ON CACHE BOOL "CPU architecture")
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Default release build")
set(MACHINE_VARIANT "hip-mpi" CACHE STRING "Default build for CUDA and MPI")

# variants
set(MACHINE_CXX_FLAGS "")
if (${MACHINE_VARIANT} MATCHES "hip")
  set(Kokkos_ARCH_VEGA908 ON CACHE BOOL "GPU architecture")
  set(Kokkos_ENABLE_HIP ON CACHE BOOL "Enable HIP")
  set(CMAKE_CXX_COMPILER hipcc CACHE STRING "Use hip wrapper")
else()
  set(CMAKE_CXX_COMPILER $ENV{ROCM_PATH}/llvm/bin/clang++ CACHE STRING "Use g++")
  set(MACHINE_CXX_FLAGS "${MACHINE_CXX_FLAGS} -fopenmp-simd -fprefetch-loop-arrays")
endif()

# Setting launcher options independent of parallel or serial test as the launcher always
# needs to be called from the batch node (so that the tests are actually run on the
# compute nodes.
set(TEST_MPIEXEC srun CACHE STRING "Command to launch MPI applications")
set(TEST_NUMPROC_FLAG "-n" CACHE STRING "Flag to set number of processes")
set(NUM_GPU_DEVICES_PER_NODE "4" CACHE STRING "4x MI100 per node")
set(PARTHENON_ENABLE_GPU_MPI_CHECKS OFF CACHE BOOL "Disable check by default")

if (${MACHINE_VARIANT} MATCHES "mpi")
  # need to set include flags here as the target is not know yet when this file is parsed
  set(MACHINE_CXX_FLAGS "${MACHINE_CXX_FLAGS} -I$ENV{MPICH_DIR}/include")
  set(CMAKE_EXE_LINKER_FLAGS "-L$ENV{MPICH_DIR}/lib -lmpi -L$ENV{CRAY_MPICH_ROOTDIR}/gtl/lib -lmpi_gtl_hsa" CACHE STRING "Default flags for this config")
else()
  set(PARTHENON_DISABLE_MPI ON CACHE BOOL "Disable MPI")
endif()

set(CMAKE_CXX_FLAGS "${MACHINE_CXX_FLAGS}" CACHE STRING "Default flags for this config")
