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

# OPTIONS:
# - `RZANSEL_VIEW_DATE` - The date the dependencies were installed.
#       Default:
#           unset, which results in using the view associated with your
#           current commit. See `RZANSEL_VIEW_DATE_LATEST`
# - `RZANSEL_COMPILER` - Compiler family to use
#       Default: "GCC"
#       Possible Values: "GCC", "GCC8"
# - `RZANSEL_CUDA` - Build for CUDA
#       Default: ON 
# - `RZANSEL_PROJECT_PREFIX`
#   Description: [ADVANCED] Point to an alternative parthenon-project path
#       Default: /usr/gapps/parthenon_shared/parthenon-project

cmake_minimum_required(VERSION 3.15)

message(STATUS "Loading machine configuration for LLNL RZAnsel.\n"
  "Supported MACHINE_VARIANT includes 'cuda', 'mpi', and 'cuda-mpi'\n")

set(RZANSEL_ARCH "ppc64le")
# NOTE: When updating dependencies with new compilers or packages, you should
# ideally only need to update these variables to change the default behavior.
set(RZANSEL_VIEW_DATE_LATEST "2021-01-04")
set(RZANSEL_GCC_PREFERRED "GCC8")
set(RZANSEL_COMPILER_PREFERRED "GCC")
set(RZANSEL_GCC8_VERSION "8.3.1")
set(RZANSEL_VARIANT_PREFIX "_no-mpi")
set(RZANSEL_MPI_DATE "2020.08.19")
set(MPIEXEC_EXECUTABLE "/usr/tcetmp/bin/jsrun" CACHE STRING "Mpi executable to use on RZansel system")
set(RZANSEL_CUDA_VERSION "10.1.243")
set(MACHINE_VARIANT "cuda-mpi" CACHE STRING "Machine variant to use when building on RZAnsel")
set(NUM_RANKS 4) 
set(GPU_COUNT "4")

if (${MACHINE_VARIANT} MATCHES "cuda")
  set(RZANSEL_CUDA_DEFAULT ON)
else()
  set(RZANSEL_CUDA_DEFAULT OFF)
endif()

set(RZANSEL_CUDA ${RZANSEL_CUDA_DEFAULT} CACHE BOOL "Build for CUDA")

# It would be nice if we could let this variable float with the current code
# checkout, but unfortunately CMake caches enough other stuff (like find
# packages), that it's easier to just issue a warning if the view date is
# different from the "latest" date.
set(RZANSEL_VIEW_DATE ${RZANSEL_VIEW_DATE_LATEST}
  CACHE STRING "RZAnsel dependency view being used")
if (NOT RZANSEL_VIEW_DATE_LATEST STREQUAL RZANSEL_VIEW_DATE)
    message(WARNING "Your current build directory was configured with a \
    set of RZAnsel dependencies from ${RZANSEL_VIEW_DATE}, but your current \
    code checkout prefers a set of RZAnsel dependencies from \
        ${RZANSEL_VIEW_DATE_LATEST}. Consider configuring a new build \
        directory.")
endif()

if (NOT DEFINED RZANSEL_COMPILER)
    set(RZANSEL_COMPILER ${RZANSEL_COMPILER_PREFERRED})
endif()

if (RZANSEL_COMPILER STREQUAL "GCC")
    set(RZANSEL_COMPILER ${RZANSEL_GCC_PREFERRED})
endif()

set(RZANSEL_PROJECT_PREFIX /usr/gapps/parthenon_shared/parthenon-project
    CACHE STRING "Path to parthenon-project checkout")

mark_as_advanced(RZANSEL_PROJECT_PREFIX)

message(STATUS "RZAansel Build Settings
              RZANSEL_ARCH: ${RZANSEL_ARCH}
         RZANSEL_VIEW_DATE: ${RZANSEL_VIEW_DATE}
          RZANSEL_COMPILER: ${RZANSEL_COMPILER}
              RZANSEL_CUDA: ${RZANSEL_CUDA}
    RZANSEL_PROJECT_PREFIX: ${RZANSEL_PROJECT_PREFIX}
                 GPU_COUNT: ${GPU_COUNT}
           MACHINE_VARIANT: ${MACHINE_VARIANT}
")

set(RZANSEL_ARCH_PREFIX ${RZANSEL_PROJECT_PREFIX}/views/rzansel/${RZANSEL_ARCH})

if (NOT EXISTS ${RZANSEL_ARCH_PREFIX})
    message(WARNING "No dependencies detected for \
        RZANSEL_ARCH=\"${RZANSEL_ARCH}\" at ${RZANSEL_ARCH_PREFIX}")
    return()
endif()

if (RZANSEL_CUDA)
    # Location of CUDA
    set(CUDA_ROOT /usr/tce/packages/cuda/cuda-${RZANSEL_CUDA_VERSION} 
      CACHE STRING "CUDA Location")

    # This code ensures that the CUDA build uses the correct nvcc, and
    # that we don't have to depend on the user's environment.

    # This only holds for the length of the cmake process, but is necessary
    # for making sure the compiler checks pass
    set(ENV{CUDA_ROOT} "${CUDA_ROOT}")

    # nvcc_wrapper must be the CXX compiler for CUDA builds. Ideally this would
    # go in "CMAKE_CXX_COMPILER_LAUNCHER", but that interferes with Kokkos'
    # compiler detection.
    
    # Resolve path and set it as the CMAKE_CXX_COMPILER
    get_filename_component(
        NVCC_WRAPPER
        ${CMAKE_CURRENT_LIST_DIR}/../../external/Kokkos/bin/nvcc_wrapper
        ABSOLUTE)
    set(CMAKE_CXX_COMPILER ${NVCC_WRAPPER} CACHE STRING "nvcc_wrapper")
endif()

string(TOLOWER "${RZANSEL_COMPILER}" RZANSEL_COMPILER_LOWER)
set(RZANSEL_COMPILER_PREFIX ${RZANSEL_ARCH_PREFIX}/${RZANSEL_COMPILER_LOWER})

if (NOT EXISTS ${RZANSEL_COMPILER_PREFIX})
    message(WARNING "No dependencies detected for \
        RZANSEL_COMPILER=\"${RZANSEL_COMPILER}\" at ${RZANSEL_COMPILER_PREFIX}")
    return()
endif()

# Let the user specify the compiler if they really want to. Otherwise, point
# to the compilers specified by the RZANSEL_ options
if (RZANSEL_COMPILER MATCHES "GCC")
    set(GCC_VERSION ${RZANSEL_${RZANSEL_COMPILER}_VERSION})
    set(GCC_PREFIX /usr/tce/packages/gcc/gcc-${GCC_VERSION})

    if (GCC_VERSION)
        set(CMAKE_C_COMPILER ${GCC_PREFIX}/bin/gcc
            CACHE STRING "gcc ${GCC_VERSION}")

        set(RZANSEL_CXX_COMPILER ${GCC_PREFIX}/bin/g++)
        if (RZANSEL_CUDA)
            set(CMAKE_CXX_FLAGS "-ccbin ${RZANSEL_CXX_COMPILER}")
        else()
            set(CMAKE_CXX_COMPILER ${RZANSEL_CXX_COMPILER}
            CACHE STRING "gcc ${GCC_VERSION}")
        endif()

        set(CMAKE_BUILD_RPATH ${GCC_PREFIX}/lib64
            CACHE STRING "rpath libs")
    endif()
endif()

if (NOT DEFINED CMAKE_CXX_COMPILER)
    message(
        FATAL_ERROR
        "Found view on RZAnsel for compiler version ${RZANSEL_COMPILER}, but \
        don't know how to map it to a specific compiler. Either update \
        your Parthenon checkout or explicitly set CMAKE_C_COMPILER and \
        CMAKE_CXX_COMPILER")
endif()

# MPI - We use the system modules since replicating them in spack can be
# difficult.
if (RZANSEL_COMPILER MATCHES "GCC")
    set(MPI_ROOT /usr/tce/packages/spectrum-mpi/spectrum-mpi-${RZANSEL_MPI_DATE}-gcc-${RZANSEL_${RZANSEL_COMPILER}_VERSION}
        CACHE STRING "MPI Location")
endif()

# clang-format
set(CLANG_FORMAT
    ${RZANSEL_PROJECT_PREFIX}/tools/${RZANSEL_ARCH}/bin/clang-format-8
    CACHE STRING "clang-format-8")

# Kokkos settings
set(Kokkos_ARCH_POWER9 ON CACHE BOOL "Target Power9")

if (RZANSEL_CUDA)
    set(Kokkos_ENABLE_CUDA ON CACHE BOOL "Cuda")
    set(Kokkos_ARCH_VOLTA70 ON CACHE BOOL "Target V100s")
    set(NUM_GPU_DEVICES_PER_NODE ${NUM_RANKS} CACHE STRING "Number of gpu devices to use when testing if built with Kokkos_ENABLE_CUDA")
endif()

if (${MACHINE_VARIANT} MATCHES "mpi")
  set(RZANSEL_VARIANT_PREFIX "")
  set(NUM_MPI_PROC_TESTING ${NUM_RANKS} CACHE STRING "CI runs tests with 2 MPI ranks by default.")
  set(MPIEXEC_EXECUTABLE "/usr/tcetmp/bin/jsrun" CACHE STRING "Command to launch MPI applications")
  set(TEST_NUMPROC_FLAG "-a" CACHE STRING "Flag to set number of processes")
  if (${MACHINE_VARIANT} MATCHES "cuda")
    string(APPEND TEST_MPIOPTS "-c 1 -n 1 -g ${NUM_GPU_DEVICES_PER_NODE} -r 1 -d packed --smpiargs='-gpu'")
  else()
    string(APPEND TEST_MPIOPTS "-c 1 -n 1 -r 1 -d packed")
  endif()
else()
  set(PARTHENON_DISABLE_MPI ON CACHE STRING "MPI is enabled by default if found, set this to True to disable MPI" FORCE)
  set(PARTHENON_ENABLE_GPU_MPI_CHECKS OFF CACHE STRING "Checks if possible that the mpi num of procs and the number of gpu devices detected are appropriate.")
endif()

set(RZANSEL_VIEW_PREFIX ${RZANSEL_COMPILER_PREFIX}${RZANSEL_VARIANT_PREFIX}/${RZANSEL_VIEW_DATE})
if (NOT EXISTS ${RZANSEL_VIEW_PREFIX})
    message(WARNING "No view detected for \
        RZANSEL_VIEW_DATE=\"${RZANSEL_VIEW_DATE}\" at ${RZANSEL_VIEW_PREFIX}")
    return()
endif()

# Add dependencies into `CMAKE_PREFIX_PATH`
list(PREPEND CMAKE_PREFIX_PATH ${RZANSEL_VIEW_PREFIX})
