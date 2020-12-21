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
#       Possible Values: "GCC", "GCC9"
# - `RZANSEL_CUDA` - Build for CUDA
#       Default: ON if nvidia-smi finds at least one GPU, OFF otherwise
# - `RZANSEL_PROJECT_PREFIX`
#   Description: [ADVANCED] Point to an alternative parthenon-project path
#       Default: /usr/gapps/parthenon_shared/parthenon-project

cmake_minimum_required(VERSION 3.15)
set(RZANSEL_ARCH "ppc64le")
# NOTE: When updating dependencies with new compilers or packages, you should
# ideally only need to update these variables to change the default behavior.

set(RZANSEL_VIEW_DATE_LATEST "2020-12-17")
set(RZANSEL_GCC_PREFERRED "GCC8")
set(RZANSEL_COMPILER_PREFERRED "GCC")
set(RZANSEL_GCC8_VERSION "8.3.1")

set(RZANSEL_MPI_DATE "2020.08.19")
set(MPIEXEC_EXECUTABLE "/usr/tcetmp/bin/jsrun" CACHE STRING "Mpi executable to use on RZansel system")
set(RZANSEL_CUDA_VERSION "10.1.243")

set(RZANSEL_CUDA_DEFAULT ON)

set(GPU_COUNT "4")
execute_process(
    COMMAND nvidia-smi -L
    OUTPUT_VARIABLE FOUND_GPUS)
string(REPLACE "\n" ";" FOUND_GPUS ${FOUND_GPUS})

list(FILTER FOUND_GPUS INCLUDE REGEX "GPU [0-9]")
list(LENGTH FOUND_GPUS GPU_COUNT)

if (GPU_COUNT EQUAL 0)
    set(RZANSEL_CUDA_DEFAULT OFF)
else()
    set(RZANSEL_CUDA_DEFAULT ON)
endif()

set(RZANSEL_CUDA ${RZANSEL_CUDA_DEFAULT} CACHE BOOL "Build for CUDA")

# It would be nice if we could let this variable float with the current code
# checkout, but unfortunately CMake caches enough other stuff (like find
# packages), that it's easier to just issue a warning if the view date is
# different from the "latest" date.
set(RZANSEL_VIEW_DATE ${RZANSEL_VIEW_DATE_LATEST}
    CACHE STRING "Darwin dependency view being used")
if (NOT RZANSEL_VIEW_DATE_LATEST STREQUAL RZANSEL_VIEW_DATE)
    message(WARNING "Your current build directory was configured with a \
        set of Darwin dependencies from ${RZANSEL_VIEW_DATE}, but your current \
        code checkout prefers a set of Darwin dependencies from \
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

message(STATUS "Darwin Build Settings
              RZANSEL_ARCH: ${RZANSEL_ARCH}
         RZANSEL_VIEW_DATE: ${RZANSEL_VIEW_DATE}
          RZANSEL_COMPILER: ${RZANSEL_COMPILER}
              RZANSEL_CUDA: ${RZANSEL_CUDA}
    RZANSEL_PROJECT_PREFIX: ${RZANSEL_PROJECT_PREFIX}
                GPU_COUNT: ${GPU_COUNT}
")

set(RZANSEL_ARCH_PREFIX ${RZANSEL_PROJECT_PREFIX}/views/rzansel/${RZANSEL_ARCH})

if (NOT EXISTS ${RZANSEL_ARCH_PREFIX})
    message(WARNING "No dependencies detected for \
        RZANSEL_ARCH=\"${RZANSEL_ARCH}\" at ${RZANSEL_ARCH_PREFIX}")
    return()
endif()

string(TOLOWER "${RZANSEL_COMPILER}" RZANSEL_COMPILER_LOWER)
set(RZANSEL_COMPILER_PREFIX ${RZANSEL_ARCH_PREFIX}/${RZANSEL_COMPILER_LOWER})

if (NOT EXISTS ${RZANSEL_COMPILER_PREFIX})
    message(WARNING "No dependencies detected for \
        RZANSEL_COMPILER=\"${RZANSEL_COMPILER}\" at ${RZANSEL_COMPILER_PREFIX}")
    return()
endif()

set(RZANSEL_VIEW_PREFIX ${RZANSEL_COMPILER_PREFIX}/${RZANSEL_VIEW_DATE})
if (NOT EXISTS ${RZANSEL_VIEW_PREFIX})
    message(WARNING "No view detected for \
        RZANSEL_VIEW_DATE=\"${RZANSEL_VIEW_DATE}\" at ${RZANSEL_VIEW_PREFIX}")
    return()
endif()

if (RZANSEL_CUDA)
    # Location of CUDA
    set(CUDAToolkit_ROOT /usr/tce/packages/cuda/cuda-${RZANSEL_CUDA_VERSION} 
      CACHE STRING "CUDA Location")

    # All of this code ensures that the CUDA build uses the correct nvcc, and
    # that we don't have to depend on the user's environment. As of this
    # writing, nvcc_wrapper can only find nvcc by way of the PATH variable -
    # there's no way to specify it in the command line. Therefore, we need to
    # make sure that the expected nvcc is in the PATH when nvcc_wrapper is
    # executed.

    # This only holds for the length of the cmake process, but is necessary
    # for making sure the compiler checks pass
    set(ENV{PATH} "${CUDAToolkit_ROOT}/bin:$ENV{PATH}")

    # We must make sure nvcc is in the path for the compilation and link stages,
    # so these launch rules add it to the path. This unfortunately caches the
    # PATH environment variable at the time of configuration. I couldn't figure
    # out how to evaluate the PATH variable at build time. Ideally, changing
    # your PATH shouldn't change the build anyway, so I think this is low
    # impact, but it does differ from typical CMake usage.

    # CMAKE_CXX_COMPILER_LAUNCHER expects a list of command line arguments
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CMAKE_COMMAND} -E env "PATH=$ENV{PATH}")
    # RULE_LAUNCH_LINK expects a space separated string
    set_property(
        GLOBAL PROPERTY
        RULE_LAUNCH_LINK "${CMAKE_COMMAND} -E env PATH=$ENV{PATH}")

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

# Let the user specify the compiler if they really want to. Otherwise, point
# to the compilers specified by the RZANSEL_ options
if (RZANSEL_COMPILER MATCHES "GCC")
    set(GCC_VERSION ${RZANSEL_${RZANSEL_COMPILER}_VERSION})
    set(GCC_PREFIX 
      /usr/tce/packages/gcc/gcc-${GCC_VERSION})

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
        "Found view on Darwin for compiler version ${RZANSEL_COMPILER}, but \
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
endif()

# Add dependencies into `CMAKE_PREFIX_PATH`
list(PREPEND CMAKE_PREFIX_PATH ${RZANSEL_VIEW_PREFIX})

# Testing parameters
if (RZANSEL_CUDA AND GPU_COUNT LESS 4)
    set(NUM_RANKS ${GPU_COUNT})
else()
    set(NUM_RANKS 4) 
endif()

set(NUM_MPI_PROC_TESTING ${NUM_RANKS} CACHE STRING "CI runs tests with 2 MPI ranks by default.")
set(NUM_GPU_DEVICES_PER_NODE ${NUM_RANKS} CACHE STRING "Number of gpu devices to use when testing if built with Kokkos_ENABLE_CUDA")
set(NUM_OMP_THREADS_PER_RANK 1 CACHE STRING "Number of threads to use when testing if built with Kokkos_ENABLE_OPENMP")
set(TEST_NUMPROC_FLAG "-a" CACHE STRING "Flag to set number of processes")
string(APPEND TEST_MPIOPTS "-n 1 -g ${NUM_GPU_DEVICES_PER_NODE} -c ${NUM_MPI_PROC_TESTING} -r 1 -d packed --smpiargs='-gpu'")
