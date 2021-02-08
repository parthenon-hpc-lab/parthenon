#========================================================================================
# Parthenon performance portable AMR framework
# Copyright(C) 2020 The Parthenon collaboration
# Licensed under the 3-clause BSD License, see LICENSE file for details
#========================================================================================
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
#========================================================================================

# OPTIONS:
# - `DARWIN_ARCH` - define which processor is being used
#       Default:
#           `DARWIN_ARCH` environment variable, if it exists, otherwise
#           the result of `uname -m`.
#       Possible Values:
#           `x86_64` (assumes at least haswell)
#           `ppc64le` (assumes power9 + volta gpus)
# - `DARWIN_VIEW_DATE` - The date the dependencies were installed.
#       Default:
#           unset, which results in using the view associated with your
#           current commit. See `DARWIN_VIEW_DATE_LATEST`
# - `DARWIN_COMPILER` - Compiler family to use
#       Default: "GCC"
#       Possible Values: "GCC", "GCC9"
# - `DARWIN_CUDA` - Build for CUDA
#       Default: ON if nvidia-smi finds at least one GPU, OFF otherwise
# - `DARWIN_PROJECT_PREFIX`
#   Description: [ADVANCED] Point to an alternative parthenon-project path
#       Default: /projects/parthenon-int/parthenon-project

# This little bit picks up the target architecture, which determines the
# target environment and system modules.
if (DEFINED ENV{DARWIN_ARCH})
    set(DARWIN_ARCH_INIT $ENV{DARWIN_ARCH})
else()
    execute_process(COMMAND uname -m OUTPUT_VARIABLE DARWIN_ARCH_INIT)
    string(STRIP "${DARWIN_ARCH_INIT}" DARWIN_ARCH_INIT)
endif()

set(DARWIN_ARCH ${DARWIN_ARCH_INIT} CACHE STRING "Target Darwin architecture")

# NOTE: When updating dependencies with new compilers or packages, you should
# ideally only need to update these variables to change the default behavior.
if (DARWIN_ARCH STREQUAL "x86_64")
    set(DARWIN_VIEW_DATE_LATEST "2020-12-08")
    set(DARWIN_GCC_PREFERRED "GCC9")
    set(DARWIN_COMPILER_PREFERRED "GCC")

    set(DARWIN_GCC9_VERSION "9.3.0")

    set(DARWIN_MPI_PACKAGE "openmpi")
    set(DARWIN_MPI_VERSION "4.0.3")
    set(DARWIN_CUDA_VERSION "11.0")

    set(DARWIN_CUDA_DEFAULT OFF)
elseif (DARWIN_ARCH STREQUAL "ppc64le")
    set(DARWIN_VIEW_DATE_LATEST "2021-02-08")

    set(DARWIN_GCC_PREFERRED "GCC9")
    set(DARWIN_COMPILER_PREFERRED "GCC")

    set(DARWIN_GCC9_VERSION "9.3.0")
    set(DARWIN_MICROARCH_PATH "/p9")

    set(DARWIN_MPI_PACKAGE "smpi")
    set(DARWIN_MPI_VERSION "10.3.0.1")
    set(DARWIN_CUDA_VERSION "11.0")

    set(DARWIN_CUDA_DEFAULT ON)
    # Because using spectrum serial version requires calling mpiexec
    set(SERIAL_WITH_MPIEXEC ON)
    string(APPEND TEST_MPIOPTS "-gpu")
else()
    message(
        FATAL_ERROR
        "Darwin does not have any configuration for arch ${DARWIN_ARCH}")
endif()

execute_process(
    COMMAND nvidia-smi -L
    OUTPUT_VARIABLE FOUND_GPUS)
string(REPLACE "\n" ";" FOUND_GPUS ${FOUND_GPUS})

list(FILTER FOUND_GPUS INCLUDE REGEX "GPU [0-9]")
list(LENGTH FOUND_GPUS GPU_COUNT)

if (GPU_COUNT EQUAL 0)
    set(DARWIN_CUDA_DEFAULT OFF)
else()
    set(DARWIN_CUDA_DEFAULT ON)
endif()

set(DARWIN_CUDA ${DARWIN_CUDA_DEFAULT} CACHE BOOL "Build for CUDA")

# It would be nice if we could let this variable float with the current code
# checkout, but unfortunately CMake caches enough other stuff (like find
# packages), that it's easier to just issue a warning if the view date is
# different from the "latest" date.
set(DARWIN_VIEW_DATE ${DARWIN_VIEW_DATE_LATEST}
    CACHE STRING "Darwin dependency view being used")
if (NOT DARWIN_VIEW_DATE_LATEST STREQUAL DARWIN_VIEW_DATE)
    message(WARNING "Your current build directory was configured with a \
        set of Darwin dependencies from ${DARWIN_VIEW_DATE}, but your current \
        code checkout prefers a set of Darwin dependencies from \
        ${DARWIN_VIEW_DATE_LATEST}. Consider configuring a new build \
        directory.")
endif()

if (NOT DEFINED DARWIN_COMPILER)
    set(DARWIN_COMPILER ${DARWIN_COMPILER_PREFERRED})
endif()

if (DARWIN_COMPILER STREQUAL "GCC")
    set(DARWIN_COMPILER ${DARWIN_GCC_PREFERRED})
endif()

set(DARWIN_PROJECT_PREFIX /projects/parthenon-int/parthenon-project
    CACHE STRING "Path to parthenon-project checkout")
mark_as_advanced(DARWIN_PROJECT_PREFIX)

message(STATUS "Darwin Build Settings
              DARWIN_ARCH: ${DARWIN_ARCH}
         DARWIN_VIEW_DATE: ${DARWIN_VIEW_DATE}
          DARWIN_COMPILER: ${DARWIN_COMPILER}
              DARWIN_CUDA: ${DARWIN_CUDA}
    DARWIN_PROJECT_PREFIX: ${DARWIN_PROJECT_PREFIX}
                GPU_COUNT: ${GPU_COUNT}
")

set(DARWIN_ARCH_PREFIX ${DARWIN_PROJECT_PREFIX}/views/darwin/${DARWIN_ARCH})

if (NOT EXISTS ${DARWIN_ARCH_PREFIX})
    message(WARNING "No dependencies detected for \
        DARWIN_ARCH=\"${DARWIN_ARCH}\" at ${DARWIN_ARCH_PREFIX}")
    return()
endif()

string(TOLOWER "${DARWIN_COMPILER}" DARWIN_COMPILER_LOWER)
set(DARWIN_COMPILER_PREFIX ${DARWIN_ARCH_PREFIX}/${DARWIN_COMPILER_LOWER})

if (NOT EXISTS ${DARWIN_COMPILER_PREFIX})
    message(WARNING "No dependencies detected for \
        DARWIN_COMPILER=\"${DARWIN_COMPILER}\" at ${DARWIN_COMPILER_PREFIX}")
    return()
endif()

set(DARWIN_VIEW_PREFIX ${DARWIN_COMPILER_PREFIX}/${DARWIN_VIEW_DATE})
if (NOT EXISTS ${DARWIN_VIEW_PREFIX})
    message(WARNING "No view detected for \
        DARWIN_VIEW_DATE=\"${DARWIN_VIEW_DATE}\" at ${DARWIN_VIEW_PREFIX}")
    return()
endif()

if (DARWIN_CUDA)
    # Location of CUDA
    set(CUDAToolkit_ROOT /usr/local/cuda-${DARWIN_CUDA_VERSION}
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
# to the compilers specified by the DARWIN_ options
if (DARWIN_COMPILER MATCHES "GCC")
    set(GCC_VERSION ${DARWIN_${DARWIN_COMPILER}_VERSION})
    set(GCC_PREFIX
        /projects/opt/${DARWIN_ARCH}${DARWIN_MICROARCH_PATH}/gcc/${GCC_VERSION})


    if (GCC_VERSION)
        set(CMAKE_C_COMPILER ${GCC_PREFIX}/bin/gcc
            CACHE STRING "gcc ${GCC_VERSION}")

        set(DARWIN_CXX_COMPILER ${GCC_PREFIX}/bin/g++)
        if (DARWIN_CUDA)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ccbin ${DARWIN_CXX_COMPILER}")
        else()
            set(CMAKE_CXX_COMPILER ${DARWIN_CXX_COMPILER}
            CACHE STRING "gcc ${GCC_VERSION}")
        endif()

        set(CMAKE_BUILD_RPATH ${GCC_PREFIX}/lib64
            CACHE STRING "rpath libs")
    endif()
endif()

if (NOT DEFINED CMAKE_CXX_COMPILER)
    message(
        FATAL_ERROR
        "Found view on Darwin for compiler version ${DARWIN_COMPILER}, but \
        don't know how to map it to a specific compiler. Either update \
        your Parthenon checkout or explicitly set CMAKE_C_COMPILER and \
        CMAKE_CXX_COMPILER")
endif()

# MPI - We use the system modules since replicating them in spack can be
# difficult.
if (DARWIN_COMPILER MATCHES "GCC")
  if(DARWIN_MPI_PACKAGE STREQUAL "openmpi")
    set(MPI_ROOT 
    /projects/opt/${DARWIN_ARCH}${DARWIN_MICROARCH_PATH}/openmpi/${DARWIN_MPI_VERSION}-gcc_${DARWIN_${DARWIN_COMPILER}_VERSION}
        CACHE STRING "MPI Location")
  elseif(DARWIN_MPI_PACKAGE STREQUAL "smpi")
    set(MPI_ROOT /projects/opt/${DARWIN_ARCH}/ibm/smpi-${DARWIN_MPI_VERSION}
        CACHE STRING "MPI Location")
  endif()
endif()

# clang-format
set(CLANG_FORMAT
    ${DARWIN_PROJECT_PREFIX}/tools/${DARWIN_ARCH}/bin/clang-format-8
    CACHE STRING "clang-format-8")

# Kokkos settings
if (DARWIN_ARCH STREQUAL "x86_64")
    set(Kokkos_ARCH_HSW ON CACHE BOOL "Target Haswell")
elseif(DARWIN_ARCH STREQUAL "ppc64le")
    set(Kokkos_ARCH_POWER9 ON CACHE BOOL "Target Power9")
endif()

if (DARWIN_CUDA)
    set(Kokkos_ENABLE_CUDA ON CACHE BOOL "Cuda")
    set(Kokkos_ARCH_VOLTA70 ON CACHE BOOL "Target V100s")
endif()

# Add dependencies into `CMAKE_PREFIX_PATH`
list(PREPEND CMAKE_PREFIX_PATH ${DARWIN_VIEW_PREFIX})

# Testing parameters
if (DARWIN_CUDA AND GPU_COUNT LESS 2)
    set(NUM_RANKS ${GPU_COUNT})
else()
    set(NUM_RANKS 2)
endif()
set(NUM_MPI_PROC_TESTING ${NUM_RANKS} CACHE STRING "CI runs tests with 2 MPI ranks")
set(NUM_GPU_DEVICES_PER_NODE ${NUM_RANKS} CACHE STRING "Number of gpu devices to use when testing if built with Kokkos_ENABLE_CUDA")
set(NUM_OMP_THREADS_PER_RANK 1 CACHE STRING "Number of threads to use when testing if built with Kokkos_ENABLE_OPENMP")
