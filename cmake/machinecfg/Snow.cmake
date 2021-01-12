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
# - `SNOW_VIEW_DATE` - The date the dependencies were installed.
#       Default:
#           unset, which results in using the view associated with your
#           current commit. See `SNOW_VIEW_DATE_LATEST`
# - `SNOW_COMPILER` - Compiler family to use
#       Default: "INTEL"
#       Possible Values: "GCC", "GCC9", "INTEL", "INTEL19"
# - `SNOW_PROJECT_PREFIX`
#   Description: [ADVANCED] Point to an alternative parthenon-project path
#       Default: /projects/parthenon-int/parthenon-project

# NOTE: When updating dependencies with new compilers or packages, you should
# ideally only need to update these variables to change the default behavior.
set(SNOW_VIEW_DATE_LATEST "2021-01-11")
set(SNOW_GCC_PREFERRED "GCC9")
set(SNOW_INTEL_PREFERRED "INTEL19")
set(SNOW_COMPILER_PREFERRED "INTEL")

set(SNOW_GCC9_VERSION "9.3.0")

set(SNOW_INTEL19_VERSION "19.1.3")
set(SNOW_INTEL19_GCC "GCC9")
set(SNOW_INTEL19_LOCATION "/usr/projects/hpcsoft/toss3/common/x86_64/intel-clusterstudio/2020.4.912/compilers_and_libraries_2020.4.304/linux")

set(SNOW_MPI_VERSION "3.1.6")

# It would be nice if we could let this variable float with the current code
# checkout, but unfortunately CMake caches enough other stuff (like find
# packages), that it's easier to just issue a warning if the view date is
# different from the "latest" date.
set(SNOW_VIEW_DATE ${SNOW_VIEW_DATE_LATEST}
    CACHE STRING "Snow dependency view being used")
if (NOT SNOW_VIEW_DATE_LATEST STREQUAL SNOW_VIEW_DATE)
    message(WARNING "Your current build directory was configured with a \
        set of Snow dependencies from ${SNOW_VIEW_DATE}, but your current \
        code checkout prefers a set of Snow dependencies from \
        ${SNOW_VIEW_DATE_LATEST}. Consider configuring a new build \
        directory.")
endif()

if (NOT DEFINED SNOW_COMPILER)
    set(SNOW_COMPILER ${SNOW_COMPILER_PREFERRED})
endif()

if (SNOW_COMPILER STREQUAL "GCC")
    set(SNOW_COMPILER ${SNOW_GCC_PREFERRED})
elseif (SNOW_COMPILER STREQUAL "INTEL")
    set(SNOW_COMPILER ${SNOW_INTEL_PREFERRED})
endif()

set(SNOW_PROJECT_PREFIX /usr/projects/parthenon/parthenon-project
    CACHE STRING "Path to parthenon-project checkout")
mark_as_advanced(SNOW_PROJECT_PREFIX)

message(STATUS "Snow Build Settings
         SNOW_VIEW_DATE: ${SNOW_VIEW_DATE}
          SNOW_COMPILER: ${SNOW_COMPILER}
    SNOW_PROJECT_PREFIX: ${SNOW_PROJECT_PREFIX}
")

set(SNOW_ARCH_PREFIX ${SNOW_PROJECT_PREFIX}/views/snow/x86_64)

if (NOT EXISTS ${SNOW_ARCH_PREFIX})
    message(WARNING "No dependencies detected at ${SNOW_ARCH_PREFIX}")
    return()
endif()

# Set SNOW_GCC
if (SNOW_COMPILER MATCHES "GCC")
    set(SNOW_GCC ${SNOW_COMPILER})
elseif(SNOW_COMPILER MATCHES "INTEL")
    set(SNOW_GCC ${SNOW_INTEL19_GCC})
endif()

string(TOLOWER "${SNOW_GCC}" SNOW_GCC_LOWER)
set(SNOW_COMPILER_PREFIX ${SNOW_ARCH_PREFIX}/${SNOW_GCC_LOWER})

if (NOT EXISTS ${SNOW_COMPILER_PREFIX})
    message(WARNING "No dependencies detected for \
        SNOW_COMPILER=\"${SNOW_COMPILER}\" at ${SNOW_COMPILER_PREFIX}")
    return()
endif()

set(SNOW_VIEW_PREFIX ${SNOW_COMPILER_PREFIX}/${SNOW_VIEW_DATE})
if (NOT EXISTS ${SNOW_VIEW_PREFIX})
    message(WARNING "No view detected for \
        SNOW_VIEW_DATE=\"${SNOW_VIEW_DATE}\" at ${SNOW_VIEW_PREFIX}")
    return()
endif()

set(GCC_VERSION ${SNOW_${SNOW_GCC}_VERSION})
if (GCC_VERSION)
    set(GCC_PREFIX /usr/projects/hpcsoft/toss3/common/x86_64/gcc/${GCC_VERSION})
endif()

# Let the user specify the compiler if they really want to. Otherwise, point
# to the compilers specified by the SNOW_ options
if (SNOW_COMPILER MATCHES "GCC")
    if (GCC_VERSION)
        set(CMAKE_C_COMPILER ${GCC_PREFIX}/bin/gcc
            CACHE STRING "gcc ${GCC_VERSION}")
        set(CMAKE_CXX_COMPILER ${GCC_PREFIX}/bin/g++
            CACHE STRING "gcc ${GCC_VERSION}")

        set(CMAKE_BUILD_RPATH ${GCC_PREFIX}/lib64
            CACHE STRING "rpath libs")
    endif()
elseif (SNOW_COMPILER MATCHES "INTEL")
    set(INTEL_VERSION ${SNOW_${SNOW_COMPILER}_VERSION})

    if (INTEL_VERSION)
        set(INTEL_PREFIX ${SNOW_${SNOW_COMPILER}_LOCATION})
        set(CMAKE_C_COMPILER ${INTEL_PREFIX}/bin/intel64/icc CACHE STRING "intel ${INTEL_VERSION}")
        set(CMAKE_CXX_COMPILER ${INTEL_PREFIX}/bin/intel64/icpc CACHE STRING "intel ${INTEL_VERSION}")
    endif()

    if (GCC_VERSION)
        list(APPEND CMAKE_C_FLAGS_INIT -gcc-name=${GCC_PREFIX}/bin/gcc)
        list(APPEND CMAKE_CXX_FLAGS_INIT -gxx-name=${GCC_PREFIX}/bin/g++)
    endif()
endif()

if (NOT DEFINED CMAKE_CXX_COMPILER)
    message(
        FATAL_ERROR
        "Found view on Snow for compiler version ${SNOW_COMPILER}, but \
        don't know how to map it to a specific compiler. Either update \
        your Parthenon checkout or explicitly set CMAKE_C_COMPILER and \
        CMAKE_CXX_COMPILER")
endif()

# MPI - We use the system modules since replicating them in spack can be
# difficult.
if (GCC_VERSION)
    set(MPI_ROOT
        /usr/projects/hpcsoft/toss3/snow/openmpi/${SNOW_MPI_VERSION}-gcc-${GCC_VERSION}
        CACHE STRING "MPI Location")
endif()

# clang-format
set(CLANG_FORMAT
    ${SNOW_PROJECT_PREFIX}/tools/x86_64/bin/clang-format-8
    CACHE STRING "clang-format-8")

# Kokkos settings
set(Kokkos_ARCH_BDW ON CACHE BOOL "Target Broadwell")

# Add dependencies into `CMAKE_PREFIX_PATH`
list(PREPEND CMAKE_PREFIX_PATH ${SNOW_VIEW_PREFIX})

# Testing parameters
set(NUM_RANKS 4)

set(NUM_MPI_PROC_TESTING ${NUM_RANKS} CACHE STRING "CI runs tests with 4 MPI ranks")
set(NUM_OMP_THREADS_PER_RANK 1 CACHE STRING "Number of threads to use when testing if built with Kokkos_ENABLE_OPENMP")
set(TEST_MPIEXEC /usr/bin/srun CACHE STRING "Use srun to run executables")
set(SERIAL_WITH_MPIEXEC ON)