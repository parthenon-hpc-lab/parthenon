#=========================================================================================
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
#=========================================================================================

# This file swaps out the specified C++ compiler for nvcc_wrapper. This is done because
# nvcc_wrapper is not a real compiler, and is just a workaround for the fact that nvcc
# doesn't like to compile .cpp files as Cuda.

if (NOT Kokkos_ENABLE_CUDA)
    # TODO: Support installs of Kokkos. This will have to be done by looking up the
    # Kokkos settings and checking if CUDA is enabled.
    return()
endif()

# Cache the environment-provided C++ compiler
if (NOT DEFINED PARTHENON_HOST_CXX_COMPILER)
    if (CMAKE_CXX_COMPILER)
        set(CXX_COMPILER ${CMAKE_CXX_COMPILER})
    elseif(DEFINED ENV{CXX})
        set(CXX_COMPILER $ENV{CXX})
    endif()
    find_program(PARTHENON_HOST_CXX_COMPILER NAMES ${CXX_COMPILER})
endif()

# Check if the user specified nvcc_wrapper for the host compiler and issue a warning.
if (PARTHENON_HOST_CXX_COMPILER MATCHES "nvcc_wrapper")
    message(
        WARNING
        "Parthenon can automatically apply nvcc_wrapper. It is recommended you just \
        allow nvcc_wrapper to be detected.")
    unset(PARTHENON_HOST_CXX_COMPILER CACHE)
    return()
endif()

# Search for nvcc_wrapper. Specify that it can be found in the submodule.
find_program(NVCC_WRAPPER
    NAMES nvcc_wrapper
    PATHS ${CMAKE_SOURCE_DIR}/external/Kokkos/bin)

# Error out if nvcc_wrapper cannot be found
if (NOT NVCC_WRAPPER)
    message(FATAL_ERROR "Could not find nvcc_wrapper - do you have a valid \
        installation of Kokkos? Try `git submodule update --init`")
endif()

# Forcefully override the CMAKE_CXX_COMPILER with `nvcc_wrapper`.
set(CMAKE_CXX_COMPILER ${NVCC_WRAPPER} CACHE STRING "nvcc_wrapper" FORCE)

# Hard-code the host compiler to the provided compiler. This allows you continue to build
# after changing your environment, and matches CMake's behavior of caching the C++
# compiler.
if (PARTHENON_HOST_CXX_COMPILER)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ccbin ${PARTHENON_HOST_CXX_COMPILER}")
endif()

# Apply some flags for compatibility with GCC 8+. This isn't specifically related to
# nvcc_wrapper.
if (PARTHENON_HOST_CXX_COMPILER)
    get_gcc_version(${PARTHENON_HOST_CXX_COMPILER} GCC_VERSION)
    if (GCC_VERSION)
        if (NOT GCC_VERSION VERSION_LESS "8.0.0")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mno-float128")
        endif()
    endif()
endif()
