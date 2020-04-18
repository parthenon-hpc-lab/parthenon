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

message(STATUS "Detected Darwin Power9 with IBM XL. Applying customizations.")

if(CMAKE_VERSION VERSION_LESS "3.15")
    message(
        FATAL_ERROR
        "IBM XL is only supported on cmake version 3.15 and newer. You are \
        running ${CMAKE_VERSION}."
    )
endif()

# Everything seems to be configured with Cuda 10.1. We should perhaps just
# auto-detect this, though, in case the CUDA version is bumped at some point.
set(XL_CUDA_VERSION "10.1")

if ("16.1.1.7" VERSION_GREATER CMAKE_CXX_COMPILER_VERSION)
    message(
        FATAL_ERROR
        "Your version of IBM XL is non supported on Darwin Power9. You must \
        use version 16.1.1.7 or newer.")
endif()

find_program(GCC NAMES g++)
execute_process(
    COMMAND ${GCC} --version
    OUTPUT_VARIABLE GCC_VERSION_OUTPUT
)

string(
    REGEX MATCH "g\\+\\+ \\(GCC\\) ([0-9]+\.[0-9]+\.[0-9]+)"
    DISCARD ${GCC_VERSION_OUTPUT})

set(XL_GCC_VERSION ${CMAKE_MATCH_1})
if (XL_GCC_VERSION VERSION_LESS "5.4.0")
    set(BAD_GCC_VERSION ON)
endif()

message(STATUS "GCC Version: ${XL_GCC_VERSION}")

set(XL_CONFIG_ROOT "/projects/opt/ppc64le/ibm/xlc-${CMAKE_CXX_COMPILER_VERSION}/xlC/16.1.1/etc")
set(XL_CONFIG_FILE "${XL_CONFIG_ROOT}/xlc.cfg.rhel.7.7.gcc.${XL_GCC_VERSION}.cuda.${XL_CUDA_VERSION}")
if(NOT EXISTS ${XL_CONFIG_FILE})
    set(GCC_BAD_VERSION ON)
endif()

if (GCC_BAD_VERSION)
    file(
        GLOB XL_AVAILABLE_GCC_CONFIGURATIONS
        RELATIVE "${XL_CONFIG_ROOT}"
        "${XL_CONFIG_ROOT}/xlc.cfg.rhel.7.7.gcc.*.cuda.${XL_CUDA_VERSION}"
    )

    message(STATUS "${XL_AVAILABLE_GCC_CONFIGURATIONS}")

    list(
        TRANSFORM XL_AVAILABLE_GCC_CONFIGURATIONS
        REPLACE
            "xlc\.cfg\.rhel\.7\.7\.gcc\.(.*)\.cuda\.${XL_CUDA_VERSION}"
            "\\1"
        OUTPUT_VARIABLE XL_AVAILABLE_GCC_VERSIONS
    )

    foreach(GCC_VERSION ${XL_AVAILABLE_GCC_VERSIONS})
        if ("5.4.0" VERSION_LESS GCC_VERSION)
            list(APPEND XL_COMPATIBLE_GCC_VERSIONS ${GCC_VERSION})
        endif()
    endforeach()
    
    list(JOIN XL_COMPATIBLE_GCC_VERSIONS ", " XL_COMPATIBLE_GCC_VERSIONS)
    message(
        FATAL_ERROR
        "Found GCC version ${XL_GCC_VERSION} at ${GCC}. This version is not \
        configured with your IBM XL compiler, or it is not new enough (5.4.0 \
        or newer required). Discovered configurations for GCC versions \
        (${XL_COMPATIBLE_GCC_VERSIONS}) in ${XL_CONFIG_ROOT}"
    )
endif()

list(APPEND CMAKE_CXX_FLAGS -F${XL_CONFIG_FILE})