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

if (NOT PARTHENON_SYSTEM_DETECTION)
    return()
endif()

cmake_host_system_information(RESULT HOSTNAME QUERY HOSTNAME)

if (HOSTNAME MATCHES "darwin-fe[0-9]\.lanl\.gov" OR HOSTNAME MATCHES "cn[0-9]+")
    # Running on Darwin
    if(CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "ppc64le")
        # On a PowerPC system
        if(CMAKE_CXX_COMPILER_ID STREQUAL "XL")
            # With the IBM XL compiler
            include(platform/lanl/DarwinPower9XL.cmake)
            return()
        endif()
    endif()
endif()
