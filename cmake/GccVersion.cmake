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

# This file defines a function which can extract the version of a GCC executable.

function(get_gcc_version GCC_PATH OUTPUT_VAR)
    execute_process(
        COMMAND ${GCC_PATH} --version
        OUTPUT_VARIABLE GCC_VERSION_OUTPUT
    )

    if (GCC_VERSION_OUTPUT MATCHES "g\\+\\+ \\(GCC\\) ([0-9]+\.[0-9]+\.[0-9]+)")
        set(${OUTPUT_VAR} ${CMAKE_MATCH_1} PARENT_SCOPE)
    else()
        set(${OUTPUT_VAR} FALSE PARENT_SCOPE)
    endif()
endfunction()
