#=========================================================================================
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
#=========================================================================================

# This script checks that a file has an up to date copyright. Only runs for
# users with @lanl.gov email addresses.

set(SOURCE_DIR ${CMAKE_ARGV3})
set(GIT_EXECUTABLE ${CMAKE_ARGV4})
set(INPUT ${CMAKE_ARGV5})

string(TIMESTAMP CURRENT_YEAR "%Y")

# Check if the file has an up to date copyright
file(READ ${SOURCE_DIR}/${INPUT} CONTENTS)
string(REGEX MATCH "\\(C\\) \\(or copyright\\) 2020-${CURRENT_YEAR}\\. Triad National Security, LLC\\. All rights reserved\\." HAS_COPYRIGHT ${CONTENTS})

if (HAS_COPYRIGHT)
    return()
endif()

# If not, check if the file has been modified.
execute_process(
    COMMAND ${GIT_EXECUTABLE} diff --exit-code ${INPUT}
    WORKING_DIRECTORY ${SOURCE_DIR}
    RESULT_VARIABLE MODIFIED
    OUTPUT_QUIET)

# If the file is modified, then the copyright needs to be updated
if (MODIFIED)
    set(REQUIRES_COPYRIGHT_UPDATE ON)
else()
    # Otherwise check if the file *has* been modified by somebody with a LANL email.
    execute_process(
        COMMAND
            ${GIT_EXECUTABLE} log
                --since=01-01-${CURRENT_YEAR}
                --author=@lanl\\.gov
                --exit-code
                -1
                --pretty=%H
                ${INPUT}
        WORKING_DIRECTORY ${SOURCE_DIR}
        RESULT_VARIABLE HAS_CHANGES
        OUTPUT_QUIET)

    # If the file has changes by a LANL email, the copyright must be updated
    if (HAS_CHANGES)
        set(REQUIRES_COPYRIGHT_UPDATE ON)
    endif()
endif()

if (REQUIRES_COPYRIGHT_UPDATE)
    message("
${INPUT}: Requires copyright update to
(C) (or copyright) 2020-${CURRENT_YEAR}. Triad National Security, LLC. All rights reserved.
")
    message(FATAL_ERROR)
endif()
