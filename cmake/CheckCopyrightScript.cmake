#=========================================================================================
# (C) (or copyright) 2021. Triad National Security, LLC. All rights reserved.
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

set(CHECK_COMMAND ${CMAKE_ARGV3}) # "check" or "update"
set(SOURCE_DIR ${CMAKE_ARGV4})
set(GIT_EXECUTABLE ${CMAKE_ARGV5})
set(INPUT ${CMAKE_ARGV6})

string(TIMESTAMP CURRENT_YEAR "%Y")

# Check if the file has an up to date copyright
file(READ ${SOURCE_DIR}/${INPUT} CONTENTS)

set(
    COPYRIGHT_REGEX
    "\\(C\\) \\(or copyright\\) ([0-9][0-9][0-9][0-9]\\-)?([0-9][0-9][0-9][0-9])\\. Triad National Security, LLC\\. All rights reserved\\."
)

if (NOT CONTENTS MATCHES ${COPYRIGHT_REGEX})
    message("
${INPUT}: Missing copyright
(C) (or copyright) ${CURRENT_YEAR}. Triad National Security, LLC. All rights reserved.
")
    message(FATAL_ERROR)
endif()

set(START_YEAR ${CMAKE_MATCH_1})
set(END_YEAR ${CMAKE_MATCH_2})

if (NOT START_YEAR)
    set(START_YEAR "${END_YEAR}-")
endif()

# If the current year is up-to-date, then nothing needs to be done.
if (CURRENT_YEAR STREQUAL END_YEAR)
    return()
endif()

# If not, check if the file has been modified.
execute_process(
    COMMAND ${GIT_EXECUTABLE} diff --exit-code ${INPUT}
    WORKING_DIRECTORY ${SOURCE_DIR}
    RESULT_VARIABLE MODIFIED
    OUTPUT_QUIET)

# Check if the file is potentially untracked
if (NOT MODIFIED)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} ls-files --error-unmatch ${INPUT}
        WORKING_DIRECTORY ${SOURCE_DIR}
        RESULT_VARIABLE MODIFIED
        OUTPUT_QUIET
        ERROR_QUIET)
endif()

# If the file is modified, then the copyright needs to be updated
if (MODIFIED)
    set(EXPECTED_COPYRIGHT "(C) (or copyright) ${START_YEAR}${CURRENT_YEAR}. Triad National Security, LLC. All rights reserved.")
    if (CHECK_COMMAND STREQUAL "update")
        string(REGEX REPLACE ${COPYRIGHT_REGEX} ${EXPECTED_COPYRIGHT} NEW_CONTENTS "${CONTENTS}")
        file(WRITE ${SOURCE_DIR}/${INPUT} "${NEW_CONTENTS}")
    else()
        message("
${INPUT}: Requires copyright update to
${EXPECTED_COPYRIGHT}

To update automatically run:
$ cmake --build . --target update-copyright
")
        message(FATAL_ERROR)
    endif()
endif()
