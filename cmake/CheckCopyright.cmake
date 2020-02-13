#=========================================================================================
# Athena++ astrophysical MHD code
# Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
# Licensed under the 3-clause BSD License, see LICENSE file for details
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

# TODO: Probably shouldn't be done at configure time - instead we should have a separate
# build task that does this.

file(GLOB_RECURSE COPYRIGHTABLE_SOURCES src/*.cpp)
file(GLOB_RECURSE COPYRIGHTABLE_HEADERS src/*.hpp)
file(GLOB_RECURSE COPYRIGHTABLE_CMAKE cmake/*)

set(COPYRIGHTABLE
    ${COPYRIGHTABLE_SOURCES}
    ${COPYRIGHTABLE_HEADERS}
    ${COPYRIGHTABLE_CMAKE}
    CMakeLists.txt
    CPPLINT.cfg
    LICENSE
    athdfExample.py)

string(TIMESTAMP CURRENT_YEAR "%Y")
set(LAST_UPDATED 2020)

if (NOT CURRENT_YEAR STREQUAL LAST_UPDATED)
    message(WARNING "Triad copyright is out of date. Consider updating.")
endif()

foreach(FILE ${COPYRIGHTABLE})
    file(READ ${FILE} CONTENTS)

    string(REGEX MATCH "\\(C\\) \\(or copyright\\) ${LAST_UPDATED}\\. Triad National Security, LLC\\. All rights reserved\\." HAS_COPYRIGHT ${CONTENTS})

    if (NOT HAS_COPYRIGHT)
        message(FATAL_ERROR "File ${FILE} does not contain an up to date copy of the Triad copyright")
    endif()
endforeach()
