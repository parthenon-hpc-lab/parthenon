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

# Only LANL employees need to update the Triad copyright. This only inserts the
# copyright check if the author is from LANL.

find_package(Git)
if (NOT Git_FOUND)
    return()
endif()

execute_process(
    COMMAND ${GIT_EXECUTABLE} config user.email
    OUTPUT_VARIABLE AUTHOR)

if (NOT AUTHOR MATCHES "@lanl.gov")
    return()
endif()

file(
    GLOB_RECURSE COPYRIGHTABLE
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    CONFIGURE_DEPENDS
        src/[^\.]*.cpp example/[^\.]*.cpp
        src/[^\.]*.hpp example/[^\.]*.hpp
        scripts/python/[^\.]*.py
        cmake/[^\.]*)

list(APPEND COPYRIGHTABLE
    src/CMakeLists.txt
    example/CMakeLists.txt
    CMakeLists.txt
    CPPLINT.cfg
    LICENSE)

set(OUTPUTS)
foreach(INPUT ${COPYRIGHTABLE})
    set(OUTPUT copyright/${INPUT}.copyright)

    get_filename_component(OUTPUT_DIR ${OUTPUT} DIRECTORY)

    add_custom_command(
        OUTPUT ${OUTPUT}
        COMMAND
            ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_LIST_DIR}/CheckCopyrightScript.cmake
                ${CMAKE_CURRENT_SOURCE_DIR} ${GIT_EXECUTABLE} ${INPUT}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_DIR}
        COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}
        DEPENDS ${INPUT} ${CMAKE_CURRENT_LIST_DIR}/CheckCopyrightScript.cmake
        COMMENT "Checking Triad Copyright for ${INPUT}"
    )

    list(APPEND OUTPUTS ${OUTPUT})
endforeach()

add_custom_target(
    check-copyright ALL
    DEPENDS ${OUTPUTS}
    COMMENT "Triad copyright up-to-date"
)
