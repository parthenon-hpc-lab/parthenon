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

if (PARTHENON_COPYRIGHT_CHECK_DEFAULT)
    set(ALL "ALL")
endif()

set(USE_DUMMY_COPYRIGHT_CHECK OFF)

find_package(Git)

if (NOT Git_FOUND)
  set(USE_DUMMY_COPYRIGHT_CHECK ON)
else()
  execute_process(
    COMMAND ${GIT_EXECUTABLE} log --format='%ae' -n 1
    OUTPUT_VARIABLE AUTHOR
  )
  if (NOT AUTHOR MATCHES "@lanl.gov")
    set(USE_DUMMY_COPYRIGHT_CHECK ON)
  endif()
endif()

if(USE_DUMMY_COPYRIGHT_CHECK)
  add_custom_target(
    check-copyright ${ALL}
    COMMAND ${CMAKE_COMMAND} -E echo "WARNING: Triad copyright check DISABLED"
  )

  add_custom_target(
    update-copyright
    COMMAND ${CMAKE_COMMAND} -E echo "WARNING: Triad copyright update DISABLED"
  )

  return()
endif()

file(
    GLOB_RECURSE COPYRIGHTABLE
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    CONFIGURE_DEPENDS
        src/[^\.]*.cpp example/[^\.]*.cpp tst/[^\.]*.cpp
        src/[^\.]*.hpp example/[^\.]*.hpp tst/[^\.]*.hpp
        scripts/python/[^\.]*.py
        cmake/[^\.]*)

list(APPEND COPYRIGHTABLE
    src/CMakeLists.txt
    example/CMakeLists.txt
    CMakeLists.txt
    CPPLINT.cfg
    LICENSE)

set(OUTPUTS)
set(UPDATES)
foreach(INPUT ${COPYRIGHTABLE})
    set(OUTPUT copyright/${INPUT}.copyright)

    get_filename_component(OUTPUT_DIR ${OUTPUT} DIRECTORY)

    add_custom_command(
        OUTPUT ${OUTPUT}
        COMMAND
            ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_LIST_DIR}/CheckCopyrightScript.cmake
                check ${CMAKE_CURRENT_SOURCE_DIR} ${GIT_EXECUTABLE} ${INPUT}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_DIR}
        COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}
        DEPENDS ${INPUT} ${CMAKE_CURRENT_LIST_DIR}/CheckCopyrightScript.cmake
        COMMENT "Checking Triad Copyright for ${INPUT}"
    )

    list(APPEND OUTPUTS ${OUTPUT})

    list(
        APPEND UPDATES
        COMMAND
            ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_LIST_DIR}/CheckCopyrightScript.cmake
                update ${CMAKE_CURRENT_SOURCE_DIR} ${GIT_EXECUTABLE} ${INPUT})
endforeach()



add_custom_target(
    check-copyright ${ALL}
    DEPENDS ${OUTPUTS}
    COMMENT "Triad copyright up-to-date"
)

add_custom_target(
    update-copyright
    ${UPDATES}
    COMMENT "Update Triad copyrights"
)
