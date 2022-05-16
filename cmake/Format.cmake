#=========================================================================================
# (C) (or copyright) 2020-2022. Triad National Security, LLC. All rights reserved.
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

find_program(
    CLANG_FORMAT
    NAMES
        clang-format-13 # Debian package manager, among others, provide this name
        clang-format-mp-13.0 # MacPorts
        clang-format-12 # Debian package manager, among others, provide this name
        clang-format-mp-12.0 # MacPorts
        clang-format-11 # Debian package manager, among others, provide this name
        clang-format-mp-11.0 # MacPorts
        clang-format # Default name
    )

find_program(BLACK  NAMES black)

if (CLANG_FORMAT AND NOT CLANG_FORMAT_VERSION)
    # Get clang-format --version
    execute_process(
        COMMAND ${CLANG_FORMAT} --version
        OUTPUT_VARIABLE CLANG_FORMAT_VERSION_OUTPUT)

    if (CLANG_FORMAT_VERSION_OUTPUT MATCHES "clang-format version ([0-9]+\.[0-9]+\.[0-9]+)")
        set(CLANG_FORMAT_VERSION ${CMAKE_MATCH_1})
        message(STATUS "clang-format --version: " ${CLANG_FORMAT_VERSION})

        set(CLANG_FORMAT_VERSION ${CLANG_FORMAT_VERSION} CACHE STRING "clang-format version")
    endif()
endif()

if (BLACK)
    # Get black --version
    execute_process(
      COMMAND ${BLACK} --version
      OUTPUT_VARIABLE BLACK_VERSION_OUTPUT)

    message(STATUS "black --version: " ${BLACK_VERSION_OUTPUT})

endif()

if (NOT CLANG_FORMAT_VERSION)
    message(
        WARNING
        "Couldn't determine clang-format version. clang-format >=11.0 is \
        required - results on previous versions may not be stable")

    set(CLANG_FORMAT_VERSION "0.0.0" CACHE STRING "clang-format version not found")
elseif (CLANG_FORMAT_VERSION VERSION_LESS "11.0")
    message(
        WARNING
        "clang-format version >=11.0 is required - results on previous \
        versions may not be stable")
endif()

# Specifically trying to exclude external here - I'm not sure if there's a better way
set(
    GLOBS
    ${PROJECT_SOURCE_DIR}/src/[^\.]*.cpp     ${PROJECT_SOURCE_DIR}/src/[^\.]*.hpp
    ${PROJECT_SOURCE_DIR}/tst/[^\.]*.cpp     ${PROJECT_SOURCE_DIR}/tst/[^\.]*.hpp
    ${PROJECT_SOURCE_DIR}/example/[^\.]*.cpp ${PROJECT_SOURCE_DIR}/example/[^\.]*.hpp
)

# Specifically trying to exclude external here - I'm not sure if there's a better way
set(
    PY_GLOBS
    ${parthenon_SOURCE_DIR}/scripts/[^\.]*.py
    ${parthenon_SOURCE_DIR}/tst/[^\.]*.py
    ${parthenon_SOURCE_DIR}/example/[^\.]*.py
)

file(GLOB_RECURSE FORMAT_SOURCES CONFIGURE_DEPENDS ${GLOBS})
file(GLOB_RECURSE PY_FORMAT_SOURCES CONFIGURE_DEPENDS ${PY_GLOBS})

if (CLANG_FORMAT)
  if (BLACK)
    add_custom_target(format
      COMMAND ${CLANG_FORMAT} -i ${FORMAT_SOURCES}
      COMMAND ${BLACK} ${PY_FORMAT_SOURCES}
      VERBATIM)
  else()
    add_custom_target(format
      COMMAND echo "Only C++ files will be formatted black was not found"
      COMMAND ${CLANG_FORMAT} -i ${FORMAT_SOURCES}
      VERBATIM)
  endif()
elseif (BLACK)
  add_custom_target(format
    COMMAND echo "Only Python files will be formatted clang_format was not found"
    COMMAND ${BLACK} ${PY_FORMAT_SOURCES}
    VERBATIM)
endif()
