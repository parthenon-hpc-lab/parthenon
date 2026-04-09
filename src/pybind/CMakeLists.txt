#========================================================================================
# (C) (or copyright) 2020-2026. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001 for Los
# Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
# for the U.S. Department of Energy/National Nuclear Security Administration. All rights
# in the program are reserved by Triad National Security, LLC, and the U.S. Department
# of Energy/National Nuclear Security Administration. The Government is granted for
# itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
# license in this material to reproduce, prepare derivative works, distribute copies to
# the public, perform publicly and display publicly, and to permit others to do so.
#========================================================================================
# This file was made in part with generative AI.

# Python bindings for Parthenon (optional)
option(PARTHENON_ENABLE_PYTHON_BINDINGS "Build Python bindings for parameter input" OFF)

if(PARTHENON_ENABLE_PYTHON_BINDINGS)
  # Suppress policy warning about deprecated FindPythonInterp/FindPythonLibs
  if(POLICY CMP0148)
    cmake_policy(SET CMP0148 NEW)
  endif()

  # First try to find system-installed pybind11
  find_package(pybind11 CONFIG QUIET)

  if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found in system, fetching from GitHub")
    include(FetchContent)

    FetchContent_Declare(
      pybind11
      GIT_REPOSITORY https://github.com/pybind/pybind11
      GIT_TAG v2.11.1
      GIT_SHALLOW TRUE
    )

    FetchContent_MakeAvailable(pybind11)
  else()
    message(STATUS "Using system pybind11: ${pybind11_VERSION}")
  endif()

  # Build the Python module
  # Note: target name is parthenon_py to avoid collision with main library,
  # but OUTPUT_NAME is parthenon so Python imports as "import parthenon"
  pybind11_add_module(parthenon_py parameter_input_bindings.cpp)
  set_target_properties(parthenon_py PROPERTIES
    OUTPUT_NAME parthenon
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/python
  )

  # Link against parthenon library (use :: to avoid ambiguity)
  target_link_libraries(parthenon_py PRIVATE parthenon)

  # Create symlinks to Python packages in build/lib/python for convenient PYTHONPATH
  # This allows a single PYTHONPATH entry: export PYTHONPATH=build/lib/python:$PYTHONPATH
  set(PYTHON_LIB_DIR ${CMAKE_BINARY_DIR}/lib/python)
  set(PYTHON_PACKAGES_DIR ${CMAKE_SOURCE_DIR}/scripts/python/packages)

  # Create lib/python directory if it doesn't exist
  file(MAKE_DIRECTORY ${PYTHON_LIB_DIR})

  # Symlink parthenon_input and parthenon_tools packages
  add_custom_command(TARGET parthenon_py POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink
      ${PYTHON_PACKAGES_DIR}/parthenon_input
      ${PYTHON_LIB_DIR}/parthenon_input
    COMMAND ${CMAKE_COMMAND} -E create_symlink
      ${PYTHON_PACKAGES_DIR}/parthenon_tools
      ${PYTHON_LIB_DIR}/parthenon_tools
    COMMENT "Creating symlinks to Python packages in ${PYTHON_LIB_DIR}"
  )

  # Install the Python module to a standard location
  install(TARGETS parthenon_py
          LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/python)

  message(STATUS "Python bindings enabled and will be built")
  message(STATUS "Python module and packages will be in: ${PYTHON_LIB_DIR}")
  message(STATUS "Add to PYTHONPATH: export PYTHONPATH=${CMAKE_BINARY_DIR}/lib/python:\$PYTHONPATH")
else()
  message(STATUS "Python bindings disabled (use -DPARTHENON_ENABLE_PYTHON_BINDINGS=ON to enable)")
endif()
