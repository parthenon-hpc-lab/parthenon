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

# Don't include machine cfg more than once
if (MACHINE_CFG_FULL)
  return()
endif()

# Load machine specific defaults such as architecture and mpi launch command.
# Command line argument takes precedence over environment variable.
# Loading this before project definition to allow setting the compiler.
if (MACHINE_CFG)
  set(USE_MACHINE_CFG ${MACHINE_CFG})
elseif (DEFINED ENV{MACHINE_CFG})
  set(USE_MACHINE_CFG $ENV{MACHINE_CFG})
endif()

if (USE_MACHINE_CFG)
  # Get the Parthenon root directory, which is one directory up from
  # from the "cmake" directory. This is done in-case MachineCfg.cmake is
  # included from a project using Parthenon.
  get_filename_component(PAR_ROOT ${CMAKE_CURRENT_LIST_DIR} DIRECTORY)

  # The `@PAR_ROOT@` string allows the devleper to specify a relative path to
  # the parthenon root directory via an environment variable.
  string(
    REPLACE "@PAR_ROOT@" ${PAR_ROOT} USE_MACHINE_CFG ${USE_MACHINE_CFG})

  # Resolve path to absolute and cache it
  get_filename_component(MACHINE_CFG_FULL ${USE_MACHINE_CFG} ABSOLUTE)

  if (NOT EXISTS ${MACHINE_CFG_FULL})
    message(FATAL_ERROR
      "Given machine configuration at ${MACHINE_CFG_FULL} not found.")
  endif()

  # Cache this in-case of environment variable being unset later
  set(MACHINE_CFG_FULL ${MACHINE_CFG_FULL} CACHE INTERNAL "Full path to machine configuration")
endif()

if (MACHINE_CFG_FULL)
  if (CMAKE_PROJECT_NAME)
    message(WARNING "MachineCfg.cmake should be included before calling the \
      project() function. If you're using parthenon as a subdirectory, try \
      adding an include for it in your top level CMakeLists.txt prior to \
      calling project().")
  endif()

  include(${MACHINE_CFG_FULL})
else()
  message(WARNING "Not using any machine configuration. Consider creating a configuration "
    "file following the examples in ${PROJECT_SOURCE_DIR}/cmake/machine_cfgs/ and then "
    "point the MACHINE_CFG cmake or environment variable to your custom file."
    "Note, that the machine file can be placed in any directory (also outside the repo).")
endif()
