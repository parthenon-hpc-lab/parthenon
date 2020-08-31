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
cmake_minimum_required(VERSION 3.12)
project(parthenon_examples)

if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  # Get Parthenon if this is the top level directory

  option(PARTHENON_EXAMPLE_FIND_PARTHENON "ON for find_package(parthenon), OFF for add_subdirectory(.. parthenon" OFF)

  if(NOT TARGET Parthenon::parthenon)
    if (PARTHENON_EXAMPLE_FIND_PARTHENON)
      find_package(parthenon REQUIRED)
    else()
      add_subdirectory(.. parthenon)
    endif()
  endif()
endif()

add_subdirectory(advection)
add_subdirectory(calculate_pi)
add_subdirectory(face_fields)
add_subdirectory(kokkos_pi)
