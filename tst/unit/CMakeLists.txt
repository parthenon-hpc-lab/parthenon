##========================================================================================
## Athena++ astrophysical MHD code
## Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
## Licensed under the 3-clause BSD License, see LICENSE file for details
##========================================================================================
## (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
##
## This program was produced under U.S. Government contract 89233218CNA000001 for Los
## Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
## for the U.S. Department of Energy/National Nuclear Security Administration. All rights
## in the program are reserved by Triad National Security, LLC, and the U.S. Department
## of Energy/National Nuclear Security Administration. The Government is granted for
## itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
## license in this material to reproduce, prepare derivative works, distribute copies to
## the public, perform publicly and display publicly, and to permit others to do so.
##========================================================================================

list(APPEND unit_tests_SOURCES test_unit_params.cpp kokkos_abstraction.cpp)

add_executable(unit_tests ${unit_tests_SOURCES})
target_link_libraries(unit_tests PRIVATE parthenon catch2_define kokkos)

set(EXCEPTION_HANDLING_OPTION ENABLE_EXCEPTIONS) # TODO: Add option to disable exceptions
set(COMPILED_WITH ${CMAKE_CXX_COMPILER})
set(COMPILER_COMMAND "<not-implemented>") # TODO: Put something more descriptive here
set(COMPILER_FLAGS "<not-implemented>") # TODO: Put something more descriptive here
set(NHYDRO_VARIABLES 0) # TODO: Remove
set(NFIELD_VARIABLES 0) # TODO: Remove
set(NWAVE_VALUE 5) # TODO: Remove
set(NUMBER_GHOST_CELLS 2) # TODO: Make this a CMake option and check it

configure_file(../../src/defs.hpp.in generated/defs.hpp @ONLY)

target_include_directories(unit_tests PUBLIC
  ${CMAKE_CURRENT_BINARY_DIR}/generated
)

ParseAndAddCatchTests(unit_tests)



