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

list(APPEND unit_tests_SOURCES test_unit_params.cpp wrappers.cpp)

add_executable(unit_tests ${unit_tests_SOURCES})
target_link_libraries(unit_tests PRIVATE parthenon catch2_define kokkos)

# TODO(pgrete) This is duplicated code (from src/CMakeLists.txt) to set the
# defalt variables. Any suggestion on how properly solve this without code duplications
# are welcome! Any cmake expers?

set(PROBLEM_GENERATOR "<not-implemented>") # TODO: Figure out what to put here
set(SINGLE_PRECISION_ENABLED 0) # TODO: Add CMake option for this
set(MPI_OPTION MPI_PARALLEL) # TODO: Without MPI
set(OPENMP_OPTION OPENMP_PARALLEL) # TODO: Without OPENMP
set(HDF5_OPTION HDF5OUTPUT) # TODO: No HDF5

if (${Kokkos_ENABLE_CUDA})
  set(PAR_LOOP_LAYOUT "MANUAL1D_LOOP")
elseif(${Kokkos_ENABLE_HPX})
  message( FATAL_ERROR "Need to add/fix/test default loop layouts for HPX backend.")
else()
  set(PAR_LOOP_LAYOUT "SIMDFOR_LOOP")
endif()
set(PAR_LOOP_LAYOUT_INNER INNER_TTR)
set(PAR_VECTOR_LENGTH 1)

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



