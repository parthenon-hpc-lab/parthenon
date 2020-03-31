
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

list(APPEND unit_tests_SOURCES 

    test_taskid.cpp
    test_unit_face_variables.cpp
    test_unit_params.cpp
    kokkos_abstraction.cpp
    test_metadata.cpp
    test_pararrays.cpp

    )

# TODO(JMM): Should these be global to the build system?
option(CHECK_REGISTRY_PRESSURE "Check the registry pressure for Kokkos CUDA kernels" OFF)
if (${Kokkos_ENABLE_CUDA})
  if(CHECK_REGISTRY_PRESSURE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xptxas=-v")
  endif()
endif()
option(TEST_INTEL_OPTIMIZATION "Test intel optimization and vectorization" OFF)
if (TEST_INTEL_OPTIMIZATION)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fp-model fast=2 -qopt_report5 -vec-threshold0 -qopt_report_phase=vec")
endif()	       

add_executable(unit_tests ${unit_tests_SOURCES})
target_compile_features(unit_tests PRIVATE cxx_std_14)
target_link_libraries(unit_tests PRIVATE parthenon catch2_define Kokkos::kokkos)

ParseAndAddCatchTests(unit_tests)
