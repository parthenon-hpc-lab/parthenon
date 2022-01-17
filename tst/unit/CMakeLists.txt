
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
    test_data_collection.cpp
    test_taskid.cpp
    test_tasklist.cpp
    test_unit_face_variables.cpp
    test_unit_params.cpp
    test_unit_constants.cpp
    test_unit_domain.cpp
    test_unit_sort.cpp
    kokkos_abstraction.cpp
    test_metadata.cpp
    test_pararrays.cpp
    test_meshblock_data_iterator.cpp
    test_mesh_data.cpp
    test_swarm.cpp
    test_required_desired.cpp
    test_error_checking.cpp
    test_partitioning.cpp
    test_state_descriptor.cpp
)

add_executable(unit_tests "${unit_tests_SOURCES}")

target_link_libraries(unit_tests PRIVATE Parthenon::parthenon catch2_define Kokkos::kokkos)

if (TEST_ERROR_CHECKING)
   message(WARNING "\tTesting error checking. This test will FAIL.")
   target_compile_definitions(unit_tests PRIVATE PARTHENON_TEST_ERROR_CHECKING)
endif()

lint_target(unit_tests)

ParseAndAddCatchTests(unit_tests)

get_property(ALL_TESTS DIRECTORY . PROPERTY TESTS)

add_coverage_label("${ALL_TESTS}" "performance")
