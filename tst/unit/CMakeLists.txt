
##========================================================================================
## Parthenon performance portable AMR framework
## Copyright(C) 2020-2022 The Parthenon collaboration
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
    test_concepts_lite.cpp    
    test_data_collection.cpp
    test_taskid.cpp
    test_tasklist.cpp
    test_unit_params.cpp
    test_unit_constants.cpp
    test_unit_domain.cpp
    test_unit_sort.cpp
    kokkos_abstraction.cpp
    test_index_split.cpp
    test_logical_location.cpp
    test_forest.cpp
    test_metadata.cpp
    test_meshblock_data_iterator.cpp
    test_mesh_data.cpp
    test_nan_tags.cpp
    test_output_utils.cpp
    test_pararrays.cpp
    test_sparse_pack.cpp
    test_swarm.cpp
    test_required_desired.cpp
    test_error_checking.cpp
    test_partitioning.cpp
    test_state_descriptor.cpp
    test_unit_integrators.cpp
    test_upper_bound.cpp
)

add_executable(unit_tests "${unit_tests_SOURCES}")

target_link_libraries(unit_tests PRIVATE Parthenon::parthenon catch2_define Kokkos::kokkos)

if (TEST_ERROR_CHECKING)
   message(WARNING "\tTesting error checking. This test will FAIL.")
   target_compile_definitions(unit_tests PRIVATE PARTHENON_TEST_ERROR_CHECKING)
endif()

lint_target(unit_tests)

# separating test cases here because performance tests should (later) only be run
# for Release build types.
catch_discover_tests(unit_tests TEST_SPEC "~[performance]" PROPERTIES LABELS "unit")
catch_discover_tests(unit_tests TEST_SPEC "[performance]" PROPERTIES LABELS "performance")

get_property(ALL_TESTS DIRECTORY . PROPERTY TESTS)

add_coverage_label("${ALL_TESTS}" "performance")
