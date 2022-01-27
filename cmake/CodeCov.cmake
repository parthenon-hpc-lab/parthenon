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

# Function will add the coverage label to all tests provided
# The function can be called by invoking
#
# list(APPEND all_tests test1 test2)
#
# add_coverage_label("${all_tests}")
#
# You also have the option of excluding tests with specific labels if desired
# by passing a second argument
#
# list(APPEND all_tests test1 test2)
# list(APPEND exclude_tests_with_these_labels "performance;CGS")
#
# add_coverage_label("${all_tests}" "${exclude_tests_with_these_labels}")
#
# This will only add the coverage label to tests that do not contain the
# performance and CGS labels
function(add_coverage_label tests )
  if( CODE_COVERAGE )
    foreach( CHECK_TEST ${tests})
      set(exclude FALSE)
      get_test_property(${CHECK_TEST} LABELS TEST_LABELS)
      foreach( exclude_if_contains_this_label ${ARGN})
        if( ${exclude_if_contains_this_label} IN_LIST TEST_LABELS)
          set(exclude TRUE)
          continue()
        endif()
      endforeach()
      if(${exclude})
        continue()
      endif()
      set_property(TEST "${CHECK_TEST}" APPEND PROPERTY LABELS "coverage")
    endforeach()
  endif()
endfunction()

# This function creates the code coverage targets for parthenon:
#
# coverage - this target will build the coverage reports by running all tests
#            with the coverage label.
#
# make coverage
#
# coverage-upload - this target will upload the reports to code cov for remote viewing
#
# make coverage-upload
#
# The 'create_parthenon_coverage_targets' should only be called after the parthenon
# library has been defined.
function(create_pathenon_coverage_targets)
  if(CODE_COVERAGE)

    find_program( PATH_GCOV gcov )
    find_program( PATH_LCOV lcov )

    if(NOT PATH_GCOV)
        message(FATAL_ERROR "Unable to build with code coverage gcov was not found.")
    endif()

    if(NOT PATH_LCOV)
        message(FATAL_ERROR "Unable to build with code coverage lcov was not found.")
    endif()

    if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
        message(WARNING "Using code coverage with an optimized build is discouraged, as it may lead to misleading results.")
    endif()

    message(STATUS "Coverage reports will be placed in ${COVERAGE_PATH}/${COVERAGE_NAME}")

    get_target_property(PARTHENON_SOURCES parthenon SOURCES)
    get_target_property(UNIT_TEST_SOURCES unit_tests SOURCES)

    add_custom_target(coverage)
    add_custom_command(TARGET coverage

      COMMAND echo "====================== Code Coverage ======================"
      COMMAND mkdir -p ${COVERAGE_PATH}/${COVERAGE_NAME}
      COMMAND ${PATH_LCOV} --version
      # Clean
      COMMAND ${PATH_LCOV} --gcov-tool ${PATH_GCOV} --directory ${PROJECT_BINARY_DIR} -b ${PROJECT_SOURCE_DIR} --zerocounters
      # Base report
      COMMAND ctest -L coverage --verbose --timeout 7200
      COMMAND ${PATH_LCOV} --gcov-tool ${PATH_GCOV} -c -i -d ${PROJECT_BINARY_DIR} -b ${PROJECT_SOURCE_DIR} -o ${COVERAGE_PATH}/${COVERAGE_NAME}/report.base.old
      # Remove Kokkos and tst info from code coverage
      COMMAND ${PATH_LCOV} --remove ${COVERAGE_PATH}/${COVERAGE_NAME}/report.base.old "*/Kokkos/*" "*/tst/*" -o ${COVERAGE_PATH}/${COVERAGE_NAME}/report.base

      # Capture information from test runs
      COMMAND ${PATH_LCOV} --gcov-tool ${PATH_GCOV} --directory ${PROJECT_BINARY_DIR} -b ${PROJECT_SOURCE_DIR} --capture --output-file ${COVERAGE_PATH}/${COVERAGE_NAME}/report.test.old
      # Remove Kokkos and tst info from code coverage
      COMMAND ${PATH_LCOV} --remove ${COVERAGE_PATH}/${COVERAGE_NAME}/report.test.old "*/Kokkos/*" "*/tst/*" -o ${COVERAGE_PATH}/${COVERAGE_NAME}/report.test

      # Combining base line counters with counters from running tests
      COMMAND ${PATH_LCOV} --gcov-tool ${PATH_GCOV} -a ${COVERAGE_PATH}/${COVERAGE_NAME}/report.base -a ${COVERAGE_PATH}/${COVERAGE_NAME}/report.test --output-file ${COVERAGE_PATH}/${COVERAGE_NAME}/report.all
      # Remove unneeded reports
      COMMAND rm ${COVERAGE_PATH}/${COVERAGE_NAME}/report.test.old
      COMMAND rm ${COVERAGE_PATH}/${COVERAGE_NAME}/report.base.old
      COMMAND rm ${COVERAGE_PATH}/${COVERAGE_NAME}/report.base
      COMMAND rm ${COVERAGE_PATH}/${COVERAGE_NAME}/report.test
      WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
      )

    add_custom_target(coverage-upload)
    add_custom_command(TARGET coverage-upload
      COMMAND echo "================ Uploading Code Coverage =================="
      # Upload coverage report
      COMMAND ${parthenon_SOURCE_DIR}/scripts/combine_coverage.sh ${PATH_LCOV} ${PATH_GCOV} ${COVERAGE_PATH}
      COMMAND ${CMAKE_COMMAND} -DCOVERAGE_PATH=${COVERAGE_PATH} -P "${parthenon_SOURCE_DIR}/cmake/CodeCovBashDownloadScript.cmake"
      COMMAND cat ${COVERAGE_PATH}/CombinedCoverage/script.coverage
      COMMAND cd ${COVERAGE_PATH}/CombinedCoverage && bash ${COVERAGE_PATH}/CombinedCoverage/script.coverage -p ${PROJECT_BINARY_DIR} -s ${COVERAGE_PATH}/CombinedCoverage
      WORKING_DIRECTORY ${COVERAGE_PATH}
      )

    if(ENABLE_UNIT_TESTS)
      add_dependencies(coverage unit_tests)
    endif()

  else()
    add_custom_target(coverage)
    add_custom_command(TARGET coverage
      COMMAND echo "====================== Code Coverage ======================"
      COMMENT "Code coverage has not been enabled"
      )
  endif()
endfunction()


