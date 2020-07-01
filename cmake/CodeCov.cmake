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
 
  set(OBJECT_DIR ${CMAKE_BINARY_DIR}/obj)
  get_target_property(PARTHENON_SOURCES parthenon SOURCES)
  get_target_property(UNIT_TEST_SOURCES unit_tests SOURCES)

  add_custom_target(coverage)
  add_custom_command(TARGET coverage

    COMMAND echo "====================== Code Coverage ======================"
    COMMAND mkdir -p ${COVERAGE_PATH}/${COVERAGE_NAME}
    COMMAND ${PATH_LCOV} --version
    # Clean
    COMMAND ${PATH_LCOV} --gcov-tool ${PATH_GCOV} --directory ${CMAKE_BINARY_DIR} -b ${CMAKE_SOURCE_DIR} --zerocounters
    # Base report
    COMMAND ctest -LE performance -R unit --verbose
    COMMAND ${PATH_LCOV} --gcov-tool ${PATH_GCOV} -c -i -d ${CMAKE_BINARY_DIR} -b ${CMAKE_SOURCE_DIR} -o ${COVERAGE_PATH}/${COVERAGE_NAME}/report.base.old
    # Remove Kokkos info from code coverage
    COMMAND ${PATH_LCOV} --remove ${COVERAGE_PATH}/${COVERAGE_NAME}/report.base.old 'Kokkos/*' -o ${COVERAGE_PATH}/${COVERAGE_NAME}/report.base
    # Capture information from test runs
    COMMAND ${PATH_LCOV} --gcov-tool ${PATH_GCOV} --directory ${CMAKE_BINARY_DIR} -b ${CMAKE_SOURCE_DIR} --capture --output-file ${COVERAGE_PATH}/${COVERAGE_NAME}/report.test.old
    # Remove Kokkos info from code coverage
    COMMAND ${PATH_LCOV} --remove ${COVERAGE_PATH}/${COVERAGE_NAME}/report.test.old 'Kokkos/*' -o ${COVERAGE_PATH}/${COVERAGE_NAME}/report.test
    # Combining base line counters with counters from running tests
    COMMAND ${PATH_LCOV} --gcov-tool ${PATH_GCOV} -a ${COVERAGE_PATH}/${COVERAGE_NAME}/report.base -a ${COVERAGE_PATH}/${COVERAGE_NAME}/report.test --output-file ${COVERAGE_PATH}/${COVERAGE_NAME}/report.all
    # Remove unneeded reports
    COMMAND rm ${COVERAGE_PATH}/${COVERAGE_NAME}/report.test.old
    COMMAND rm ${COVERAGE_PATH}/${COVERAGE_NAME}/report.base.old
    COMMAND rm ${COVERAGE_PATH}/${COVERAGE_NAME}/report.base
    COMMAND rm ${COVERAGE_PATH}/${COVERAGE_NAME}/report.test
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )

  set(UPLOAD_COMMAND "bash <\(curl -s https://codecov.io/bash\) \|\| echo \"code coverage failed to upload\"")
  add_custom_target(coverage-upload) 
  add_custom_command(TARGET coverage-upload
    COMMAND echo "================ Uploading Code Coverage =================="
    # Upload coverage report
    COMMAND ${CMAKE_SOURCE_DIR}/scripts/combine_coverage.sh ${PATH_LCOV} ${PATH_GCOV} ${COVERAGE_PATH}
    COMMAND curl -s https://codecov.io/bash > ${COVERAGE_PATH}/CombinedCoverage/script.coverage
    COMMAND cat ${COVERAGE_PATH}/CombinedCoverage/script.coverage
    COMMAND cd ${COVERAGE_PATH}/CombinedCoverage && bash ${COVERAGE_PATH}/CombinedCoverage/script.coverage -p ${CMAKE_BINARY_DIR} -s ${COVERAGE_PATH}/CombinedCoverage
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


