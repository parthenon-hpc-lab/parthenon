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

add_custom_target(coverage 
  COMMAND mkdir -p coverage
  COMMAND ${CMAKE_MAKE_PROGRAM} test
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )

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

#  SET(GCC_COVERAGE_COMPILE_FLAGS "-fprofile-arcs -ftest-coverage")
#  SET(GCC_COVERAGE_LINK_FLAGS    "-lgcov")
#
#  SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")
#  SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${GCC_COVERAGE_LINK_FLAGS}")
 
  set(OBJECT_DIR ${CMAKE_BINARY_DIR}/obj)
  get_target_property(PARTHENON_SOURCES parthenon SOURCES)
  get_target_property(UNIT_TEST_SOURCES unit_tests SOURCES)

  message(${CMAKE_SOURCE_DIR})
  add_custom_command(TARGET coverage
    COMMAND echo "====================== Code Coverage ======================"
    COMMAND ${PATH_LCOV} --version
    # Clean
    COMMAND ${PATH_LCOV} --gcov-tool ${PATH_GCOV} --directory ${CMAKE_BINARY_DIR} -b ${CMAKE_SOURCE_DIR} --zerocounters
    # Base report
    COMMAND ${PATH_LCOV} --gcov-tool ${PATH_GCOV} -c -i -d ${CMAKE_BINARY_DIR} -b ${CMAKE_SOURCE_DIR} -o ${CMAKE_BINARY_DIR}/coverage/report.base.old
    # Remove Kokkos info from code coverage
    COMMAND ${PATH_LCOV} --remove ${CMAKE_BINARY_DIR}/coverage/report.base.old 'Kokkos/*' -o ${CMAKE_BINARY_DIR}/coverage/report.base
    # Capture information from test runs
    COMMAND ${PATH_LCOV} --gcov-tool ${PATH_GCOV} --directory ${CMAKE_BINARY_DIR} -b ${CMAKE_SOURCE_DIR} --capture --output-file ${CMAKE_BINARY_DIR}/coverage/report.test.old
    # Remove Kokkos info from code coverage
    COMMAND ${PATH_LCOV} --remove ${CMAKE_BINARY_DIR}/coverage/report.test.old 'Kokkos/*' -o ${CMAKE_BINARY_DIR}/coverage/test.base
    # Combining base line counters with counters from running tests
    COMMAND ${PATH_LCOV} --gcov-tool ${PATH_GCOV} -a {CMAKE_BINARY_DIR}/coverage/report.base -a {CMAKE_BINARY_DIR}/coverage/report.test --output-file ${CMAKE_BINARY_DIR}/coverage/report.all
    #COMMENT "Coverage files have been output to ${CMAKE_BINARY_DIR}/coverage"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
 
#  add_custom_command(TARGET coverage
#    COMMAND echo "====================== Code Coverage ======================"
#    COMMAND gcov -b ${PARTHENON_SOURCES};${UNIT_TEST_SOURCES} -o ${OBJECT_DIR} 
#    COMMAND echo "-- Coverage files have been output to ${CMAKE_BINARY_DIR}/coverage"
#    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/coverage
#    )
# 
  if(ENABLE_UNIT_TESTS)
    add_dependencies(coverage unit_tests)
  endif()

else()
  add_custom_command(TARGET coverage
    COMMAND echo "====================== Code Coverage ======================"
    COMMENT "Code coverage has not been enabled"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/coverage
    )
endif()


