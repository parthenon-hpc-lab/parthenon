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

# Adds the drivers used in the regression tests to a global cmake property: DRIVERS_USED_IN_TESTS
function(record_driver arg)
    list(LENGTH arg len_list)
    math(EXPR list_end "${len_list} - 1")
    foreach(ind RANGE ${list_end})
      list(GET arg ${ind} arg2)
      if("${arg2}" STREQUAL "--driver")
        MATH(EXPR ind "${ind}+1")
        list(GET arg ${ind} driver)
        get_filename_component(driver ${driver} NAME)
        set_property(GLOBAL APPEND PROPERTY DRIVERS_USED_IN_TESTS "${driver}" )
      endif()
    endforeach()
endfunction()

# Adds test that will run in serial
# test output will be sent to /tst/regression/outputs/dir
# test property labels: regression, mpi-no
function(setup_test dir arg)
  separate_arguments(arg) 
  add_test( NAME regression_test:${dir} COMMAND python "${CMAKE_CURRENT_SOURCE_DIR}/run_test.py" 
    ${arg} --test_dir "${CMAKE_CURRENT_SOURCE_DIR}/test_suites/${dir}"
    --output_dir "${CMAKE_BINARY_DIR}/tst/regression/outputs/${dir}")
  set_tests_properties(regression_test:${dir} PROPERTIES LABELS "regression;mpi-no" )
  record_driver("${arg}")
endfunction()

# Adds test that will run in serial with code coverage
# test output will be sent to /tst/regression/outputs/dir_cov
# test property labels: regression, mpi-no; coverage
function(setup_test_coverage dir arg)
  separate_arguments(arg) 
  add_test( NAME regression_coverage_test:${dir} COMMAND python "${CMAKE_CURRENT_SOURCE_DIR}/run_test.py" 
    ${arg} 
    --coverage
    --test_dir "${CMAKE_CURRENT_SOURCE_DIR}/test_suites/${dir}"
    --output_dir "${CMAKE_BINARY_DIR}/tst/regression/outputs/${dir}_cov")
  set_tests_properties(regression_coverage_test:${dir} PROPERTIES LABELS "regression;coverage;mpi-no" )
  record_driver("${arg}")
endfunction()

# Adds test that will run in parallel with mpi
# test output will be sent to /tst/regression/outputs/dir_mpi
# test property labels: regression, mpi-yes
function(setup_test_mpi nproc dir arg)
  if( MPI_FOUND )
    separate_arguments(arg) 
    add_test( NAME regression_mpi_test:${dir} COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/run_test.py
      --mpirun ${MPIEXEC_EXECUTABLE} 
      --mpirun_opts=${MPIEXEC_NUMPROC_FLAG} --mpirun_opts=${nproc}
      --mpirun_opts=${MPIEXEC_PREFLAGS} ${arg}
      --test_dir ${CMAKE_CURRENT_SOURCE_DIR}/test_suites/${dir}
      --output_dir "${CMAKE_BINARY_DIR}/tst/regression/outputs/${dir}_mpi")
    set_tests_properties(regression_mpi_test:${dir} PROPERTIES LABELS "regression;mpi-yes" RUN_SERIAL ON )
    record_driver("${arg}")
  else()
    message(STATUS "MPI not found, not building regression tests with mpi")
  endif()
endfunction()

# Adds test that will run in parallel with mpi and code coverage
# test output will be sent to /tst/regression/outputs/dir_mpi_cov
# test property labels: regression, mpi-yes, coverage
function(setup_test_mpi_coverage nproc dir arg)
  if( MPI_FOUND )
    separate_arguments(arg) 
    add_test( NAME regression_mpi_coverage_test:${dir} COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/run_test.py
      --coverage
      --mpirun ${MPIEXEC_EXECUTABLE} 
      --mpirun_opts=${MPIEXEC_NUMPROC_FLAG} --mpirun_opts=${nproc}
      --mpirun_opts=${MPIEXEC_PREFLAGS} ${arg}
      --test_dir ${CMAKE_CURRENT_SOURCE_DIR}/test_suites/${dir}
      --output_dir "${CMAKE_BINARY_DIR}/tst/regression/outputs/${dir}_mpi_cov"
      )
    set_tests_properties(regression_mpi_coverage_test:${dir} PROPERTIES LABELS "regression;coverage;mpi-yes" RUN_SERIAL ON )
  else()
    message(STATUS "MPI not found, not building coverage regression tests with mpi")
  endif()
endfunction()


