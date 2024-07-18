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



# Search for the python interpreter
# Version number has been intentionally excluded from find_package call, so that latest version 
# will be grabbed. Including the version number would prioritise the version provided over more 
#
if( NOT Python3_Interpreter_FOUND)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
endif()
if(Python3_Interpreter_FOUND)
  if( "${Python3_VERSION}" VERSION_LESS "3.5")
    message(FATAL_ERROR "Python version requirements not satisfied for running regression tests.")
  endif()
endif()

if (PARTHENON_ENABLE_PYTHON_MODULE_CHECK)
  # Ensure all required packages are present
  include(${parthenon_SOURCE_DIR}/cmake/PythonModuleCheck.cmake)
  python_modules_found("${REQUIRED_PYTHON_MODULES}" "${DESIRED_PYTHON_MODULES}")
endif()

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

# If binaries cannot be run in serial without mpiexec or the equivalent this function allows
# the specification of the appropriate flags.
# By default the function will populate the arguments with the appropriate mpi flags that
# are registered from the find_package command. They can however be overwritten with 
# TEST_MPIEXEC and TEST_MPIEXEC_FLAG if needed. 
function(process_args_serial_with_mpi)
  if ( ${TEST_MPIEXEC} ) 
    set(TMPARGS "--mpirun=${TEST_MPIEXEC}")
  else()
    set(TMPARGS "--mpirun=${MPIEXEC_EXECUTABLE}")
  endif()
  if ( ${TEST_MPIEXEC_FLAG} )
    list(APPEND TMPARGS "--mpirun_ranks_flag=${TEST_MPIEXEC_FLAG}")
  else()
    list(APPEND TMPARGS "--mpirun_ranks_flag=${MPIEXEC_NUMPROC_FLAG}")
  endif()
  list(APPEND TMPARGS "--mpirun_ranks_num=1")
  
  # make the result accessible in the calling function
  set(SUFFIX_SERIAL_REGRESSION_TEST ${TMPARGS} PARENT_SCOPE)
endfunction()

function(process_mpi_args nproc)
  list(APPEND TMPARGS "--mpirun")
  # use custom mpiexec
  if (TEST_MPIEXEC)
    list(APPEND TMPARGS "${TEST_MPIEXEC}")
  # use CMake determined mpiexec
  else()
    list(APPEND TMPARGS "${MPIEXEC_EXECUTABLE}")
  endif()
  # use custom numproc flag
  if (TEST_NUMPROC_FLAG)
    list(APPEND TMPARGS "--mpirun_ranks_flag=${TEST_NUMPROC_FLAG}")
  # use CMake determined numproc flag
  else()
    list(APPEND TMPARGS "--mpirun_ranks_flag=${MPIEXEC_NUMPROC_FLAG}")
  endif()
  list(APPEND TMPARGS "--mpirun_ranks_num=${nproc}")
  # set additional options from machine configuration
  foreach(MPIARG ${TEST_MPIOPTS})
    list(APPEND TMPARGS "--mpirun_opts=${MPIARG}")
  endforeach()

  # make the result accessible in the calling function
  set(MPIARGS ${TMPARGS} PARENT_SCOPE)
endfunction()

# Adds test that will run in serial
# test output will be sent to /tst/regression/outputs/dir
# test property labels: regression, serial
function(setup_test_serial dir arg extra_labels)
  separate_arguments(arg) 
  list(APPEND labels "regression;serial")
  list(APPEND labels "${extra_labels}")
  if (Kokkos_ENABLE_OPENMP)
    set(PARTHENON_KOKKOS_TEST_ARGS "${PARTHENON_KOKKOS_TEST_ARGS} --kokkos-threads=${NUM_OMP_THREADS_PER_RANK}")
  endif()
  if (SERIAL_WITH_MPIEXEC)
    process_args_serial_with_mpi()
  endif()
  add_test(
    NAME regression_test:${dir}
    COMMAND ${Python3_EXECUTABLE} "${parthenon_SOURCE_DIR}/tst/regression/run_test.py"
      ${arg} --test_dir "${CMAKE_CURRENT_SOURCE_DIR}/test_suites/${dir}"
      --output_dir "${PROJECT_BINARY_DIR}/tst/regression/outputs/${dir}"
      --kokkos_args=${PARTHENON_KOKKOS_TEST_ARGS}
      ${SUFFIX_SERIAL_REGRESSION_TEST})
  set_tests_properties(regression_test:${dir} PROPERTIES LABELS "${labels}" )
  record_driver("${arg}")
endfunction()

# Adds test that will run in serial with code coverage
# test output will be sent to /tst/regression/outputs/dir_cov
# test property labels: regression, serial; coverage
function(setup_test_coverage dir arg extra_labels)
  if( CODE_COVERAGE )
    separate_arguments(arg) 

    list(APPEND labels "regression;coverage;serial")
    list(APPEND labels "${extra_labels}")
    if (SERIAL_WITH_MPIEXEC)
      process_args_serial_with_mpi()
    endif()
    add_test( NAME regression_coverage_test:${dir} COMMAND ${Python3_EXECUTABLE} "${parthenon_SOURCE_DIR}/tst/regression/run_test.py"
      ${arg} 
      --coverage
      --test_dir "${CMAKE_CURRENT_SOURCE_DIR}/test_suites/${dir}"
      --output_dir "${PROJECT_BINARY_DIR}/tst/regression/outputs/${dir}_cov"
      ${SUFFIX_SERIAL_REGRESSION_TEST})
    set_tests_properties(regression_coverage_test:${dir} PROPERTIES LABELS "${labels}" )
    record_driver("${arg}")
  endif()
endfunction()

# Adds test that will run in parallel with mpi
# test output will be sent to /tst/regression/outputs/dir_mpi
# test property labels: regression, mpi
function(setup_test_parallel nproc dir arg extra_labels)
  if( MPI_FOUND )
    separate_arguments(arg) 
    list(APPEND labels "regression;mpi")
    list(APPEND labels "${extra_labels}")

    if(Kokkos_ENABLE_CUDA OR Kokkos_ENABLE_HIP)
      set(PARTHENON_KOKKOS_TEST_ARGS "--kokkos-map-device-id-by=mpi_rank")
      list(APPEND labels "cuda")
    endif()
    if (Kokkos_ENABLE_OPENMP)
      set(PARTHENON_KOKKOS_TEST_ARGS "${PARTHENON_KOKKOS_TEST_ARGS} --kokkos-threads=${NUM_OMP_THREADS_PER_RANK}")
    endif()
    process_mpi_args(${nproc})
    add_test( NAME regression_mpi_test:${dir} COMMAND ${Python3_EXECUTABLE} ${parthenon_SOURCE_DIR}/tst/regression/run_test.py
      ${MPIARGS} ${arg}
      --test_dir ${CMAKE_CURRENT_SOURCE_DIR}/test_suites/${dir}
      --output_dir "${PROJECT_BINARY_DIR}/tst/regression/outputs/${dir}_mpi"
      --kokkos_args=${PARTHENON_KOKKOS_TEST_ARGS})

    # When targeting CUDA we don't have a great way of controlling how tests
    # get mapped to GPUs, so just enforce serial execution
    if (Kokkos_ENABLE_CUDA OR Kokkos_ENABLE_HIP)
      set(TEST_PROPERTIES
        RUN_SERIAL ON)
    else()
      set(TEST_PROPERTIES
        PROCESSOR_AFFINITY ON
        PROCESSORS ${nproc})
    endif()
    set_tests_properties(
      regression_mpi_test:${dir}
      PROPERTIES
        LABELS "${labels}"
        ${TEST_PROPERTIES})
    record_driver("${arg}")
  else()
    message(STATUS "TestSetup for parallel regression tests: MPI not found, not building regression tests with mpi."
      "To enable parallel regression tests ensure to include MPI in your project.")
  endif()
endfunction()

# Adds test that will run in parallel with mpi and code coverage
# test output will be sent to /tst/regression/outputs/dir_mpi_cov
# test property labels: regression, mpi, coverage
function(setup_test_mpi_coverage nproc dir arg extra_labels)
  if( MPI_FOUND )
    if( CODE_COVERAGE )

      list(APPEND labels "regression;coverage;mpi")
      list(APPEND labels "${extra_labels}")
      separate_arguments(arg) 
      process_mpi_args(${nproc})
      add_test( NAME regression_mpi_coverage_test:${dir} COMMAND ${Python3_EXECUTABLE} ${parthenon_SOURCE_DIR}/tst/regression/run_test.py
        --coverage
        ${MPIARGS} ${arg}
        --test_dir ${CMAKE_CURRENT_SOURCE_DIR}/test_suites/${dir}
        --output_dir "${PROJECT_BINARY_DIR}/tst/regression/outputs/${dir}_mpi_cov"
        )
      set_tests_properties(regression_mpi_coverage_test:${dir} PROPERTIES LABELS "${labels}" RUN_SERIAL ON )
    endif()
  else()
    message(STATUS "MPI not found, not building coverage regression tests with mpi")
  endif()
endfunction()

