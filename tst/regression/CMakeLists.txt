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

# Adding regression tests
#
# To add additional regression tests to ctest, to things must be added to 
# this file:
# 
# 1. there must exist within the regression/test_suites directory a folder associated with the test
#   * For the calculate_pi example the folder is called 'calculate_pi', This
#     folder must be added to the cmake variable TEST_DIRS as shown below. 

# 2. the correct arguments to pass to the 'run_test.py' script must be specfied
#   * This arguments must be added to the CMAKE variable TEST_ARGS as shown below
#
# Note each new test must append exactly 1 item to each of the CMAKE variables

function(setup_test dir arg)
  separate_arguments(arg) 
  add_test( NAME regression_test:${dir} COMMAND python "${CMAKE_CURRENT_SOURCE_DIR}/run_test.py" 
    ${arg} --test_dir "${CMAKE_CURRENT_SOURCE_DIR}/test_suites/${dir}"
    --output_dir "${CMAKE_BINARY_DIR}/tst/regression/outputs/${dir}")
  set_tests_properties(regression_test:${dir} PROPERTIES LABELS "regression;mpi-no" )
endfunction()

function(setup_test_mpi nproc dir arg)
  if( MPI_FOUND )
    separate_arguments(arg) 
    add_test( NAME regression_mpi_test:${dir} COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/run_test.py
      --mpirun ${MPIEXEC_EXECUTABLE} 
      --mpirun_opts=${MPIEXEC_NUMPROC_FLAG} --mpirun_opts=${nproc}
      --mpirun_opts=${MPIEXEC_PREFLAGS} ${arg}
      --test_dir ${CMAKE_CURRENT_SOURCE_DIR}/test_suites/${dir}
      --output_dir "${CMAKE_BINARY_DIR}/tst/regression/outputs/${dir}")
    set_tests_properties(regression_mpi_test:${dir} PROPERTIES LABELS "regression;mpi-yes" RUN_SERIAL ON )
  else()
    message(STATUS "MPI not found, not building regression tests with mpi")
  endif()
endfunction()

function(setup_test_mpi_coverage nproc dir arg)
  if( MPI_FOUND )
    separate_arguments(arg) 
    add_test( NAME regression_mpi_coverage_test:${dir} COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/run_test.py
      --coverage
      --mpirun ${MPIEXEC_EXECUTABLE} 
      --mpirun_opts=${MPIEXEC_NUMPROC_FLAG} --mpirun_opts=${nproc}
      --mpirun_opts=${MPIEXEC_PREFLAGS} ${arg}
      --test_dir ${CMAKE_CURRENT_SOURCE_DIR}/test_suites/${dir}
      --output_dir "${CMAKE_BINARY_DIR}/tst/regression/outputs/${dir}"
      )
    set_tests_properties(regression_mpi_coverage_test:${dir} PROPERTIES LABELS "regression;coverage;mpi-yes" RUN_SERIAL ON )
  else()
    message(STATUS "MPI not found, not building coverage regression tests with mpi")
  endif()
endfunction()


list(APPEND TEST_DIRS calculate_pi)
list(APPEND TEST_ARGS "--driver ${CMAKE_BINARY_DIR}/example/calculate_pi/pi-example \
--driver_input ${CMAKE_CURRENT_SOURCE_DIR}/test_suites/calculate_pi/parthinput.regression")

# Advection test (here added manually to override default nproc for MPI runs)
setup_test( advection_convergence "--driver ${CMAKE_BINARY_DIR}/example/advection/advection-example \
--driver_input ${CMAKE_CURRENT_SOURCE_DIR}/test_suites/advection_convergence/parthinput.advection \
--num_steps 25")
setup_test_mpi( 1 advection_convergence "--driver ${CMAKE_BINARY_DIR}/example/advection/advection-example \
--driver_input ${CMAKE_CURRENT_SOURCE_DIR}/test_suites/advection_convergence/parthinput.advection \
--num_steps 25")
setup_test_mpi_coverage( 1 advection_convergence "--driver ${CMAKE_BINARY_DIR}/example/advection/advection-example \
--driver_input ${CMAKE_CURRENT_SOURCE_DIR}/test_suites/advection_convergence/parthinput.advection \
--num_steps 25")

if (ENABLE_HDF5)
setup_test( output_hdf5 "--driver ${CMAKE_BINARY_DIR}/example/advection/advection-example \
--driver_input ${CMAKE_CURRENT_SOURCE_DIR}/test_suites/output_hdf5/parthinput.advection \
--num_steps 2")
setup_test_mpi( ${NUM_MPI_PROC_TESTING} output_hdf5
"--driver ${CMAKE_BINARY_DIR}/example/advection/advection-example \
--driver_input ${CMAKE_CURRENT_SOURCE_DIR}/test_suites/output_hdf5/parthinput.advection \
--num_steps 2")
endif()

# Add additional regression tests to ctest below this line by calling
#
# list(APPEND TEST_DIRS name_of_folder )   
# list(APPEND TEST_ARGS args_to_pass_to_run_test.py )
#
# Here name_of_folder should be the test folder located in parthenon/tst/regression/test_suites
# it is the, it is the parameter that is passed to run_test.py with the --test_dir argument

list(LENGTH TEST_DIRS len)
math(EXPR end_index "${len} - 1")

foreach( index RANGE ${end_index})
  list(GET TEST_DIRS ${index} TEST_DIR)
  list(GET TEST_ARGS ${index} TEST_ARG)
  setup_test( ${TEST_DIR} ${TEST_ARG})
  setup_test_mpi( ${NUM_MPI_PROC_TESTING} ${TEST_DIR} ${TEST_ARG})
  setup_test_mpi_coverage( ${NUM_MPI_PROC_TESTING} ${TEST_DIR} ${TEST_ARG})
endforeach()
