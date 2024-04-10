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

# Adding regression tests
#
# To add additional regression tests to ctest, four things must be added to
# this file:
#
# 1. there must exist within the regression/test_suites directory a folder associated with the test
#   * For the calculate_pi example the folder is called 'calculate_pi', This
#     folder must be added to the cmake variable TEST_DIRS as shown below.
#
# 2. the correct arguments to pass to the 'run_test.py' script must be specified
#   * This arguments must be added to the CMAKE variable TEST_ARGS as shown below
#
# 3. the number of mpi processors that should be used when testing the mpi implementation
#   * This argument needs to be added to the TEST_PROCS variable
#
# 4. any extra labels that need to be appended to a test
#
# NOTE: each new test must append exactly 1 item to each of the CMAKE variables
#
# Add additional regression tests to ctest below this line by calling
#
# list(APPEND TEST_DIRS name_of_folder )
# list(APPEND TEST_PROCS ${NUM_MPI_PROC_TESTING})
# list(APPEND TEST_ARGS args_to_pass_to_run_test.py )
# list(APPEND EXTRA_TEST_LABELS "my-awesome-test")
#
# Here name_of_folder should be the test folder located in parthenon/tst/regression/test_suites
# it is the parameter that is passed to run_test.py with the --test_dir argument

# Performance regression test
list(APPEND TEST_DIRS advection_performance)
list(APPEND TEST_PROCS ${NUM_MPI_PROC_TESTING})
list(APPEND TEST_ARGS "--driver ${PROJECT_BINARY_DIR}/example/advection/advection-example \
--driver_input ${CMAKE_CURRENT_SOURCE_DIR}/test_suites/advection_performance/parthinput.advection_performance \
--num_steps 4")
list(APPEND EXTRA_TEST_LABELS "perf-reg")

list(APPEND TEST_DIRS particle_leapfrog)
list(APPEND TEST_PROCS ${NUM_MPI_PROC_TESTING})
list(APPEND TEST_ARGS "--driver ${PROJECT_BINARY_DIR}/example/particle_leapfrog/particle-leapfrog \
--driver_input ${CMAKE_CURRENT_SOURCE_DIR}/test_suites/particle_leapfrog/parthinput.particle_leapfrog \
--num_steps 2")
list(APPEND EXTRA_TEST_LABELS "")

list(APPEND TEST_DIRS particle_leapfrog_outflow)
list(APPEND TEST_PROCS ${NUM_MPI_PROC_TESTING})
list(APPEND TEST_ARGS "--driver ${PROJECT_BINARY_DIR}/example/particle_leapfrog/particle-leapfrog \
--driver_input ${CMAKE_CURRENT_SOURCE_DIR}/test_suites/particle_leapfrog_outflow/parthinput.particle_leapfrog_outflow \
--num_steps 1")
list(APPEND EXTRA_TEST_LABELS "")

if (ENABLE_HDF5)

  # h5py is needed for restart and hdf5 test
  list(APPEND REQUIRED_PYTHON_MODULES h5py)

  # Restart
  list(APPEND TEST_DIRS restart)
  list(APPEND TEST_PROCS ${NUM_MPI_PROC_TESTING})
  list(APPEND TEST_ARGS "--driver ${PROJECT_BINARY_DIR}/example/sparse_advection/sparse_advection-example \
    --driver_input ${CMAKE_CURRENT_SOURCE_DIR}/test_suites/restart/parthinput.restart \
    --num_steps 3")
  list(APPEND EXTRA_TEST_LABELS "")

  # Calculate pi example
  list(APPEND TEST_DIRS calculate_pi)
  list(APPEND TEST_PROCS ${NUM_MPI_PROC_TESTING})
  list(APPEND TEST_ARGS "--driver ${PROJECT_BINARY_DIR}/example/calculate_pi/pi-example \
  --driver_input ${CMAKE_CURRENT_SOURCE_DIR}/test_suites/calculate_pi/parthinput.regression")
  list(APPEND EXTRA_TEST_LABELS "")

  # Advection test
  list(APPEND TEST_DIRS advection_convergence)
  list(APPEND TEST_PROCS ${NUM_MPI_PROC_TESTING})
  list(APPEND TEST_ARGS "--driver ${PROJECT_BINARY_DIR}/example/advection/advection-example \
  --driver_input ${CMAKE_CURRENT_SOURCE_DIR}/test_suites/advection_convergence/parthinput.advection \
  --num_steps 25")
  list(APPEND EXTRA_TEST_LABELS "")

  list(APPEND TEST_DIRS output_hdf5)
  list(APPEND TEST_PROCS ${NUM_MPI_PROC_TESTING})
  list(APPEND TEST_ARGS "--driver ${PROJECT_BINARY_DIR}/example/advection/advection-example \
    --driver_input ${CMAKE_CURRENT_SOURCE_DIR}/test_suites/output_hdf5/parthinput.advection \
    --num_steps 4")
  list(APPEND EXTRA_TEST_LABELS "")

  list(APPEND TEST_DIRS advection_outflow)
  list(APPEND TEST_PROCS ${NUM_MPI_PROC_TESTING})
  list(APPEND TEST_ARGS "--driver ${PROJECT_BINARY_DIR}/example/advection/advection-example \
    --driver_input ${CMAKE_CURRENT_SOURCE_DIR}/test_suites/advection_outflow/parthinput.advection_outflow")
  list(APPEND EXTRA_TEST_LABELS "")

  list(APPEND TEST_DIRS bvals)
  list(APPEND TEST_PROCS ${NUM_MPI_PROC_TESTING})
  list(APPEND TEST_ARGS "--driver ${PROJECT_BINARY_DIR}/example/advection/advection-example \
    --driver_input ${CMAKE_CURRENT_SOURCE_DIR}/test_suites/bvals/parthinput.advection_bvals \
    --num_steps 3")
  list(APPEND EXTRA_TEST_LABELS "")

  list(APPEND TEST_DIRS poisson)
  list(APPEND TEST_PROCS ${NUM_MPI_PROC_TESTING})
  list(APPEND TEST_ARGS "--driver ${PROJECT_BINARY_DIR}/example/poisson/poisson-example \
    --driver_input ${CMAKE_CURRENT_SOURCE_DIR}/test_suites/poisson/parthinput.poisson")
  list(APPEND EXTRA_TEST_LABELS "poisson")

  list(APPEND TEST_DIRS poisson_gmg)
  list(APPEND TEST_PROCS ${NUM_MPI_PROC_TESTING})
  list(APPEND TEST_ARGS "--driver ${PROJECT_BINARY_DIR}/example/poisson_gmg/poisson-gmg-example \
    --driver_input ${CMAKE_CURRENT_SOURCE_DIR}/test_suites/poisson_gmg/parthinput.poisson")
  list(APPEND EXTRA_TEST_LABELS "poisson_gmg")

  list(APPEND TEST_DIRS sparse_advection)
  list(APPEND TEST_PROCS ${NUM_MPI_PROC_TESTING})
  list(APPEND TEST_ARGS "--driver ${PROJECT_BINARY_DIR}/example/sparse_advection/sparse_advection-example \
    --driver_input ${CMAKE_CURRENT_SOURCE_DIR}/test_suites/sparse_advection/parthinput.sparse_advection")
  list(APPEND EXTRA_TEST_LABELS "")

endif()

# Any external modules that are required by python can be added to REQUIRED_PYTHON_MODULES
# list variable, before including TestSetup.cmake.
list(APPEND REQUIRED_PYTHON_MODULES numpy)
list(APPEND DESIRED_PYTHON_MODULES matplotlib)

# Include test setup functions, and check for python interpreter and modules
#  setup_test_serial
#  setup_test_coverage
#  setup_test_parallel
#  setup_test_mpi_coverage
include(${PROJECT_SOURCE_DIR}/cmake/TestSetup.cmake)

list(LENGTH TEST_DIRS len)
math(EXPR end_index "${len} - 1")

foreach( index RANGE ${end_index})
  list(GET TEST_DIRS ${index} TEST_DIR)
  list(GET TEST_PROCS ${index} TEST_PROC)
  list(GET TEST_ARGS ${index} TEST_ARG)
  list(GET EXTRA_TEST_LABELS ${index} TEST_LABELS )

  if (PARTHENON_DISABLE_SPARSE)
    string(APPEND TEST_ARG " --sparse_disabled")
  endif()

  setup_test_serial(${TEST_DIR} ${TEST_ARG} "${TEST_LABELS}")
  setup_test_coverage(${TEST_DIR} ${TEST_ARG} "${TEST_LABELS}")
  setup_test_parallel(${TEST_PROC} ${TEST_DIR} ${TEST_ARG} "${TEST_LABELS}")
  setup_test_mpi_coverage(${TEST_PROC} ${TEST_DIR} ${TEST_ARG} "${TEST_LABELS}")
endforeach()

