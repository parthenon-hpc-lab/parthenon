#=========================================================================================
# Parthenon performance portable AMR framework
# Copyright(C) 2023-2025 The Parthenon collaboration
# Licensed under the 3-clause BSD License, see LICENSE file for details
#=========================================================================================

get_property(DRIVER_LIST GLOBAL PROPERTY DRIVERS_USED_IN_TESTS)
if(NOT PARTHENON_DISABLE_EXAMPLES)
  add_executable(
      energy-transfer
          energy_transfer_driver.cpp
          energy_transfer_driver.hpp
  )
  find_package(ADIOS2 REQUIRED COMPONENTS CXX MPI)
  target_link_libraries(energy-transfer PRIVATE Parthenon::parthenon adios2::cxx_mpi MPI::MPI_CXX)

  if(PARTHENON_ENABLE_TESTING AND PARTHENON_ENABLE_REGRESSION_TESTS)
    execute_process(COMMAND ${Python3_EXECUTABLE} -c "import numpy, adios2, openpmd_api"
                    RESULT_VARIABLE flux_python_status OUTPUT_QUIET ERROR_QUIET)
    if(flux_python_status EQUAL 0)
      set(flux_test_args ${CMAKE_CURRENT_SOURCE_DIR}/test_flux.py $<TARGET_FILE:energy-transfer>)
      if(PARTHENON_SINGLE_PRECISION)
        list(APPEND flux_test_args --single)
      endif()
      add_test(NAME energy_transfer_flux COMMAND ${Python3_EXECUTABLE} ${flux_test_args})
      set_tests_properties(energy_transfer_flux PROPERTIES
        LABELS "regression;energy_transfer"
        ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1")
    else()
      message(STATUS "Skipping energy transfer flux test: requires numpy, adios2, openpmd_api")
    endif()
  endif()
endif()
