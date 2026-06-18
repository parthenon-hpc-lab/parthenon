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
endif()
