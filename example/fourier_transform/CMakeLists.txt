#=========================================================================================
# Parthenon performance portable AMR framework
# Copyright(C) 2026 The Parthenon collaboration
# Licensed under the 3-clause BSD License, see LICENSE file for details
#========================================================================================

get_property(DRIVER_LIST GLOBAL PROPERTY DRIVERS_USED_IN_TESTS)
if(NOT PARTHENON_DISABLE_EXAMPLES AND PARTHENON_ENABLE_FFT)
  add_executable(
      fourier-transform
          fourier_driver.cpp
          fourier_driver.hpp
  )
  target_link_libraries(fourier-transform PRIVATE Parthenon::parthenon)
endif()
