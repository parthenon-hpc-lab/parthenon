#=========================================================================================
# Parthenon performance portable AMR framework
# Copyright(C) 2025 The Parthenon collaboration
# Licensed under the 3-clause BSD License, see LICENSE file for details
#========================================================================================

find_package(Heffte PATHS /home/pgrete/src/heffte/install-host)

#get_property(DRIVER_LIST GLOBAL PROPERTY DRIVERS_USED_IN_TESTS)
#if( "energy_spectra" IN_LIST DRIVER_LIST OR NOT PARTHENON_DISABLE_EXAMPLES)
  add_executable(
      energy_spectra
          calc_spec.cpp
          calc_spec.hpp
          spec_driver.cpp
          spec_driver.hpp
  )
  target_link_libraries(energy_spectra PRIVATE Parthenon::parthenon)
  target_link_libraries(energy_spectra PRIVATE Heffte::Heffte)
# This will automatically link any extra libraries (or no extra libraries)
# depending on the compiler, so it doesn't require an if() statement
target_link_libraries(energy_spectra PUBLIC std::filesystem)
target_compile_definitions(energy_spectra PRIVATE
    FS_HEADER=<${CXX_FILESYSTEM_HEADER}>
    FS_NAMESPACE=${CXX_FILESYSTEM_NAMESPACE}
    )
  lint_target(energy_spectra)
#endif()
