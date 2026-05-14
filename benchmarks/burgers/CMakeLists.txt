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

get_property(DRIVER_LIST GLOBAL PROPERTY DRIVERS_USED_IN_TESTS)
if( "burgers-benchmark" IN_LIST DRIVER_LIST OR NOT PARTHENON_DISABLE_EXAMPLES)
  add_executable(
      burgers-benchmark
          burgers_driver.cpp
          burgers_driver.hpp
          burgers_package.cpp
          burgers_package.hpp
          main.cpp
          parthenon_app_inputs.cpp
          recon.hpp
  )
  target_link_libraries(burgers-benchmark PRIVATE Parthenon::parthenon)
  lint_target(burgers-benchmark)

  # JMM: Only run burgers in 2D so startup time is reasonable because
  # -p flag goes through pgen
  add_custom_command(
    OUTPUT ${DOC_GEN_PATH}/burgers-table.csv
    COMMAND ${CMAKE_COMMAND} -E env
    bash -c
    "${CMAKE_CURRENT_BINARY_DIR}/burgers-benchmark -i ${CMAKE_CURRENT_LIST_DIR}/burgers.pin parthenon/mesh/nx3=1 parthenon/meshblock/nx3=1 -p '^(?!.*parthenon/).*' > ${DOC_GEN_PATH}/burgers-table.csv"
    DEPENDS burgers-benchmark burgers.pin
    VERBATIM
    )

  add_custom_target(burgers-csv
    DEPENDS
    ${DOC_GEN_PATH}/burgers-table.csv
    )

endif()
