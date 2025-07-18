#=========================================================================================
# (C) (or copyright) 2023-2025. Triad National Security, LLC. All rights reserved.
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
if( "diffusion-example" IN_LIST DRIVER_LIST OR NOT PARTHENON_DISABLE_EXAMPLES)
  add_executable(
      diffusion-example
          diffusion_driver.cpp
          diffusion_driver.hpp
          diffusion_equation.hpp
          diffusion_package.cpp
          diffusion_package.hpp
          main.cpp
          parthenon_app_inputs.cpp
  )
  target_link_libraries(diffusion-example PRIVATE Parthenon::parthenon)
  lint_target(diffusion-example)

  add_custom_command(
    OUTPUT ${DOC_GEN_PATH}/diffusion-parth-table.csv
    COMMAND ${CMAKE_COMMAND} -E env
    bash -c
    "${CMAKE_CURRENT_BINARY_DIR}/diffusion-example -i ${CMAKE_CURRENT_LIST_DIR}/parthinput.diffusion parthenon/output0/dt=-1 -p 'parthenon/(.*)' > ${DOC_GEN_PATH}/diffusion-parth-table.csv"
    DEPENDS diffusion-example parthinput.diffusion
    VERBATIM
    )
  add_custom_command(
    OUTPUT ${DOC_GEN_PATH}/diffusion-table.csv
    COMMAND ${CMAKE_COMMAND} -E env
    bash -c
    "${CMAKE_CURRENT_BINARY_DIR}/diffusion-example -i ${CMAKE_CURRENT_LIST_DIR}/parthinput.diffusion parthenon/output0/dt=-1 -p '^(?!.*parthenon/).*' > ${DOC_GEN_PATH}/diffusion-table.csv"
    DEPENDS diffusion-example parthinput.diffusion
    VERBATIM
    )

  add_custom_target(diffusion-csv
    DEPENDS
    ${DOC_GEN_PATH}/diffusion-parth-table.csv
    ${DOC_GEN_PATH}/diffusion-table.csv
    )

endif()
