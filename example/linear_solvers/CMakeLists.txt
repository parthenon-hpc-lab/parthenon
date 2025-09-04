#=========================================================================================
# (C) (or copyright) 2023. Triad National Security, LLC. All rights reserved.
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
if( "linear-solvers-example" IN_LIST DRIVER_LIST OR NOT PARTHENON_DISABLE_EXAMPLES)
  add_executable(
      linear-solvers-example
          linear_solver_driver.cpp
          linear_solver_driver.hpp
          poisson_cell_equation.hpp
          poisson_cell_package.cpp
          poisson_cell_package.hpp
          poisson_nodal_equation.hpp
          poisson_nodal_package.cpp
          poisson_nodal_package.hpp
          helmholtz_equation.cpp
          helmholtz_equation.hpp
          helmholtz_package.cpp
          helmholtz_package.hpp
          main.cpp
          parthenon_app_inputs.cpp
          variable_type.hpp
  )
  target_link_libraries(linear-solvers-example PRIVATE Parthenon::parthenon)
  lint_target(linear-solvers-example)
endif()
