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

get_property(LIST_OF_TESTS GLOBAL PROPERTY DRIVER_LIST)
if( "loop-advection" IN_LIST DRIVER_LIST OR NOT PARTHENON_DISABLE_EXAMPLES)
  add_executable(
      loop-advection
          loop_advection.hpp
          loop_advection_driver.cpp
          loop_advection_main.cpp
          loop_advection_package.cpp
  )
  target_link_libraries(loop-advection PRIVATE Parthenon::parthenon)
  lint_target(loop-advection)
endif()
