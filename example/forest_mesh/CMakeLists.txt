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
if( "forest-mesh-example" IN_LIST DRIVER_LIST OR NOT PARTHENON_DISABLE_EXAMPLES)
  add_executable(
      forest-mesh-example
          main.cpp
  )
  target_link_libraries(forest-mesh-example PRIVATE Parthenon::parthenon)
  lint_target(forest-mesh-example)
endif()
