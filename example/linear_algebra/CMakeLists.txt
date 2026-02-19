#=========================================================================================
# (C) (or copyright) 2021. Triad National Security, LLC. All rights reserved.
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

#get_property(DRIVER_LIST GLOBAL PROPERTY DRIVERS_USED_IN_TESTS)
#if( "linear_algebra-example" IN_LIST DRIVER_LIST OR NOT PARTHENON_DISABLE_EXAMPLES)
#add_executable(
#linear_algebra-example
#golub_svd_example.cpp
#par_for_example.cpp
#par_for_outer_example.cpp
#)
#target_link_libraries(linear_algebra-example PRIVATE Parthenon::parthenon)
#
#lint_target(linear_algebra-example)
#endif()

get_property(DRIVER_LIST GLOBAL PROPERTY DRIVERS_USED_IN_TESTS)

set(LINALG_EXAMPLES
  golub_svd_example
  par_for_example
  par_for_outer_example
)

foreach(example ${LINALG_EXAMPLES})
  if("${example}" IN_LIST DRIVER_LIST OR NOT PARTHENON_DISABLE_EXAMPLES)
    add_executable(
      ${example}
      ${example}.cpp
    )

    target_link_libraries(${example}
      PRIVATE
        Parthenon::parthenon
    )

    lint_target(${example})
  endif()
endforeach()
