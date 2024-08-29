#=========================================================================================
# (C) (or copyright) 2024. Triad National Security, LLC. All rights reserved.
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

# There is no test but we build it if we build examples
if (NOT PARTHENON_DISABLE_EXAMPLES)
  set (SRC_LIST
  main.cpp

  driver.hpp

  ellipse/ellipse.cpp
  ellipse/ellipse.hpp

  indicator/indicator.cpp
  indicator/indicator.hpp

  pgen.cpp
  pgen.hpp
  )

  add_executable(ellipse ${SRC_LIST})
  target_include_directories(ellipse PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
    )
  target_compile_features(ellipse PRIVATE cxx_std_17)
  
  if (Kokkos_ENABLE_CUDA)
    target_compile_options(ellipse --expt-relaxed-constexpr)
  endif()
  
  target_link_libraries(ellipse PRIVATE parthenon)
endif()
