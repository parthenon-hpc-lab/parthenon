
add_library(catch2_define catch2_define.cpp)
target_link_libraries(catch2_define PUBLIC Catch2::Catch2 Kokkos::kokkos)

if(${ENABLE_UNIT_TESTS})
  message(STATUS "Building unit tests.")
  add_subdirectory(unit)
endif()
if(${ENABLE_INTEGRATION_TESTS})
  message(STATUS "Building integration tests.")
  add_subdirectory(integration)
endif()
if(${ENABLE_PERFORMANCE_TESTS})
  message(STATUS "Building performance tests.")
  target_compile_definitions(catch2_define PUBLIC CATCH_CONFIG_ENABLE_BENCHMARKING)
  add_subdirectory(performance)
endif()
if(${ENABLE_REGRESSION_TESTS})
  message(STATUS "Building regression tests.")
  add_subdirectory(regression)
endif()


