
add_library(catch2_define catch2_define.cpp)
target_link_libraries(catch2_define PUBLIC Catch2::Catch2)

if(${ENABLE_UNIT_TESTS})
  message("Building unit tests.")
  add_subdirectory(unit)
endif()
if(${ENABLE_INTEGRATION_TESTS})
  message("Building integration tests.")
  add_subdirectory(integration)
endif()
if(${ENABLE_REGRESSION_TESTS})
  message("Building regression tests.")
  add_subdirectory(regression)
endif()


