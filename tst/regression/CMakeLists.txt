
function(setup_test dir arg)
  separate_arguments(arg) 
  add_test( NAME regression_test:${dir} COMMAND python3 "${CMAKE_CURRENT_SOURCE_DIR}/run_test.py" ${arg} --test "${CMAKE_CURRENT_SOURCE_DIR}/test_suites/${dir}")
  set_tests_properties(regression_test:${dir} PROPERTIES LABELS "regression" )        
endfunction()

list(APPEND TEST_DIRS calculate_pi)
list(APPEND TEST_ARGS "--driver ${CMAKE_BINARY_DIR}/example/calculate_pi/parthenon-example --driver_input ${CMAKE_CURRENT_SOURCE_DIR}/test_suites/calculate_pi/parthinput.regression")

list(LENGTH TEST_DIRS len)
math(EXPR end_index "${len} - 1")

foreach( index RANGE ${end_index})
  list(GET TEST_DIRS ${index} TEST_DIR)
  list(GET TEST_ARGS ${index} TEST_ARG)
  setup_test( ${TEST_DIR} ${TEST_ARG})
endforeach()
