
set(INTEGRATION_TEST_NAME integration_tests)

list(APPEND integration_tests_SOURCES test_integration_mesh.cpp)

add_executable(${INTEGRATION_TEST_NAME}_serial ${integration_tests_SOURCES})
add_executable(${INTEGRATION_TEST_NAME}_parallel ${integration_tests_SOURCES})

target_link_libraries(${INTEGRATION_TEST_NAME}_serial PUBLIC parthenon Catch2::Catch2) 
target_link_libraries(${INTEGRATION_TEST_NAME}_parallel PUBLIC parthenon Catch2::Catch2) 

set_target_properties(${INTEGRATION_TEST_NAME}_serial PROPERTIES LABELS "integration;serial")
set_target_properties(${INTEGRATION_TEST_NAME}_parallel PROPERTIES LABELS "integration;parallel")

ParseAndAddCatchTests(${INTEGRATION_TEST_NAME}_serial)
set(OptionalCatchTestLauncher ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4)
ParseAndAddCatchTests(${INTEGRATION_TEST_NAME}_parallel)
unset(OptionalCatchTestLauncher)

