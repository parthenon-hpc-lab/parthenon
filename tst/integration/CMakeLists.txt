foreach( TEST
      test_integration_mesh
    )

  file(GLOB ${TEST}_SOURCES ${TEST}.cpp)
  add_executable(${TEST} ${${TEST}_SOURCES})
  target_link_libraries(${TEST} PUBLIC parthenon Catch2::Catch2) 
  add_test(NAME ${TEST}_MPI_serial COMMAND mpirun -np 1 ${TEST})
  add_test(NAME ${TEST}_MPI_parallel COMMAND mpirun -np 4 ${TEST})

  set_tests_properties(${TEST}_MPI_serial PROPERTIES LABELS "integration;serial")
  set_tests_properties(${TEST}_MPI_parallel PROPERTIES LABELS "integration;parallel")
  endforeach(TEST)
