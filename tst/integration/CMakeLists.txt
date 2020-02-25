foreach( TEST
      test_integration_mesh
    )

  add_executable(${TEST} ${TEST}.cpp)
  target_link_libraries(${TEST} PUBLIC parthenon Catch2::Catch2) 
  add_test(NAME ${TEST}_MPI_serial COMMAND ${TEST})
  add_test(NAME ${TEST}_MPI_parallel COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS} ${TEST} ${MPIEXEC_POSTFLAGS})

  set_tests_properties(${TEST}_MPI_serial PROPERTIES LABELS "integration;serial")
  set_tests_properties(${TEST}_MPI_parallel PROPERTIES LABELS "integration;parallel")
endforeach(TEST)
