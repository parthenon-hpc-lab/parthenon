# examples/CMakeLists.txt

set(EXAMPLES_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(par_for_example
    "${EXAMPLES_DIR}/par_for_example.cpp"
)

target_link_libraries(par_for_example
    PRIVATE MALG
)

add_executable(par_for_outer_example
    "${EXAMPLES_DIR}/par_for_outer_example.cpp"
)

target_link_libraries(par_for_outer_example
    PRIVATE MALG
)

add_executable(golub_svd_example
    "${EXAMPLES_DIR}/golub_svd_example.cpp"
)

target_link_libraries(golub_svd_example
    PRIVATE MALG
)