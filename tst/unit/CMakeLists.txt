add_executable(parthenon-unit basic.cpp)
target_link_libraries(parthenon-unit PRIVATE catch2_define)

ParseAndAddCatchTests(parthenon-unit)