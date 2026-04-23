#=========================================================================================
# (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
#=========================================================================================

get_property(DRIVER_LIST GLOBAL PROPERTY DRIVERS_USED_IN_TESTS)
if("loop-benchmarks" IN_LIST DRIVER_LIST OR NOT PARTHENON_DISABLE_EXAMPLES)
  add_executable(
    loop-benchmarks
      benchmark_driver.cpp
      benchmark_driver.hpp
      config.hpp
      decomposition.hpp
      kernels.hpp
      loop_patterns.cpp
      loop_patterns.hpp
      main.cpp
      output.cpp
      output.hpp
      ragged_metadata.hpp
      raw_memory_indexer.hpp
  )

  target_include_directories(
    loop-benchmarks
    PRIVATE
      "${CMAKE_CURRENT_LIST_DIR}"
      "${CMAKE_CURRENT_LIST_DIR}/../../src"
  )

  target_link_libraries(loop-benchmarks PRIVATE Kokkos::kokkos)
  target_compile_options(
    loop-benchmarks
    PRIVATE
      $<$<CXX_COMPILER_ID:AppleClang,Clang,GNU>:-Wall -Wextra -Wpedantic>
      $<$<AND:$<NOT:$<BOOL:${Kokkos_ENABLE_HIP}>>,$<NOT:$<BOOL:${Kokkos_ENABLE_CUDA}>>,$<CXX_COMPILER_ID:GNU,Clang>>:-fopenmp-simd>
      $<$<AND:$<NOT:$<BOOL:${Kokkos_ENABLE_HIP}>>,$<NOT:$<BOOL:${Kokkos_ENABLE_CUDA}>>,$<CXX_COMPILER_ID:AppleClang>>:-Xclang -fopenmp-simd>
  )

  lint_target(loop-benchmarks)
endif()
