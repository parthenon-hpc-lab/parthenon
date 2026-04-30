#=========================================================================================
# (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
#=========================================================================================

get_property(DRIVER_LIST GLOBAL PROPERTY DRIVERS_USED_IN_TESTS)
if("loop-benchmarks-v2" IN_LIST DRIVER_LIST OR NOT PARTHENON_DISABLE_EXAMPLES)
  add_executable(
    loop-benchmarks-v2
      benchmark_driver.cpp
      benchmark_driver.hpp
      dataset.cpp
      dataset.hpp
      kernels.hpp
      multicase.cpp
      multicase.hpp
      loop_patterns.hpp
      problem_spec.hpp
      runner.cpp
      runner.hpp
      main.cpp
  )

  target_include_directories(
    loop-benchmarks-v2
    PRIVATE
      "${PROJECT_BINARY_DIR}/src/generated"
      "${CMAKE_CURRENT_LIST_DIR}"
      "${CMAKE_CURRENT_LIST_DIR}/../../src"
  )

  target_link_libraries(loop-benchmarks-v2 PRIVATE Parthenon::parthenon)
  target_compile_options(
    loop-benchmarks-v2
    PRIVATE
      $<$<CXX_COMPILER_ID:AppleClang,Clang,GNU>:-Wall -Wextra -Wpedantic>
      $<$<AND:$<NOT:$<BOOL:${Kokkos_ENABLE_HIP}>>,$<NOT:$<BOOL:${Kokkos_ENABLE_CUDA}>>,$<CXX_COMPILER_ID:GNU,Clang>>:-fopenmp-simd>
      $<$<AND:$<NOT:$<BOOL:${Kokkos_ENABLE_HIP}>>,$<NOT:$<BOOL:${Kokkos_ENABLE_CUDA}>>,$<CXX_COMPILER_ID:AppleClang>>:-Xclang -fopenmp-simd>
      $<$<CXX_COMPILER_ID:AppleClang,Clang>:-Rpass=loop-vectorize>
      $<$<CXX_COMPILER_ID:AppleClang,Clang>:-Rpass-missed=loop-vectorize>
      $<$<CXX_COMPILER_ID:AppleClang,Clang>:-Rpass-analysis=loop-vectorize>
  )

  lint_target(loop-benchmarks-v2)

  add_executable(
    loop-benchmarks-v2-sandbox
      loop_abstraction_sandbox.cpp
  )
   
  target_compile_options(loop-benchmarks-v2-sandbox PRIVATE
    -O3
    -Rpass=loop-vectorize
    -Rpass-missed=loop-vectorize
    -Rpass-analysis=loop-vectorize
    -fsave-optimization-record
    -fvectorize
    -fslp-vectorize
    -Xclang -fopenmp-simd
  )

  target_include_directories(
    loop-benchmarks-v2-sandbox
    PRIVATE
      "${PROJECT_BINARY_DIR}/src/generated"
      "${CMAKE_CURRENT_LIST_DIR}"
      "${CMAKE_CURRENT_LIST_DIR}/../../src"
  )

  target_link_libraries(loop-benchmarks-v2-sandbox PRIVATE Parthenon::parthenon)
  target_compile_options(
    loop-benchmarks-v2-sandbox
    PRIVATE
      $<$<CXX_COMPILER_ID:AppleClang,Clang,GNU>:-Wall -Wextra -Wpedantic>
      $<$<AND:$<NOT:$<BOOL:${Kokkos_ENABLE_HIP}>>,$<NOT:$<BOOL:${Kokkos_ENABLE_CUDA}>>,$<CXX_COMPILER_ID:GNU,Clang>>:-fopenmp-simd>
      $<$<AND:$<NOT:$<BOOL:${Kokkos_ENABLE_HIP}>>,$<NOT:$<BOOL:${Kokkos_ENABLE_CUDA}>>,$<CXX_COMPILER_ID:AppleClang>>:-Xclang -fopenmp-simd>
      $<$<CXX_COMPILER_ID:AppleClang,Clang>:-Rpass=loop-vectorize>
      $<$<CXX_COMPILER_ID:AppleClang,Clang>:-Rpass-missed=loop-vectorize>
      $<$<CXX_COMPILER_ID:AppleClang,Clang>:-Rpass-analysis=loop-vectorize>
  )

  lint_target(loop-benchmarks-v2-sandbox)
endif()
