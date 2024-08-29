if (NOT PARTHENON_DISABLE_EXAMPLES)
  set (SRC_LIST
    main.cpp
    mccirc/mccirc.hpp
    mccirc/mccirc.cpp
    pgen.cpp
    pgen.hpp
    )
  
  add_executable(mcpi ${SRC_LIST})
  target_include_directories(mcpi PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
    )
  target_compile_features(mcpi PRIVATE cxx_std_17)
  
  if (Kokkos_ENABLE_CUDA)
    target_compile_options(mcpi --expt-relaxed-constexpr)
  endif()
  
  target_link_libraries(mcpi PRIVATE parthenon)
endif()
