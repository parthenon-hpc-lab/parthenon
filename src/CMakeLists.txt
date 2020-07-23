
#=========================================================================================
# (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001 for Los
# Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
# for the U.S. Department of Energy/National Nuclear Security Administration. All rights
# in the program are reserved by Triad National Security, LLC, and the U.S. Department
# of Energy/National Nuclear Security Administration. The Government is granted for
# itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
# license in this material to reproduce, prepare derivative works, distribute copies to
# the public, perform publicly and display publicly, and to permit others to do so.
#=========================================================================================

# Configure config.hpp
set(PROBLEM_GENERATOR "<not-implemented>") # TODO: Figure out what to put here

if (ENABLE_MPI)
  set(MPI_OPTION MPI_PARALLEL)
else ()
  set(MPI_OPTION NOT_MPI_PARALLEL)
endif()

if (ENABLE_OPENMP)
  set(OPENMP_OPTION OPENMP_PARALLEL)
else()
  set(OPENMP_OPTION NOT_OPENMP_PARALLEL)
endif()

if (ENABLE_HDF5)
  set(HDF5_OPTION HDF5OUTPUT)
else()
  set(HDF5_OPTION NO_HDF5OUTPUT)
endif()

if (${Kokkos_ENABLE_CUDA})
  set(PAR_LOOP_LAYOUT "MANUAL1D_LOOP" CACHE STRING
    "Default loop layout for parallel_for wrapper")

  set(PAR_LOOP_LAYOUT_VALUES "MANUAL1D_LOOP;MDRANGE_LOOP;TPTTR_LOOP;TPTTRTVR_LOOP"
    CACHE STRING "Possible loop layout options.")

  set(PAR_LOOP_INNER_LAYOUT "TVR_INNER_LOOP" CACHE STRING
    "Default loop layout for par_for_inner wrapper")

  set(PAR_LOOP_INNER_LAYOUT_VALUES "TVR_INNER_LOOP"
    CACHE STRING "Possible inner loop layout options.")

elseif(${Kokkos_ENABLE_HPX})
  message( FATAL_ERROR "Need to add/fix/test default loop layouts for HPX backend.")

else()
  # use simd for loop when not using Nvidia GPUs
  set(PAR_LOOP_LAYOUT "SIMDFOR_LOOP" CACHE STRING
    "Default loop layout for parallel_for wrapper")
  set(PAR_LOOP_LAYOUT_VALUES "SIMDFOR_LOOP;MANUAL1D_LOOP;MDRANGE_LOOP;TPTTR_LOOP;TPTVR_LOOP;TPTTRTVR_LOOP"
    CACHE STRING "Possible loop layout options.")

  set(PAR_LOOP_INNER_LAYOUT "SIMDFOR_INNER_LOOP" CACHE STRING
    "Default loop layout for par_for_inner wrapper")

  set(PAR_LOOP_INNER_LAYOUT_VALUES "SIMDFOR_INNER_LOOP;TVR_INNER_LOOP"
    CACHE STRING "Possible inner loop layout options.")

endif()

set_property(CACHE PAR_LOOP_LAYOUT PROPERTY STRINGS ${PAR_LOOP_LAYOUT_VALUES})

set_property(CACHE PAR_LOOP_INNER_LAYOUT PROPERTY STRINGS ${PAR_LOOP_INNER_LAYOUT_VALUES})

message(STATUS "PAR_LOOP_LAYOUT='${PAR_LOOP_LAYOUT}' (default par_for wrapper layout)")

message(STATUS "PAR_LOOP_INNER_LAYOUT='${PAR_LOOP_INNER_LAYOUT}' (default par_for_inner wrapper layout)")

set(EXCEPTION_HANDLING_OPTION ENABLE_EXCEPTIONS) # TODO: Add option to disable exceptions
set(COMPILED_WITH ${CMAKE_CXX_COMPILER})
set(COMPILER_COMMAND "<not-implemented>") # TODO: Put something more descriptive here
set(COMPILER_FLAGS "<not-implemented>") # TODO: Put something more descriptive here

set(NFIELD_VARIABLES 0) # TODO: Remove
set(NWAVE_VALUE 5) # TODO: Remove
set(COORDINATE_TYPE UniformCartesian) # TODO: Make this an option when more are available

configure_file(config.hpp.in generated/config.hpp @ONLY)

file( GLOB base_src *.cpp )
file( GLOB base_hdr *.hpp )
file( GLOB bvals_src bvals/*.cpp bvals/fc/*.cpp bvals/cc/*.cpp)
file( GLOB bvals_hdr bvals/*.hpp bvals/fc/*.hpp bvals/cc/*.hpp)
file( GLOB coordinates_src coordinates/*.cpp )
file( GLOB coordinates_hdr coordinates/*.hpp )
file( GLOB driver_src driver/*.cpp )
file( GLOB driver_hdr driver/*.hpp )
file( GLOB interface_src interface/*.cpp )
file( GLOB interface_hdr interface/*.hpp )
file( GLOB mesh_src mesh/*.cpp )
file( GLOB mesh_hdr mesh/*.hpp )
file( GLOB outputs_src outputs/*.cpp )
file( GLOB outputs_hdr outputs/*.hpp )
file( GLOB parthenon_src parthenon/*.cpp )
file( GLOB parthenon_hdr parthenon/*.hpp )
file( GLOB pgen_src pgen/*.cpp )
file( GLOB pgen_hdr pgen/*.hpp )
file( GLOB reconstruct_src reconstruct/*.cpp )
file( GLOB reconstruct_hdr reconstruct/*.hpp )
file( GLOB refinement_src refinement/*.cpp )
file( GLOB refinement_hdr refinement/*.hpp )
file( GLOB tasks_src tasks/*.cpp )
file( GLOB tasks_hdr tasks/*.hpp )
file( GLOB utils_src utils/*.cpp )
file( GLOB utils_hdr utils/*.hpp )

list(APPEND parthenon_subdirs
  "bvals"
  "coordinates"
  "driver"
  "interface"
  "mesh"
  "outputs"
  "parthenon"
  "pgen"
  "reconstruct"
  "refinement"
  "tasks"
  "utils")

list(APPEND parthenon_sources
     "${bvals_src}"
     "${coordinates_src}"
     "${driver_src}"
     "${interface_src}"
     "${mesh_src}"
     "${outputs_src}"
     "${parthenon_src}"
     "${pgen_src}"
     "${reconstruct_src}"
     "${refinement_src}"
     "${tasks_src}"
     "${utils_src}"
     "${base_src}"
     )

add_library(parthenon SHARED "${parthenon_sources}")

# Other headers will be installed manually later to maintain directory structure
list(APPEND public_headers "${base_hdr}")
set_target_properties(parthenon PROPERTIES PUBLIC_HEADER "${public_headers}")


target_compile_features(parthenon PUBLIC cxx_std_14)
if (CMAKE_CXX_COMPILER_ID STREQUAL "XL")
  target_compile_options(parthenon PUBLIC -std=c++1y -qxflag=disable__cplusplusOverride)
endif()


if (ENABLE_MPI)
  target_link_libraries(parthenon PUBLIC MPI::MPI_CXX)
endif()

if (ENABLE_OPENMP)
  target_link_libraries(parthenon PUBLIC OpenMP::OpenMP_CXX)
endif()

if (ENABLE_HDF5)
  target_link_libraries(parthenon PUBLIC HDF5_C)
endif()

if (Kokkos_ENABLE_CUDA)
   target_compile_options(parthenon PUBLIC --expt-relaxed-constexpr)
endif()

target_link_libraries(parthenon PUBLIC Kokkos::kokkos)

target_include_directories(parthenon PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}/generated
)

lint_target(parthenon)

install(TARGETS parthenon LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/parthenon")

# Maintain directory structure in installed include files
install(FILES ${base_hdr} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
foreach( subdir ${parthenon_subdirs} )
  file(TO_CMAKE_PATH "${subdir}" hdr_subdir)
  install(DIRECTORY "${hdr_subdir}" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN "*.hpp")
  unset(hdr_subdir)
endforeach()

# Install generated config header file
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/generated/config.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
