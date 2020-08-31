
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

# The following lines determine the default loop pattern by setting the
# default defines to the appropriate "tags" in the config.hpp file.
# See `kokkos_abstraction.hpp` for available tags and what they translate to.
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

if (${PAR_LOOP_LAYOUT} STREQUAL "MANUAL1D_LOOP")
  set(PAR_LOOP_LAYOUT_TAG loop_pattern_flatrange_tag)
elseif (${PAR_LOOP_LAYOUT} STREQUAL "SIMDFOR_LOOP")
  set(PAR_LOOP_LAYOUT_TAG loop_pattern_simdfor_tag)
elseif (${PAR_LOOP_LAYOUT} STREQUAL "MDRANGE_LOOP")
  set(PAR_LOOP_LAYOUT_TAG loop_pattern_mdrange_tag)
elseif (${PAR_LOOP_LAYOUT} STREQUAL "TP_TTR_LOOP")
  set(PAR_LOOP_LAYOUT_TAG loop_pattern_tpttr_tag)
elseif (${PAR_LOOP_LAYOUT} STREQUAL "TP_TVR_LOOP")
  set(PAR_LOOP_LAYOUT_TAG loop_pattern_tptvr_tag)
elseif (${PAR_LOOP_LAYOUT} STREQUAL "TPTTRTVR_LOOP")
  set(PAR_LOOP_LAYOUT_TAG loop_pattern_tpttrtvr_tag)
else()
  set(PAR_LOOP_LAYOUT_TAG loop_pattern_undefined_tag)
endif()

if (${PAR_LOOP_INNER_LAYOUT} STREQUAL "TVR_INNER_LOOP")
  set(PAR_LOOP_INNER_LAYOUT_TAG inner_loop_pattern_tvr_tag)
elseif (${PAR_LOOP_INNER_LAYOUT} STREQUAL "SIMDFOR_INNER_LOOP")
  set(PAR_LOOP_INNER_LAYOUT_TAG inner_loop_pattern_simdfor_tag)
else()
  set(PAR_LOOP_INNER_LAYOUT_TAG loop_pattern_undefined_tag)
endif()

set(EXCEPTION_HANDLING_OPTION ENABLE_EXCEPTIONS) # TODO: Add option to disable exceptions
set(COMPILED_WITH ${CMAKE_CXX_COMPILER})
set(COMPILER_COMMAND "<not-implemented>") # TODO: Put something more descriptive here
set(COMPILER_FLAGS "<not-implemented>") # TODO: Put something more descriptive here

set(NFIELD_VARIABLES 0) # TODO: Remove
set(NWAVE_VALUE 5) # TODO: Remove
set(COORDINATE_TYPE UniformCartesian) # TODO: Make this an option when more are available

configure_file(${PROJECT_SOURCE_DIR}/src/config.hpp.in ${PROJECT_SOURCE_DIR}/include/parthenon/config.hpp @ONLY)

list(APPEND PARTHENON_SOURCES
  bvals/cc/bvals_cc.cpp
  bvals/cc/flux_correction_cc.cpp

  bvals/fc/bvals_fc.cpp
  bvals/fc/flux_correction_fc.cpp

  bvals/boundary_conditions.cpp

  bvals/bvals.cpp
  bvals/bvals_base.cpp
  bvals/boundary_flag.cpp
  bvals/bvals_refine.cpp
  bvals/bvals_var.cpp

  driver/driver.cpp
  driver/multistage.cpp

  interface/container_collection.cpp
  interface/container.cpp
  interface/metadata.cpp
  interface/properties_interface.cpp
  interface/sparse_variable.cpp
  interface/update.cpp
  interface/variable.cpp

  mesh/amr_loadbalance.cpp
  mesh/mesh_refinement.cpp
  mesh/mesh.cpp
  mesh/meshblock_tree.cpp
  mesh/meshblock.cpp
  mesh/weighted_ave.cpp

  outputs/formatted_table.cpp
  outputs/history.cpp
  outputs/io_wrapper.cpp
  outputs/outputs.cpp
  outputs/parthenon_hdf5.cpp
  outputs/restart.cpp
  outputs/vtk.cpp

  pgen/default_pgen.cpp

  reconstruct/characteristic.cpp
  reconstruct/dc.cpp
  reconstruct/plm.cpp
  reconstruct/ppm.cpp
  reconstruct/reconstruction.cpp

  refinement/amr_criteria.cpp
  refinement/refinement.cpp

  tasks/task_id.cpp

  utils/buffer_utils.cpp
  utils/change_rundir.cpp
  utils/show_config.cpp
  utils/signal_handler.cpp
  utils/trim_string.cpp

  globals.cpp
  parameter_input.cpp
  parthenon_manager.cpp
)

add_library(parthenon ${PARTHENON_SOURCES} ${PARTHENON_PUBLIC_HEADERS})
add_library(Parthenon::parthenon ALIAS parthenon)

set_target_properties(parthenon PROPERTIES SOVERSION ${parthenon_VERSION})

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
  $<BUILD_INTERFACE:${parthenon_SOURCE_DIR}/include/parthenon>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include/parthenon/generated>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/parthenon>
  )

lint_target(parthenon)

include(${parthenon_SOURCE_DIR}/cmake/ParthenonInstall.cmake)
