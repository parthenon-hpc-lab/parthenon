#=========================================================================================
# (C) (or copyright) 2020-2023. Triad National Security, LLC. All rights reserved.
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

# The following lines determine the default loop pattern by setting the
# default defines to the appropriate "tags" in the config.hpp file.
# See `kokkos_abstraction.hpp` for available tags and what they translate to.
if (Kokkos_ENABLE_CUDA OR Kokkos_ENABLE_HIP)
  set(PAR_LOOP_LAYOUT "MANUAL1D_LOOP" CACHE STRING
    "Default loop layout for parallel_for wrapper")

  set(PAR_LOOP_LAYOUT_VALUES "MANUAL1D_LOOP;MDRANGE_LOOP;TPTTR_LOOP;TPTTRTVR_LOOP"
    CACHE STRING "Possible loop layout options.")

  set(PAR_LOOP_INNER_LAYOUT "TVR_INNER_LOOP" CACHE STRING
    "Default loop layout for par_for_inner wrapper")

  set(PAR_LOOP_INNER_LAYOUT_VALUES "TVR_INNER_LOOP"
    CACHE STRING "Possible inner loop layout options.")

elseif(Kokkos_ENABLE_HPX)
  message( FATAL_ERROR "Need to add/fix/test default loop layouts for HPX backend.")

else()
  # use simd for loop when running on host
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

set(COORDINATE_TYPE UniformCartesian) # TODO: Make this an option when more are available

configure_file(config.hpp.in generated/config.hpp @ONLY)

add_library(parthenon
  bvals/comms/bvals_in_one.hpp
  bvals/comms/bvals_utils.hpp
  bvals/comms/build_boundary_buffers.cpp
  bvals/comms/bnd_info.cpp
  bvals/comms/bnd_info.hpp
  bvals/comms/boundary_communication.cpp
  bvals/comms/flux_correction.cpp 
  bvals/comms/tag_map.cpp 
  bvals/comms/tag_map.hpp 
  
  bvals/boundary_conditions_generic.hpp
  bvals/boundary_conditions.cpp
  bvals/boundary_conditions.hpp

  bvals/bvals.cpp
  bvals/bvals.hpp
  bvals/bvals_base.cpp
  bvals/bvals_interfaces.hpp
  bvals/boundary_flag.cpp
  bvals/bvals_var.cpp
  bvals/bvals_swarm.cpp

  coordinates/coordinates.hpp
  coordinates/uniform_cartesian.hpp

  driver/driver.cpp
  driver/driver.hpp
  driver/multistage.cpp
  driver/multistage.hpp

  interface/data_collection.cpp
  interface/data_collection.hpp
  interface/mesh_data.hpp
  interface/meshblock_data.cpp
  interface/meshblock_data.hpp
  interface/swarm_container.cpp
  interface/swarm.cpp
  interface/swarm.hpp
  interface/swarm_boundaries.hpp
  interface/swarm_device_context.hpp
  interface/make_pack_descriptor.hpp
  interface/metadata.cpp
  interface/metadata.hpp
  interface/packages.hpp
  interface/params.cpp
  interface/params.hpp
  interface/sparse_pack.hpp
  interface/sparse_pack_base.cpp
  interface/sparse_pack_base.hpp
  interface/sparse_pool.cpp
  interface/sparse_pool.hpp
  interface/state_descriptor.hpp
  interface/state_descriptor.cpp
  interface/update.cpp
  interface/update.hpp
  interface/var_id.hpp
  interface/variable_pack.hpp
  interface/variable_state.hpp
  interface/variable_state.cpp
  interface/variable.cpp
  interface/variable.hpp

  mesh/amr_loadbalance.cpp
  mesh/domain.hpp
  mesh/logical_location.cpp
  mesh/logical_location.hpp
  mesh/mesh_refinement.cpp
  mesh/mesh_refinement.hpp
  mesh/mesh-gmg.cpp
  mesh/mesh.cpp
  mesh/mesh.hpp
  mesh/meshblock.hpp
  mesh/meshblock_pack.hpp
  mesh/meshblock_tree.cpp
  mesh/meshblock_tree.hpp
  mesh/meshblock.cpp

  outputs/ascent.cpp
  outputs/history.cpp
  outputs/io_wrapper.cpp
  outputs/io_wrapper.hpp
  outputs/output_utils.cpp
  outputs/output_utils.hpp
  outputs/outputs.cpp
  outputs/outputs.hpp
  outputs/parthenon_hdf5.cpp
  outputs/parthenon_xdmf.cpp
  outputs/parthenon_hdf5.hpp
  outputs/parthenon_xdmf.hpp
  outputs/restart.cpp
  outputs/vtk.cpp

  parthenon/driver.hpp
  parthenon/package.hpp
  parthenon/parthenon.hpp
  parthenon/prelude.hpp

  pgen/default_pgen.cpp

  prolong_restrict/pr_loops.hpp
  prolong_restrict/pr_ops.hpp
  prolong_restrict/prolong_restrict.cpp
  prolong_restrict/prolong_restrict.hpp

  reconstruct/dc_inline.hpp
  reconstruct/plm_inline.hpp

  amr_criteria/amr_criteria.cpp
  amr_criteria/amr_criteria.hpp
  amr_criteria/refinement_package.cpp
  amr_criteria/refinement_package.hpp

  solvers/solver_utils.hpp

  tasks/task_id.cpp
  tasks/task_id.hpp
  tasks/task_list.hpp
  tasks/task_types.hpp

  time_integration/butcher_integrator.cpp
  time_integration/low_storage_integrator.cpp
  time_integration/staged_integrator.cpp
  time_integration/staged_integrator.hpp

  utils/alias_method.cpp
  utils/alias_method.hpp
  utils/array_to_tuple.hpp
  utils/buffer_utils.cpp
  utils/buffer_utils.hpp
  utils/change_rundir.cpp
  utils/communication_buffer.hpp
  utils/cleantypes.hpp
  utils/concepts_lite.hpp
  utils/error_checking.hpp
  utils/error_checking.cpp
  utils/hash.hpp
  utils/indexer.hpp
  utils/loop_utils.hpp
  utils/morton_number.hpp
  utils/mpi_types.hpp
  utils/multi_pointer.hpp
  utils/nan_payload_tag.hpp
  utils/object_pool.hpp
  utils/partition_stl_containers.hpp
  utils/reductions.hpp
  utils/show_config.cpp
  utils/signal_handler.cpp
  utils/sort.hpp
  utils/string_utils.cpp
  utils/string_utils.hpp
  utils/unique_id.cpp
  utils/unique_id.hpp
  utils/utils.hpp

  argument_parser.hpp
  basic_types.hpp
  defs.hpp
  globals.cpp
  globals.hpp
  kokkos_abstraction.hpp
  parameter_input.cpp
  parameter_input.hpp
  parthenon_array_generic.hpp
  parthenon_manager.cpp
  parthenon_manager.hpp
  parthenon_mpi.hpp
)
add_library(Parthenon::parthenon ALIAS parthenon)

set_target_properties(parthenon PROPERTIES SOVERSION ${parthenon_VERSION})

target_compile_features(parthenon PUBLIC cxx_std_17)

# This will automatically link any extra libraries (or no extra libraries)
# depending on the compiler, so it doesn't require an if() statement
target_link_libraries(parthenon PUBLIC std::filesystem)
target_compile_definitions(parthenon PRIVATE
    FS_HEADER=<${CXX_FILESYSTEM_HEADER}>
    FS_NAMESPACE=${CXX_FILESYSTEM_NAMESPACE}
    )

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

# For Cuda with NVCC (<11.2) and C++17 Kokkos currently does not work/compile with
# relaxed-constexpr, see https://github.com/kokkos/kokkos/issues/3496
# However, Parthenon heavily relies on it and there is no harm in compiling Kokkos
# without and Parthenon with (via Max Katz on the Kokkos Slack channel).
# Therefore, we don't use the Kokkos_ENABLE_CUDA_CONSTEXPR option add the flag manually.
# Also, not checking for NVIDIA as nvcc_wrapper is identified as GNU so we just make sure
# the flag is not added when compiling with Clang for Cuda.
if (Kokkos_ENABLE_CUDA AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
   target_compile_options(parthenon PUBLIC --expt-relaxed-constexpr)
endif()

target_link_libraries(parthenon PUBLIC Kokkos::kokkos)

if (PARTHENON_ENABLE_ASCENT)
  if (ENABLE_MPI)
    target_link_libraries(parthenon PUBLIC ascent::ascent_mpi)
  else()
    target_link_libraries(parthenon PUBLIC ascent::ascent)
  endif()
  # From the Ascent doc (last checked 06 Feb 2023 - Ascent commit 33538e3):
  # we need to make sure CUDA_RESOLVE_DEVICE_SYMBOLS is on for our target
  # (it propgates some way magically in 3.14, but not in 3.21)
  # TODO(pgrete) check if this has any performance implications
  if(CMAKE_CUDA_COMPILER)
    set_property(TARGET parthenon PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  endif()
endif()

lint_target(parthenon)

target_include_directories(parthenon PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/parthenon>
  )

install(TARGETS parthenon EXPORT parthenonTargets
  INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/parthenon"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
)

# Maintain directory structure in installed include files
install(DIRECTORY . DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/parthenon" FILES_MATCHING PATTERN "*.hpp")

# Install generated config header file
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/generated/config.hpp
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/parthenon")

install(FILES ${PROJECT_BINARY_DIR}/cmake/parthenonConfig.cmake DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/parthenon")

install(EXPORT parthenonTargets
    FILE parthenonTargets.cmake
    NAMESPACE Parthenon::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/parthenon"
)
