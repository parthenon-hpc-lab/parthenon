
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

configure_file(defs.hpp.in generated/defs.hpp @ONLY)

add_library(parthenon
  bvals/cc/bvals_cc.cpp
  bvals/cc/bvals_cc.hpp
  bvals/cc/flux_correction_cc.cpp

  bvals/fc/bvals_fc.cpp
  bvals/fc/bvals_fc.hpp
  bvals/fc/flux_correction_fc.cpp

  bvals/boundary_conditions.cpp
  bvals/boundary_conditions.hpp

  bvals/bvals.cpp
  bvals/bvals.hpp
  bvals/bvals_base.cpp
  bvals/bvals_interfaces.hpp
  bvals/boundary_flag.cpp
  bvals/bvals_refine.cpp
  bvals/bvals_var.cpp

  coordinates/coordinates.hpp
  coordinates/uniform_cartesian.hpp

  driver/driver.cpp
  driver/driver.hpp
  driver/multistage.cpp
  driver/multistage.hpp

  interface/container_collection.cpp
  interface/container_collection.hpp
  interface/container_iterator.hpp
  interface/container.cpp
  interface/container.hpp
  interface/metadata.cpp
  interface/metadata.hpp
  interface/params.hpp
  interface/properties_interface.cpp
  interface/properties_interface.hpp
  interface/sparse_variable.cpp
  interface/sparse_variable.hpp
  interface/update.cpp
  interface/update.hpp
  interface/variable_pack.hpp
  interface/variable.cpp
  interface/variable.hpp

  mesh/amr_loadbalance.cpp
  mesh/domain.hpp
  mesh/mesh_refinement.cpp
  mesh/mesh_refinement.hpp
  mesh/mesh.cpp
  mesh/mesh.hpp
  mesh/meshblock_tree.cpp
  mesh/meshblock_tree.hpp
  mesh/meshblock.cpp
  mesh/weighted_ave.cpp

  outputs/formatted_table.cpp
  outputs/history.cpp
  outputs/io_wrapper.cpp
  outputs/io_wrapper.hpp
  outputs/outputs.cpp
  outputs/outputs.hpp
  outputs/parthenon_hdf5.cpp
  outputs/restart.cpp
  outputs/vtk.cpp

  parthenon/driver.hpp
  parthenon/package.hpp
  parthenon/parthenon.hpp
  parthenon/prelude.hpp

  pgen/default_pgen.cpp

  reconstruct/characteristic.cpp
  reconstruct/dc_inline.hpp
  reconstruct/dc.cpp
  reconstruct/plm_inline.hpp
  reconstruct/plm.cpp
  reconstruct/ppm.cpp
  reconstruct/reconstruction.hpp
  reconstruct/reconstruction.cpp

  refinement/amr_criteria.cpp
  refinement/amr_criteria.hpp
  refinement/refinement.cpp
  refinement/refinement.hpp

  task_list/tasks.cpp
  task_list/tasks.hpp

  utils/buffer_utils.cpp
  utils/buffer_utils.hpp
  utils/change_rundir.cpp
  utils/error_checking.hpp
  utils/show_config.cpp
  utils/signal_handler.cpp
  utils/trim_string.cpp
  utils/trim_string.hpp
  utils/utils.hpp

  argument_parser.hpp
  athena.hpp
  basic_types.hpp
  globals.cpp
  globals.hpp
  kokkos_abstraction.hpp
  parameter_input.cpp
  parameter_input.hpp
  parthenon_manager.cpp
  parthenon_manager.hpp
  parthenon_mpi.hpp
)

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
