
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

# Configure defs.hpp
set(PROBLEM_GENERATOR "<not-implemented>") # TODO: Figure out what to put here
set(SINGLE_PRECISION_ENABLED 0) # TODO: Add CMake option for this

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

elseif(${Kokkos_ENABLE_HPX})
  message( FATAL_ERROR "Need to add/fix/test default loop layouts for HPX backend.")

else()
  # use simd for loop when not using Nvidia GPUs
  set(PAR_LOOP_LAYOUT "SIMDFOR_LOOP" CACHE STRING
    "Default loop layout for parallel_for wrapper")
  set(PAR_LOOP_LAYOUT_VALUES "SIMDFOR_LOOP;MANUAL1D_LOOP;MDRANGE_LOOP;TPTTR_LOOP;TPTVR_LOOP;TPTTRTVR_LOOP"
    CACHE STRING "Possible loop layout options.")

endif()

set_property(CACHE PAR_LOOP_LAYOUT PROPERTY STRINGS ${PAR_LOOP_LAYOUT_VALUES})

message(STATUS "PAR_LOOP_LAYOUT='${PAR_LOOP_LAYOUT}' (default par_for wrapper layout)")

set(EXCEPTION_HANDLING_OPTION ENABLE_EXCEPTIONS) # TODO: Add option to disable exceptions
set(COMPILED_WITH ${CMAKE_CXX_COMPILER})
set(COMPILER_COMMAND "<not-implemented>") # TODO: Put something more descriptive here
set(COMPILER_FLAGS "<not-implemented>") # TODO: Put something more descriptive here

set(NHYDRO_VARIABLES 0) # TODO: Remove
set(NFIELD_VARIABLES 0) # TODO: Remove
set(NWAVE_VALUE 5) # TODO: Remove
set(NUMBER_GHOST_CELLS 2) # TODO: Make this a CMake option and check it

configure_file(defs.hpp.in generated/defs.hpp @ONLY)

add_library(parthenon
  better_boundaries/boundary_conditions.cpp

  better_refinement/better_refinement.cpp

  bvals/cc/bvals_cc.cpp
  bvals/cc/flux_correction_cc.cpp

  bvals/fc/bvals_fc.cpp
  bvals/fc/flux_correction_fc.cpp

  bvals/bvals.cpp
  bvals/bvals_base.cpp
  bvals/boundary_flag.cpp
  bvals/bvals_refine.cpp
  bvals/bvals_var.cpp

  coordinates/coordinates.cpp
  coordinates/cartesian.cpp

  driver/driver.cpp
  driver/multistage.cpp

  interface/Container.cpp
  interface/PropertiesInterface.cpp
  interface/MaterialVariable.cpp
  interface/Stage.cpp
  interface/Update.cpp
  interface/Variable.cpp

  mesh/amr_loadbalance.cpp
  mesh/mesh_refinement.cpp
  mesh/mesh.cpp
  mesh/meshblock.cpp
  mesh/meshblock_tree.cpp
  mesh/weighted_ave.cpp

  outputs/athena_hdf5.cpp
  outputs/athena_hdf5_C.cpp
  outputs/formatted_table.cpp
  outputs/history.cpp
  outputs/io_wrapper.cpp
  outputs/outputs.cpp
  outputs/restart.cpp
  outputs/vtk.cpp

  pgen/default_pgen.cpp

  reconstruct/characteristic.cpp
  reconstruct/dc_simple.cpp
  reconstruct/dc.cpp
  reconstruct/plm_simple.cpp
  reconstruct/plm.cpp
  reconstruct/ppm_simple.cpp
  reconstruct/ppm.cpp
  reconstruct/reconstruction.cpp

  task_list/task_id.cpp

  utils/buffer_utils.cpp
  utils/change_rundir.cpp
  #utils/gl_quadrature.cpp
  #utils/ran2.cpp
  utils/show_config.cpp
  utils/signal_handler.cpp

  globals.cpp
  parameter_input.cpp
  parthenon_manager.cpp
)
target_compile_features(parthenon PUBLIC cxx_std_14)

if (ENABLE_MPI)
  target_link_libraries(parthenon PUBLIC MPI::MPI_CXX)
endif()

if (ENABLE_OPENMP)
  target_link_libraries(parthenon PUBLIC OpenMP::OpenMP_CXX)
endif()

if (ENABLE_HDF5)
  target_link_libraries(parthenon PUBLIC HDF5_C)
endif()

target_link_libraries(parthenon PUBLIC Kokkos::kokkos)

target_include_directories(parthenon PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}/generated
)
