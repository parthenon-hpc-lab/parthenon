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

# The header files are added to parthenon for linting
list(APPEND PARTHENON_PUBLIC_HEADERS
  ${CMAKE_CURRENT_SOURCE_DIR}/bvals/cc/bvals_cc.hpp

  ${CMAKE_CURRENT_SOURCE_DIR}/bvals/fc/bvals_fc.hpp

  ${CMAKE_CURRENT_SOURCE_DIR}/bvals/boundary_conditions.hpp

  ${CMAKE_CURRENT_SOURCE_DIR}/bvals/bvals.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/bvals/bvals_interfaces.hpp

  ${CMAKE_CURRENT_SOURCE_DIR}/coordinates/coordinates.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/coordinates/uniform_cartesian.hpp

  ${CMAKE_CURRENT_SOURCE_DIR}/driver/driver.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/driver/multistage.hpp

  ${CMAKE_CURRENT_SOURCE_DIR}/interface/container_collection.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/interface/container_iterator.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/interface/container.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/interface/metadata.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/interface/params.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/interface/properties_interface.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/interface/sparse_variable.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/interface/update.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/interface/variable_pack.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/interface/variable.hpp

  ${CMAKE_CURRENT_SOURCE_DIR}/mesh/domain.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mesh/meshblock_pack.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mesh/mesh_refinement.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mesh/mesh.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mesh/meshblock_tree.hpp

  ${CMAKE_CURRENT_SOURCE_DIR}/outputs/io_wrapper.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/outputs/outputs.hpp

  ${CMAKE_CURRENT_SOURCE_DIR}/parthenon/driver.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/parthenon/package.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/parthenon/parthenon.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/parthenon/prelude.hpp

  ${CMAKE_CURRENT_SOURCE_DIR}/reconstruct/dc_inline.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/reconstruct/plm_inline.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/reconstruct/reconstruction.hpp

  ${CMAKE_CURRENT_SOURCE_DIR}/refinement/amr_criteria.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/refinement/refinement.hpp

  ${CMAKE_CURRENT_SOURCE_DIR}/tasks/task_id.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/tasks/task_list.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/tasks/task_types.hpp

  ${CMAKE_CURRENT_SOURCE_DIR}/utils/buffer_utils.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/utils/error_checking.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/utils/trim_string.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/utils/utils.hpp

  ${CMAKE_CURRENT_SOURCE_DIR}/argument_parser.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/basic_types.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/defs.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/globals.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/kokkos_abstraction.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/parameter_input.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/parthenon_manager.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/parthenon_mpi.hpp
)

