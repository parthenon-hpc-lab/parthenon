//========================================================================================
// (C) (or copyright) 2020-2024. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001 for Los
// Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
// for the U.S. Department of Energy/National Nuclear Security Administration. All rights
// in the program are reserved by Triad National Security, LLC, and the U.S. Department
// of Energy/National Nuclear Security Administration. The Government is granted for
// itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do so.
//========================================================================================

#include "interface/variable_state.hpp"
#include "interface/metadata.hpp"

namespace parthenon {

VariableState::VariableState(const Metadata &md, int sparse_id, int pmb_gid,
                             const std::array<int, MAX_VARIABLE_DIMENSION> &dims) {
  allocation_threshold = md.GetAllocationThreshold();
  deallocation_threshold = md.GetDeallocationThreshold();
  sparse_default_val = md.GetDefaultValue();
  this->sparse_id = sparse_id;
  this->pmb_gid = pmb_gid;

  tensor_shape[0] = dims[3];
  tensor_shape[1] = dims[4];
  tensor_shape[2] = dims[5];
  tensor_components = tensor_shape[0] * tensor_shape[1] * tensor_shape[2];
  topological_type = GetTopologicalType(md);
}

} // namespace parthenon
