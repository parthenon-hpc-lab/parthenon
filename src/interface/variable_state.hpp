//========================================================================================
// (C) (or copyright) 2020-2022. Triad National Security, LLC. All rights reserved.
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
#ifndef INTERFACE_VARIABLE_STATE_HPP_
#define INTERFACE_VARIABLE_STATE_HPP_

#include <limits>

#include "basic_types.hpp"
#include "defs.hpp"
#include "globals.hpp"
#include "parthenon_arrays.hpp"

namespace parthenon {

// forward declaration
class Metadata;

static constexpr int InvalidSparseID = std::numeric_limits<int>::min();

struct VariableState : public empty_state_t {
  explicit VariableState(
      const Metadata &md, int lid, int sparse_id = InvalidSparseID,
      const std::array<int, MAX_VARIABLE_DIMENSION> &dims = [] {
        std::array<int, MAX_VARIABLE_DIMENSION> d;
        for (int i = 0; i < MAX_VARIABLE_DIMENSION; ++i)
          d[i] = 1;
        return d;
      }());

  KOKKOS_INLINE_FUNCTION
  VariableState(Real alloc, Real dealloc, Real sparse_default_val = 0.0,
                int sparse_id = InvalidSparseID)
      : allocation_threshold(alloc), deallocation_threshold(dealloc),
        sparse_default_val(sparse_default_val), sparse_id(sparse_id),
        lid(lid) {}

  KOKKOS_INLINE_FUNCTION
  VariableState(Real alloc, Real dealloc, int sparse_id)
      : allocation_threshold(alloc), deallocation_threshold(dealloc),
        sparse_default_val(0.0), sparse_id(sparse_id) {}

  KOKKOS_DEFAULTED_FUNCTION
  VariableState() = default;

  KOKKOS_INLINE_FUNCTION
  explicit VariableState(const empty_state_t &)
      : VariableState(0.0, 0.0, 0.0, InvalidSparseID) {}

  Real allocation_threshold;
  Real deallocation_threshold;
  Real sparse_default_val;
  int sparse_id, lid;
  int vector_component = NODIR;
  bool initialized = true;

  TopologicalType topological_type = TopologicalType::Cell;
  std::size_t tensor_components;
  std::size_t tensor_shape[3];
};

} // namespace parthenon
#endif // INTERFACE_VARIABLE_STATE_HPP_
