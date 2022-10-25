//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020-2022 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2021. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001
// for Los Alamos National Laboratory (LANL), which is operated by Triad
// National Security, LLC for the U.S. Department of Energy/National Nuclear
// Security Administration. All rights in the program are reserved by Triad
// National Security, LLC, and the U.S. Department of Energy/National Nuclear
// Security Administration. The Government is granted for itself and others
// acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license
// in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do
// so.
//========================================================================================

#include <algorithm>
#include <tuple> // std::tuple
#include <utility>

#include "bvals/cc/bnd_info.hpp"
#include "interface/mesh_data.hpp"
#include "interface/state_descriptor.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/mesh_refinement_ops.hpp"
#include "mesh/refinement_in_one.hpp"

namespace parthenon {
namespace refinement {

// TODO(JMM): Add a prolongate when prolongation is called in-one
// TODO(JMM): Is this actually the API we want?
void Restrict(const StateDescriptor *resolved_packages,
	      const cell_centered_bvars::BvarsSubCache_t &cache,
	      const IndexShape &cellbnds, const IndexShape &c_cellbnds) {
  const auto &ref_func_map = resolved_packages->RefinementFncsToIDs();
  for (const auto &[func,idx] : ref_func_map) {
    auto restrictor = func.restrictor;
    loops::Idx_t subset = cache.buffer_subsets.Slice(idx, Kokkos::ALL());
    loops::IdxHost_t subset_h = cache.buffer_subsets_h.Slice(idx, Kokkos::ALL());
    restrictor(cache.bnd_info, cache.bnd_info_h, subset, subset_h,
	       cellbnds, c_cellbnds, cache.buffer_subset_sizes[idx]);
  }
}

} // namespace refinement
} // namespace parthenon
