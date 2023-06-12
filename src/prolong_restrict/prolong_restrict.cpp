//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020-2022 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2023. Triad National Security, LLC. All rights reserved.
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

#include "bvals/comms/bnd_info.hpp"
#include "interface/mesh_data.hpp"
#include "interface/state_descriptor.hpp"
#include "kokkos_abstraction.hpp"
#include "prolong_restrict/pr_ops.hpp"
#include "prolong_restrict/prolong_restrict.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {
namespace refinement {

// TODO(JMM): Add a prolongate when prolongation is called in-one
// TODO(JMM): Is this actually the API we want?
void Restrict(const StateDescriptor *resolved_packages, const BvarsSubCache_t &cache,
              const IndexShape &cellbnds, const IndexShape &c_cellbnds) {
  const auto &ref_func_map = resolved_packages->RefinementFncsToIDs();
  for (const auto &[func, idx] : ref_func_map) {
    auto restrictor = func.restrictor;
    PARTHENON_DEBUG_REQUIRE_THROWS(restrictor != nullptr, "Valid restriction op");
    loops::Idx_t subset = Kokkos::subview(cache.buffer_subsets, idx, Kokkos::ALL());
    loops::IdxHost_t subset_h =
        Kokkos::subview(cache.buffer_subsets_h, idx, Kokkos::ALL());
    restrictor(cache.bnd_info, cache.bnd_info_h, subset, subset_h, cellbnds, c_cellbnds,
               cache.buffer_subset_sizes[idx]);
  }
}

void Prolongate(const StateDescriptor *resolved_packages, const BvarsSubCache_t &cache,
                const IndexShape &cellbnds, const IndexShape &c_cellbnds) {
  const auto &ref_func_map = resolved_packages->RefinementFncsToIDs();
  for (const auto &[func, idx] : ref_func_map) {
    auto prolongator = func.prolongator;
    PARTHENON_DEBUG_REQUIRE_THROWS(prolongator != nullptr, "Invalid prolongation op");
    loops::Idx_t subset = Kokkos::subview(cache.buffer_subsets, idx, Kokkos::ALL());
    loops::IdxHost_t subset_h =
        Kokkos::subview(cache.buffer_subsets_h, idx, Kokkos::ALL());
    prolongator(cache.bnd_info, cache.bnd_info_h, subset, subset_h, cellbnds, c_cellbnds,
                cache.buffer_subset_sizes[idx]);
  }
}

void ProlongateInternal(const StateDescriptor *resolved_packages,
                        const BvarsSubCache_t &cache, const IndexShape &cellbnds,
                        const IndexShape &c_cellbnds) {
  const auto &ref_func_map = resolved_packages->RefinementFncsToIDs();
  for (const auto &[func, idx] : ref_func_map) {
    auto internal_prolongator = func.internal_prolongator;
    PARTHENON_DEBUG_REQUIRE_THROWS(internal_prolongator != nullptr,
                                   "Invalid prolongation op");
    loops::Idx_t subset = Kokkos::subview(cache.buffer_subsets, idx, Kokkos::ALL());
    loops::IdxHost_t subset_h =
        Kokkos::subview(cache.buffer_subsets_h, idx, Kokkos::ALL());
    internal_prolongator(cache.bnd_info, cache.bnd_info_h, subset, subset_h, cellbnds,
                         c_cellbnds, cache.buffer_subset_sizes[idx]);
  }
}

} // namespace refinement
} // namespace parthenon
