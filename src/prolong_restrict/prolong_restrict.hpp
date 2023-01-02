//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020-2022 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2022. Triad National Security, LLC. All rights reserved.
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

#ifndef PROLONG_RESTRICT_PROLONG_RESTRICT_HPP_
#define PROLONG_RESTRICT_PROLONG_RESTRICT_HPP_

#include <algorithm>
#include <functional> // std::function
#include <string>     // std::string
#include <tuple>      // std::tuple
#include <typeinfo>   // typeid
#include <utility>    // std::forward
#include <vector>

#include "bvals/cc/bvals_cc_in_one.hpp" // for buffercache_t
#include "coordinates/coordinates.hpp"  // for coordinates
#include "globals.hpp"                  // for Globals
#include "mesh/domain.hpp"              // for IndexShape
#include "prolong_restrict/pr_loops.hpp"
#include "prolong_restrict/pr_ops.hpp"

namespace parthenon {
template <typename T>
class MeshData; // forward declaration
class StateDescriptor;
namespace cell_centered_bvars {
struct BvarsSubCache_t;
} // namespace cell_centered_bvars
namespace refinement {

// TODO(JMM): Add a prolongate when prolongation is called in-one
// TODO(JMM): Is this actually the API we want?
void Restrict(const StateDescriptor *resolved_packages,
              const cell_centered_bvars::BvarsSubCache_t &cache,
              const IndexShape &cellbnds, const IndexShape &c_cellbnds);

// std::function closures for the top-level restriction functions The
// existence of host/device overloads here allows us to avoid a
// deep-copy in the per-meshblock
// calls
// TODO(JMM): I don't love having
// two overloads here.  However when we shift entirely to in-one
// machinery, the info_h only overload will go away.
using Restrictor_t = std::function<void(
    const cell_centered_bvars::BufferCache_t &,
    const cell_centered_bvars::BufferCacheHost_t &, const loops::Idx_t &,
    const loops::IdxHost_t &, const IndexShape &, const IndexShape &, const std::size_t)>;
using RestrictorHost_t = std::function<void(
    const cell_centered_bvars::BufferCacheHost_t &, const loops::IdxHost_t &,
    const IndexShape &, const IndexShape &, const std::size_t)>;
using Prolongator_t = std::function<void(
    const cell_centered_bvars::BufferCache_t &,
    const cell_centered_bvars::BufferCacheHost_t &, const loops::Idx_t &,
    const loops::IdxHost_t &, const IndexShape &, const IndexShape &, const std::size_t)>;
using ProlongatorHost_t = std::function<void(
    const cell_centered_bvars::BufferCacheHost_t &, const loops::IdxHost_t &,
    const IndexShape &, const IndexShape &, const std::size_t)>;

// Container struct owning refinement functions/closures.
// this container needs to be uniquely hashable, and always the same
// given a registered set of Op functors. To handle this, we store a
// function of the type_ids of the registered ProlongationOp and
// RestrictionOp.
struct RefinementFunctions_t {
  RefinementFunctions_t() = default;
  explicit RefinementFunctions_t(const std::string &label) : label_(label) {}

  template <class ProlongationOp, class RestrictionOp>
  static RefinementFunctions_t RegisterOps() {
    // We use the specialization to dim = 1 for this, but any int
    // specialization will do.
    const std::string label = std::string(typeid(ProlongationOp).name()) +
                              std::string(" and ") +
                              std::string(typeid(RestrictionOp).name());

    RefinementFunctions_t funcs(label);
    funcs.restrictor = [](const cell_centered_bvars::BufferCache_t &info,
                          const cell_centered_bvars::BufferCacheHost_t &info_h,
                          const loops::Idx_t &idxs, const loops::IdxHost_t &idxs_h,
                          const IndexShape &cellbnds, const IndexShape &c_cellbnds,
                          const std::size_t nbuffers) {
      loops::DoProlongationRestrictionOp<RestrictionOp>(
          cellbnds, info, info_h, idxs, idxs_h, cellbnds, c_cellbnds,
          RefinementOp_t::Restriction, nbuffers);
    };
    funcs.restrictor_host = [](const cell_centered_bvars::BufferCacheHost_t &info_h,
                               const loops::IdxHost_t &idxs_h, const IndexShape &cellbnds,
                               const IndexShape &c_cellbnds, const std::size_t nbuffers) {
      loops::DoProlongationRestrictionOp<RestrictionOp>(
          cellbnds, info_h, idxs_h, cellbnds, c_cellbnds, RefinementOp_t::Restriction,
          nbuffers);
    };
    funcs.prolongator = [](const cell_centered_bvars::BufferCache_t &info,
                           const cell_centered_bvars::BufferCacheHost_t &info_h,
                           const loops::Idx_t &idxs, const loops::IdxHost_t &idxs_h,
                           const IndexShape &cellbnds, const IndexShape &c_cellbnds,
                           const std::size_t nbuffers) {
      loops::DoProlongationRestrictionOp<ProlongationOp>(
          cellbnds, info, info_h, idxs, idxs_h, cellbnds, c_cellbnds,
          RefinementOp_t::Prolongation, nbuffers);
    };
    funcs.prolongator_host = [](const cell_centered_bvars::BufferCacheHost_t &info_h,
                                const loops::IdxHost_t &idxs_h,
                                const IndexShape &cellbnds, const IndexShape &c_cellbnds,
                                const std::size_t nbuffers) {
      loops::DoProlongationRestrictionOp<ProlongationOp>(
          cellbnds, info_h, idxs_h, cellbnds, c_cellbnds, RefinementOp_t::Prolongation,
          nbuffers);
    };
    return funcs;
  }
  std::string label() const { return label_; }
  bool operator==(const RefinementFunctions_t &other) const {
    return (label() == other.label());
  }

  Restrictor_t restrictor;
  RestrictorHost_t restrictor_host;
  Prolongator_t prolongator;
  ProlongatorHost_t prolongator_host;

 private:
  // TODO(JMM): This could be a type_info::hash instead of a string,
  // which might be a little bit more memory efficient, but I think
  // using the label might be useful for debugging and it's also
  // easier to concatenate.
  std::string label_;
};

struct RefinementFunctionsHasher {
  auto operator()(const RefinementFunctions_t &f) const {
    return std::hash<std::string>{}(f.label());
  }
};

} // namespace refinement
} // namespace parthenon

#endif // PROLONG_RESTRICT_PROLONG_RESTRICT_HPP_
