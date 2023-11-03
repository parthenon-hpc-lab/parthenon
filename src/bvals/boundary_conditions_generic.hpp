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

#ifndef BVALS_BOUNDARY_CONDITIONS_GENERIC_HPP_
#define BVALS_BOUNDARY_CONDITIONS_GENERIC_HPP_

#include <functional>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "basic_types.hpp"
#include "interface/make_pack_descriptor.hpp"
#include "interface/meshblock_data.hpp"
#include "interface/sparse_pack.hpp"
#include "mesh/domain.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "utils/block_timer.hpp"

namespace parthenon {
namespace BoundaryFunction {

enum class BCSide { Inner, Outer };
enum class BCType { Outflow, Reflect, ConstantDeriv, Fixed };

template <CoordinateDirection DIR, BCSide SIDE, BCType TYPE, class... var_ts>
void GenericBC(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse,
               TopologicalElement el, Real val) {
  // make sure DIR is X[123]DIR so we don't have to check again
  static_assert(DIR == X1DIR || DIR == X2DIR || DIR == X3DIR, "DIR must be X[123]DIR");

  std::shared_ptr<MeshBlock> pmb = rc->GetBlockPointer();
  auto &block_cost_host = pmb->pmy_mesh->block_cost_host;
  BlockTimerHost host_timer(block_cost_host, pmb->lid, pmb->lid);

  // convenient shorthands
  constexpr bool X1 = (DIR == X1DIR);
  constexpr bool X2 = (DIR == X2DIR);
  constexpr bool X3 = (DIR == X3DIR);
  constexpr bool INNER = (SIDE == BCSide::Inner);

  std::vector<MetadataFlag> flags{Metadata::FillGhost};
  if (GetTopologicalType(el) == TopologicalType::Cell) flags.push_back(Metadata::Cell);
  if (GetTopologicalType(el) == TopologicalType::Face) flags.push_back(Metadata::Face);
  if (GetTopologicalType(el) == TopologicalType::Edge) flags.push_back(Metadata::Edge);
  if (GetTopologicalType(el) == TopologicalType::Node) flags.push_back(Metadata::Node);

  std::set<PDOpt> opts;
  if (coarse) opts = {PDOpt::Coarse};
  auto desc = MakePackDescriptor<var_ts...>(
      rc->GetBlockPointer()->pmy_mesh->resolved_packages.get(), flags, opts);
  auto q = desc.GetPack(rc.get());
  const int b = 0;
  const int lstart = q.GetLowerBoundHost(b);
  const int lend = q.GetUpperBoundHost(b);
  if (lend < lstart) return;
  auto nb = IndexRange{lstart, lend};

  const auto &bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;

  const auto &range = X1 ? bounds.GetBoundsI(IndexDomain::interior, el)
                         : (X2 ? bounds.GetBoundsJ(IndexDomain::interior, el)
                               : bounds.GetBoundsK(IndexDomain::interior, el));
  const int ref = INNER ? range.s : range.e;

  std::string label = (TYPE == BCType::Reflect ? "Reflect" : "Outflow");
  label += (INNER ? "Inner" : "Outer");
  label += "X" + std::to_string(DIR);

  constexpr IndexDomain domain =
      INNER ? (X1 ? IndexDomain::inner_x1
                  : (X2 ? IndexDomain::inner_x2 : IndexDomain::inner_x3))
            : (X1 ? IndexDomain::outer_x1
                  : (X2 ? IndexDomain::outer_x2 : IndexDomain::outer_x3));

  // used for reflections
  const int offset = 2 * ref + (INNER ? -1 : 1);

  // used for derivatives
  const int offsetin = INNER;
  const int offsetout = !INNER;
  // stop timing on the host
  host_timer.Stop();
  pmb->par_for_bndry(
      PARTHENON_AUTO_LABEL, nb, domain, el, coarse,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        if (TYPE == BCType::Reflect) {
          const bool reflect = (q(b, el, l).vector_component == DIR);
          q(b, el, l, k, j, i) =
              (reflect ? -1.0 : 1.0) *
              q(b, el, l, X3 ? offset - k : k, X2 ? offset - j : j, X1 ? offset - i : i);
        } else if (TYPE == BCType::ConstantDeriv) {
          Real dq = q(b, el, l, X3 ? ref + offsetin : k, X2 ? ref + offsetin : j,
                      X1 ? ref + offsetin : i) -
                    q(b, el, l, X3 ? ref - offsetout : k, X2 ? ref - offsetout : j,
                      X1 ? ref - offsetout : i);
          Real delta = 0.0;
          if (X1) {
            delta = i - ref;
          } else if (X2) {
            delta = j - ref;
          } else {
            delta = k - ref;
          }
          q(b, el, l, k, j, i) =
              q(b, el, l, X3 ? ref : k, X2 ? ref : j, X1 ? ref : i) + delta * dq;
        } else if (TYPE == BCType::Fixed) {
          q(b, el, l, k, j, i) = val;
        } else {
          q(b, el, l, k, j, i) = q(b, el, l, X3 ? ref : k, X2 ? ref : j, X1 ? ref : i);
        }
      });
}

template <CoordinateDirection DIR, BCSide SIDE, BCType TYPE, class... var_ts>
void GenericBC(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse, Real val = 0.0) {
  using TE = TopologicalElement;
  for (auto el : {TE::CC, TE::F1, TE::F2, TE::F3, TE::E1, TE::E2, TE::E3, TE::NN})
    GenericBC<DIR, SIDE, TYPE, var_ts...>(rc, coarse, el, val);
}

} // namespace BoundaryFunction
} // namespace parthenon

#endif // BVALS_BOUNDARY_CONDITIONS_GENERIC_HPP_
