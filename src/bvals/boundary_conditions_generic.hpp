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

#ifndef BVALS_BOUNDARY_CONDITIONS_GENERIC_HPP_
#define BVALS_BOUNDARY_CONDITIONS_GENERIC_HPP_

#include <functional>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "basic_types.hpp"
#include "interface/make_pack_descriptor.hpp"
#include "interface/meshblock_data.hpp"
#include "interface/sparse_pack.hpp"
#include "mesh/domain.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"

namespace parthenon {
namespace BoundaryFunction {

enum class BCSide { Inner, Outer };
enum class BCType { Outflow, Reflect, ConstantDeriv, Fixed, FixedFace, Periodic };

// TODO(BRR) add support for specific swarms?
template <CoordinateDirection DIR, BCSide SIDE, BCType TYPE>
void GenericSwarmBC(std::shared_ptr<Swarm> &swarm) {
  // make sure DIR is X[123]DIR so we don't have to check again
  static_assert(DIR == X1DIR || DIR == X2DIR || DIR == X3DIR, "DIR must be X[123]DIR");

  auto swarm_d_ = swarm->GetDeviceContext();
  int max_active_index = swarm->GetMaxActiveIndex();

  auto pmb = swarm->GetBlockPointer();

  auto &x_ = swarm->Get<Real>("x").Get();
  auto &y_ = swarm->Get<Real>("y").Get();
  auto &z_ = swarm->Get<Real>("z").Get();

  // TODO(BRR) do something about all these if statements
  pmb->par_for(
      PARTHENON_AUTO_LABEL, 0, max_active_index, KOKKOS_LAMBDA(const int n) {
        // convenient shorthands
        [[maybe_unused]] constexpr bool X1 = (DIR == X1DIR);
        [[maybe_unused]] constexpr bool X2 = (DIR == X2DIR);
        [[maybe_unused]] constexpr bool X3 = (DIR == X3DIR);
        // Cannot capture variables inside constexpr if context
        [[maybe_unused]] const auto &x = x_;
        [[maybe_unused]] const auto &y = y_;
        [[maybe_unused]] const auto &z = z_;
        const auto &swarm_d = swarm_d_;
        constexpr bool INNER = (SIDE == BCSide::Inner);
        if (swarm_d.IsActive(n)) {
          if constexpr (X1) {
            if constexpr (INNER) {
              if constexpr (TYPE == BCType::Periodic) {
                if (x(n) > swarm_d.x_max_global_) {
                  x(n) = swarm_d.x_min_global_ + (x(n) - swarm_d.x_max_global_);
                }
              } else if constexpr (TYPE == BCType::Outflow) {
                if (x(n) < swarm_d.x_min_global_) {
                  swarm_d.MarkParticleForRemoval(n);
                }
              }
            } else {
              if constexpr (TYPE == BCType::Periodic) {
                if (x(n) < swarm_d.x_min_global_) {
                  x(n) = swarm_d.x_max_global_ - (swarm_d.x_min_global_ - x(n));
                }
              } else if constexpr (TYPE == BCType::Outflow) {
                if (x(n) > swarm_d.x_max_global_) {
                  swarm_d.MarkParticleForRemoval(n);
                }
              }
            }
          } else if constexpr (X2) {
            if constexpr (INNER) {
              if constexpr (TYPE == BCType::Periodic) {
                if (y(n) > swarm_d.y_max_global_) {
                  y(n) = swarm_d.y_min_global_ + (y(n) - swarm_d.y_max_global_);
                }
              } else if constexpr (TYPE == BCType::Outflow) {
                if (y(n) < swarm_d.y_min_global_) {
                  swarm_d.MarkParticleForRemoval(n);
                }
              }
            } else {
              if constexpr (TYPE == BCType::Periodic) {
                if (y(n) < swarm_d.y_min_global_) {
                  y(n) = swarm_d.y_max_global_ - (swarm_d.y_min_global_ - y(n));
                }
              } else if constexpr (TYPE == BCType::Outflow) {
                if (y(n) > swarm_d.y_max_global_) {
                  swarm_d.MarkParticleForRemoval(n);
                }
              }
            }
          } else if constexpr (X3) {
            if constexpr (INNER) {
              if constexpr (TYPE == BCType::Periodic) {
                if (z(n) > swarm_d.z_max_global_) {
                  z(n) = swarm_d.z_min_global_ + (z(n) - swarm_d.z_max_global_);
                }
              } else if constexpr (TYPE == BCType::Outflow) {
                if (z(n) < swarm_d.z_min_global_) {
                  swarm_d.MarkParticleForRemoval(n);
                }
              }
            } else {
              if constexpr (TYPE == BCType::Periodic) {
                if (z(n) < swarm_d.z_min_global_) {
                  z(n) = swarm_d.z_max_global_ - (swarm_d.z_min_global_ - z(n));
                }
              } else if constexpr (TYPE == BCType::Outflow) {
                if (z(n) > swarm_d.z_max_global_) {
                  swarm_d.MarkParticleForRemoval(n);
                }
              }
            }
          }
        }
      });
}

namespace impl {
using desc_key_t = std::tuple<bool, TopologicalType>;
template <class... var_ts>
using map_bc_pack_descriptor_t =
    std::unordered_map<desc_key_t, typename SparsePack<var_ts...>::Descriptor,
                       tuple_hash<desc_key_t>>;

template <class... var_ts>
map_bc_pack_descriptor_t<var_ts...>
GetPackDescriptorMap(std::shared_ptr<MeshBlockData<Real>> &rc) {
  std::vector<std::pair<TopologicalType, MetadataFlag>> elements{
      {TopologicalType::Cell, Metadata::Cell},
      {TopologicalType::Face, Metadata::Face},
      {TopologicalType::Edge, Metadata::Edge},
      {TopologicalType::Node, Metadata::Node}};
  map_bc_pack_descriptor_t<var_ts...> my_map;
  for (auto [tt, md] : elements) {
    std::vector<MetadataFlag> flags{Metadata::FillGhost};
    flags.push_back(md);
    std::set<PDOpt> opts{PDOpt::Coarse};
    my_map.emplace(std::make_pair(desc_key_t{true, tt},
                                  MakePackDescriptor<var_ts...>(rc.get(), flags, opts)));
    my_map.emplace(std::make_pair(desc_key_t{false, tt},
                                  MakePackDescriptor<var_ts...>(rc.get(), flags)));
  }
  return my_map;
}
} // namespace impl

template <CoordinateDirection DIR, BCSide SIDE, BCType TYPE, class... var_ts>
void GenericBC(std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse,
               TopologicalElement el_, Real val_) {
  // make sure DIR is X[123]DIR so we don't have to check again
  static_assert(DIR == X1DIR || DIR == X2DIR || DIR == X3DIR, "DIR must be X[123]DIR");

  // convenient shorthands
  constexpr bool X1 = (DIR == X1DIR);
  constexpr bool X2 = (DIR == X2DIR);
  constexpr bool X3 = (DIR == X3DIR);
  constexpr bool INNER = (SIDE == BCSide::Inner);

  static auto descriptors = impl::GetPackDescriptorMap<var_ts...>(rc);
  auto q_ =
      descriptors[impl::desc_key_t{coarse, GetTopologicalType(el_)}].GetPack(rc.get());
  const int b = 0;
  const int lstart = q_.GetLowerBoundHost(b);
  const int lend = q_.GetUpperBoundHost(b);
  if (lend < lstart) return;
  auto nb = IndexRange{lstart, lend};

  MeshBlock *pmb = rc->GetBlockPointer();
  const auto &bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;

  const auto &range = X1 ? bounds.GetBoundsI(IndexDomain::interior, el_)
                         : (X2 ? bounds.GetBoundsJ(IndexDomain::interior, el_)
                               : bounds.GetBoundsK(IndexDomain::interior, el_));
  const int ref_ = INNER ? range.s : range.e;

  std::string label = (TYPE == BCType::Reflect ? "Reflect" : "Outflow");
  label += (INNER ? "Inner" : "Outer");
  label += "X" + std::to_string(DIR);

  constexpr IndexDomain domain =
      INNER ? (X1 ? IndexDomain::inner_x1
                  : (X2 ? IndexDomain::inner_x2 : IndexDomain::inner_x3))
            : (X1 ? IndexDomain::outer_x1
                  : (X2 ? IndexDomain::outer_x2 : IndexDomain::outer_x3));

  // used for reflections
  [[maybe_unused]] const int offset_ = 2 * ref_ + (INNER ? -1 : 1);

  // used for derivatives
  [[maybe_unused]] const int offsetin_ = INNER;
  [[maybe_unused]] const int offsetout_ = !INNER;
  pmb->par_for_bndry(
      PARTHENON_AUTO_LABEL, nb, domain, el_, coarse,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        // Shadow mnemonics inside kernel
        constexpr bool X1 = (DIR == X1DIR);
        constexpr bool X2 = (DIR == X2DIR);
        constexpr bool X3 = (DIR == X3DIR);
        // Redeclare variables to allow for lambda capture outside of constexpr if block
        auto &q = q_;
        auto &el = el_;
        [[maybe_unused]] const auto &offset = offset_;
        [[maybe_unused]] const auto &offsetin = offsetin_;
        [[maybe_unused]] const auto &offsetout = offsetout_;
        [[maybe_unused]] const auto &ref = ref_;
        [[maybe_unused]] const auto &val = val_;
        if constexpr (TYPE == BCType::Reflect) {
          const bool reflect = (q(b, el, l).vector_component == DIR);
          q(b, el, l, k, j, i) =
              (reflect ? -1.0 : 1.0) *
              q(b, el, l, X3 ? offset - k : k, X2 ? offset - j : j, X1 ? offset - i : i);
        } else if constexpr (TYPE == BCType::FixedFace) {
          q(b, el, l, k, j, i) = 2.0 * val - q(b, el, l, X3 ? offset - k : k,
                                               X2 ? offset - j : j, X1 ? offset - i : i);
        } else if constexpr (TYPE == BCType::ConstantDeriv) {
          Real dq = q(b, el, l, X3 ? ref + offsetin : k, X2 ? ref + offsetin : j,
                      X1 ? ref + offsetin : i) -
                    q(b, el, l, X3 ? ref - offsetout : k, X2 ? ref - offsetout : j,
                      X1 ? ref - offsetout : i);
          Real delta = 0.0;
          if constexpr (X1) {
            delta = i - ref;
          } else if constexpr (X2) {
            delta = j - ref;
          } else {
            delta = k - ref;
          }
          q(b, el, l, k, j, i) =
              q(b, el, l, X3 ? ref : k, X2 ? ref : j, X1 ? ref : i) + delta * dq;
        } else if constexpr (TYPE == BCType::Fixed) {
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
