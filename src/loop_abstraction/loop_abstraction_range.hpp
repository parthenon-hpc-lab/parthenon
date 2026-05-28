#pragma once

#include "loop_abstraction_base.hpp"

namespace loop_abstraction {

template <class Halo>
inline auto AddHaloToIndexer(const parthenon::Indexer3D &idxer) { 
  std::array<int, 3> extend_low{0, 0, 0}, extend_up{0, 0, 0};
  for (int p = 0; p < Halo::npoints; ++p) { 
    extend_low[0] = std::max(extend_low[0], -Halo::dk(p)); 
    extend_low[1] = std::max(extend_low[1], -Halo::dj(p)); 
    extend_low[2] = std::max(extend_low[2], -Halo::di(p)); 

    extend_up[0] = std::max(extend_up[0], Halo::dk(p)); 
    extend_up[1] = std::max(extend_up[1], Halo::dj(p)); 
    extend_up[2] = std::max(extend_up[2], Halo::di(p)); 
  }

  return parthenon::Indexer3D({idxer.template StartIdx<0>() - extend_low[0], idxer.template EndIdx<0>() + extend_up[0]},
                   {idxer.template StartIdx<1>() - extend_low[1], idxer.template EndIdx<1>() + extend_up[1]},
                   {idxer.template StartIdx<2>() - extend_low[2], idxer.template EndIdx<2>() + extend_up[2]});
} 

template <class IndexSpaceType>
KOKKOS_INLINE_FUNCTION InnerIndexRange<IndexSpaceType>
FlatRange(const IndexSpaceType &idx_space, int b, int logical_start, int logical_end,
          const device_team_member_t *team_member = nullptr) {
  const auto [ks, js, is] = idx_space.GetLogicalIndexer()(logical_start);
  const auto [ke, je, ie] = idx_space.GetLogicalIndexer()(logical_end);
  return FlatRange(idx_space, idx_space.GetLogicalIndexer(), b, ks, js, is, ke, je, ie, team_member);
}

template <class IndexSpaceType>
KOKKOS_INLINE_FUNCTION InnerIndexRange<IndexSpaceType>
FlatRange(const IndexSpaceType &idx_space, const parthenon::Indexer3D &logical_kji,
          int b, int ks, int js, int is, int ke, int je, int ie, 
          const device_team_member_t *team_member = nullptr) {
  InnerIndexRange<IndexSpaceType> out;
  out.pidx_space = &idx_space;
  out.logical_kji = logical_kji;
  out.block = b;
  auto &idxer = IndexSpaceType::inner_tag_v == inner_tag::memory ? idx_space.GetMemoryIndexer() : idx_space.GetLogicalIndexer(); 
  out.flat_start[0] = idxer.GetFlatIdx(ks, js, is);
  out.flat_end[0] = idxer.GetFlatIdx(ke, je, ie);
  out.ks = ks;
  out.js = js;
  out.is = is;
  out.team_member = team_member;
  return out;
}

} // namespace loop_abstraction
