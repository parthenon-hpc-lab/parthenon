#pragma once

#include "loop_abstraction_base.hpp"

namespace plb2 {

namespace loop_abstraction {

template <class IndexSpaceType>
KOKKOS_FUNCTION InnerIndexRange<IndexSpaceType> FlatRange(
    const IndexSpaceType &idx_space, int b, int logical_start, int logical_end,
    const device_team_member_t *team_member = nullptr) {
  InnerIndexRange<IndexSpaceType> out;
  out.pidx_space = &idx_space;
  const auto [ks, js, is] = idx_space.GetLogicalIndexer()(logical_start);
  out.block = b;
  const auto [ke, je, ie] = idx_space.GetLogicalIndexer()(logical_end);
  if constexpr (IndexSpaceType::inner_tag_v == inner_tag::memory) {
    out.flat_start = idx_space.GetMemoryIndexer().GetFlatIdx(ks, js, is);
    out.flat_end = idx_space.GetMemoryIndexer().GetFlatIdx(ke, je, ie);
  } else if constexpr (IndexSpaceType::inner_tag_v == inner_tag::logical_flat ||
                       IndexSpaceType::inner_tag_v == inner_tag::logical_coords) {
    out.flat_start = logical_start;
    out.flat_end = logical_end;
  }
  if constexpr (IndexSpaceType::loop_tag_v != loop_tag::boiv) {
    out.team_member = team_member;
    out.ks = ks;
    out.js = js;
    out.is = is;
  } else {
    out.k = ks;
    out.j = js;
    out.i = is;
  }
  return out;
}

} // namespace loop_abstraction

} // namespace plb2
