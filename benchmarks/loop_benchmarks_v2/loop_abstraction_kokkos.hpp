#pragma once

#include "loop_abstraction_base.hpp"
#include "loop_abstraction_range.hpp"

namespace plb2 {

namespace loop_abstraction::impl {

template <class IndexSpaceType, class F>
void outer_kokkos(IndexSpaceType idx_space, F &&f) {
  using InnerIndexRangeType = InnerIndexRange<IndexSpaceType>;
  if constexpr (IndexSpaceType::loop_tag_v == loop_tag::boiv) {
    const int cells_per_block = static_cast<int>(idx_space.GetLogicalIndexer().size());
    const int total = idx_space.GetNBlocks() * cells_per_block;
    Kokkos::parallel_for(
        "loop_abstraction::outer_kokkos_boiv",
        Kokkos::RangePolicy<parthenon::DevExecSpace>(0, total),
        KOKKOS_LAMBDA(const int flat) {
          const int b = flat / cells_per_block;
          const int local = flat % cells_per_block;
          const auto [k, j, i] = idx_space.GetLogicalIndexer()(local);
          InnerIndexRangeType idx_range;
          idx_range.pidx_space = &idx_space;
          idx_range.block = b;
          idx_range.k = k;
          idx_range.j = j;
          idx_range.i = i;
          f(idx_range, b);
        });
  } else if constexpr (IndexSpaceType::loop_tag_v == loop_tag::bovi) {
    const int nouter = GetNOuter(idx_space);
    const int league_size = idx_space.GetNBlocks() * nouter;
    const Kokkos::TeamPolicy<parthenon::DevExecSpace> policy(league_size, Kokkos::AUTO);
    Kokkos::parallel_for(
        "loop_abstraction::outer_kokkos_team", policy,
        KOKKOS_LAMBDA(const device_team_member_t &member) {
          const int league = member.league_rank();
          const int b = league / nouter;
          const int o = league % nouter;
          const int logical_start = o * idx_space.GetNInner();
          const int logical_end = std::min((o + 1) * idx_space.GetNInner() - 1,
                                           static_cast<int>(idx_space.GetLogicalIndexer().size()) -
                                               1);
          const auto idx_range = FlatRange(idx_space, b, logical_start, logical_end, &member);
          f(idx_range, b);
        });
  } else if constexpr (IndexSpaceType::loop_tag_v == loop_tag::bvoi) {
    const Kokkos::TeamPolicy<parthenon::DevExecSpace> policy(idx_space.GetNBlocks(), Kokkos::AUTO);
    const auto &logical_kji = idx_space.GetLogicalIndexer();
    const int ks = logical_kji.template StartIdx<0>();
    const int js = logical_kji.template StartIdx<1>();
    const int is = logical_kji.template StartIdx<2>();
    Kokkos::parallel_for(
        "loop_abstraction::outer_kokkos_bvoi", policy,
        KOKKOS_LAMBDA(const device_team_member_t &member) {
          const int b = member.league_rank();
          InnerIndexRangeType idx_range;
          idx_range.pidx_space = &idx_space;
          idx_range.block = b;
          idx_range.ks = ks;
          idx_range.js = js;
          idx_range.is = is;
          idx_range.team_member = &member;
          f(idx_range, b);
        });
  }
}

template <class InnerIndexRangeType, class F>
KOKKOS_FORCEINLINE_FUNCTION void inner_kokkos(const InnerIndexRangeType &idx_range, F &&f) {
  using IndexSpaceType =
      std::remove_cv_t<std::remove_reference_t<decltype(*idx_range.pidx_space)>>;
  const auto &idx_space = *(idx_range.pidx_space);
  if constexpr (IndexSpaceType::loop_tag_v == loop_tag::boiv) {
    if constexpr (IndexSpaceType::inner_tag_v == inner_tag::logical_flat) {
      if constexpr (std::is_invocable_v<F, int, int, int>) {
        f(idx_range.k, idx_range.j, idx_range.i);
      } else {
        f(idx_space.GetLogicalIndexer().GetFlatIdx(idx_range.k, idx_range.j, idx_range.i));
      }
    } else if constexpr (IndexSpaceType::inner_tag_v == inner_tag::logical_coords) {
      if constexpr (std::is_invocable_v<F, int, int, int>) {
        f(idx_range.k, idx_range.j, idx_range.i);
      } else {
        f(Index3{idx_range.k, idx_range.j, idx_range.i});
      }
    }
  } else if constexpr (IndexSpaceType::loop_tag_v == loop_tag::bovi) {
    const auto *team_member = idx_range.team_member;
    KOKKOS_ASSERT(team_member != nullptr);
    const auto &member = *team_member;
    const int start = idx_range.flat_start;
    const int end_exclusive = idx_range.flat_end + 1 - start;
    const int mem_start = idx_space.GetMemoryIndexer().GetFlatIdx(idx_range.ks, idx_range.js, idx_range.is); 
    Kokkos::parallel_for(Kokkos::TeamThreadRange(member, 0, end_exclusive),
                         KOKKOS_LAMBDA(const int idx) {
                           if constexpr (std::is_invocable_v<F, int, int, int>) {
                             if constexpr (IndexSpaceType::inner_tag_v == inner_tag::memory) {
                               const auto [k, j, i] = idx_space.GetMemoryIndexer()(idx + start);
                               f(k, j, i);
                             } else {
                               const auto [k, j, i] = idx_space.GetLogicalIndexer()(idx + start);
                               f(k, j, i);
                             }
                           } else if constexpr (IndexSpaceType::inner_tag_v ==
                                                inner_tag::memory) {
                             f(idx);
                           } else if constexpr (IndexSpaceType::inner_tag_v ==
                                                inner_tag::logical_flat) {
                             const auto [k, j, i] = idx_space.GetLogicalIndexer()(idx + start);
                             f(idx_space.GetMemoryIndexer().GetFlatIdx(k, j, i) - mem_start);
                           } else {
                             const auto [k, j, i] = idx_space.GetLogicalIndexer()(idx + start);
                             f(Index3{k, j, i});
                           }
                         });
  } else if constexpr (IndexSpaceType::loop_tag_v == loop_tag::bvoi) {
    const auto &idx_space = *(idx_range.pidx_space);
    const auto *team_member = idx_range.team_member;
    KOKKOS_ASSERT(team_member != nullptr);
    const auto &member = *team_member;
    const int nouter = GetNOuter(idx_space);
    Kokkos::parallel_for(Kokkos::TeamThreadRange(member, 0, nouter),
                         KOKKOS_LAMBDA(const int o) {
                           const int logical_start = o * idx_space.GetNInner();
                           const int logical_end =
                               std::min((o + 1) * idx_space.GetNInner() - 1,
                                        static_cast<int>(idx_space.GetLogicalIndexer().size()) - 1);
                           if constexpr (IndexSpaceType::inner_tag_v == inner_tag::memory) {
                             const auto [ks, js, is] = idx_space.GetLogicalIndexer()(logical_start);
                             const auto [ke, je, ie] = idx_space.GetLogicalIndexer()(logical_end);
                             const int flat_start = idx_space.GetMemoryIndexer().GetFlatIdx(ks, js, is);
                             const int flat_end = idx_space.GetMemoryIndexer().GetFlatIdx(ke, je, ie);
                             Kokkos::parallel_for(Kokkos::TeamThreadRange(member, flat_start,
                                                                          flat_end + 1),
                                                  KOKKOS_LAMBDA(const int idx) {
                                                    if constexpr (std::is_invocable_v<F, int, int, int>) {
                                                      const auto [k, j, i] = idx_space.GetMemoryIndexer()(idx);
                                                      f(k, j, i);
                                                    } else {
                                                      f(idx);
                                                    }
                                                  });
                           } else if constexpr (IndexSpaceType::inner_tag_v == inner_tag::logical_flat) {
                             Kokkos::parallel_for(
                                 Kokkos::TeamThreadRange(member, logical_start, logical_end + 1),
                                 KOKKOS_LAMBDA(const int idx) {
                                   const auto [k, j, i] = idx_space.GetLogicalIndexer()(idx);
                                   if constexpr (std::is_invocable_v<F, int, int, int>) { 
                                     f(k, j, i);
                                   } else { 
                                     f(idx_space.GetMemoryIndexer().GetFlatIdx(k, j, i));
                                   }
                                 });
                           } else {
                             Kokkos::parallel_for(
                                 Kokkos::TeamThreadRange(member, logical_start, logical_end + 1),
                                 KOKKOS_LAMBDA(const int idx) {
                                   const auto [k, j, i] = idx_space.GetLogicalIndexer()(idx);
                                   if constexpr (std::is_invocable_v<F, int, int, int>) { 
                                     f(k, j, i);  
                                   } else { 
                                     f(Index3{k, j, i});
                                   }
                                 });
                           }
                         });
  }
}

} // namespace loop_abstraction::impl

} // namespace plb2
