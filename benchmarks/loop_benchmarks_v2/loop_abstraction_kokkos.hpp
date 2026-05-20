#pragma once

#include "loop_abstraction_base.hpp"

namespace plb2 {

namespace loop_abstraction::impl {

template <class IndexSpaceType, class F>
void outer_kokkos(IndexSpaceType idx_space, F &&f) {
  using InnerIndexRangeType = InnerIndexRange<IndexSpaceType>;
  if constexpr (IndexSpaceType::loop_tag_v == loop_tag::boiv) {
    static_assert(IndexSpaceType::inner_tag_v == inner_tag::logical,
                  "boiv currently expects logical inner coordinates");
    const int cells_per_block = static_cast<int>(idx_space.logical_kji.size());
    const int total = idx_space.nblocks * cells_per_block;
    Kokkos::parallel_for(
        "loop_abstraction::outer_kokkos_boiv",
        Kokkos::RangePolicy<parthenon::DevExecSpace>(0, total),
        KOKKOS_LAMBDA(const int flat) {
          const int b = flat / cells_per_block;
          const int local = flat % cells_per_block;
          const auto [k, j, i] = idx_space.logical_kji(local);
          InnerIndexRangeType idx_range;
          idx_range.pidx_space = &idx_space;
          idx_range.block = b;
          idx_range.payload_.k = k;
          idx_range.payload_.j = j;
          idx_range.payload_.i = i;
          f(idx_range, b);
        });
  } else if constexpr (IndexSpaceType::loop_tag_v == loop_tag::bovi) {
    const int nouter = GetNOuter(idx_space);
    const int league_size = idx_space.nblocks * nouter;
    const Kokkos::TeamPolicy<parthenon::DevExecSpace> policy(league_size, Kokkos::AUTO);
    Kokkos::parallel_for(
        "loop_abstraction::outer_kokkos_team", policy,
        KOKKOS_LAMBDA(const device_team_member_t &member) {
          const int league = member.league_rank();
          const int b = league / nouter;
          const int o = league % nouter;
          const int logical_start = o * idx_space.ninner;
          const int logical_end = std::min((o + 1) * idx_space.ninner - 1,
                                           static_cast<int>(idx_space.logical_kji.size()) - 1);
          const auto idx_range =
              InnerIndexRangeType::FlatRange(idx_space, b, logical_start, logical_end, &member);
          f(idx_range, b);
        });
  } else if constexpr (IndexSpaceType::loop_tag_v == loop_tag::bvoi) {
    const Kokkos::TeamPolicy<parthenon::DevExecSpace> policy(idx_space.nblocks, Kokkos::AUTO);
    Kokkos::parallel_for(
        "loop_abstraction::outer_kokkos_bvoi", policy,
        KOKKOS_LAMBDA(const device_team_member_t &member) {
          const int b = member.league_rank();
          InnerIndexRangeType idx_range;
          idx_range.pidx_space = &idx_space;
          idx_range.block = b;
          idx_range.payload_.team_member = &member;
          f(idx_range, b);
        });
  }
}

template <class InnerIndexRangeType, class F>
KOKKOS_FORCEINLINE_FUNCTION void inner_kokkos(const InnerIndexRangeType &idx_range, F &&f) {
  using IndexSpaceType =
      std::remove_cv_t<std::remove_reference_t<decltype(*idx_range.pidx_space)>>;
  if constexpr (IndexSpaceType::loop_tag_v == loop_tag::boiv) {
    f(Index3{idx_range.payload_.k, idx_range.payload_.j, idx_range.payload_.i});
  } else if constexpr (IndexSpaceType::loop_tag_v == loop_tag::bovi) {
    const auto &idx_space = *(idx_range.pidx_space);
    const auto *team_member = idx_range.payload_.team_member;
    KOKKOS_ASSERT(team_member != nullptr);
    const auto &member = *team_member;
    const int start = idx_range.payload_.flat_start;
    const int end_exclusive = idx_range.payload_.flat_end + 1 - start;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(member, 0, end_exclusive),
                         KOKKOS_LAMBDA(const int idx) {
                           if constexpr (IndexSpaceType::inner_tag_v == inner_tag::memory) {
                             f(idx);
                           } else {
                             const auto [k, j, i] = idx_space.logical_kji(idx + start);
                             f(Index3{k, j, i});
                           }
                         });
  } else if constexpr (IndexSpaceType::loop_tag_v == loop_tag::bvoi) {
    const auto &idx_space = *(idx_range.pidx_space);
    const auto *team_member = idx_range.payload_.team_member;
    KOKKOS_ASSERT(team_member != nullptr);
    const auto &member = *team_member;
    const int nouter = GetNOuter(idx_space);
    Kokkos::parallel_for(Kokkos::TeamThreadRange(member, 0, nouter),
                         KOKKOS_LAMBDA(const int o) {
                           const int logical_start = o * idx_space.ninner;
                           const int logical_end =
                               std::min((o + 1) * idx_space.ninner - 1,
                                        static_cast<int>(idx_space.logical_kji.size()) - 1);
                           if constexpr (IndexSpaceType::inner_tag_v == inner_tag::memory) {
                             const auto [ks, js, is] = idx_space.logical_kji(logical_start);
                             const auto [ke, je, ie] = idx_space.logical_kji(logical_end);
                             const int flat_start = idx_space.memory_kji.GetFlatIdx(ks, js, is);
                             const int flat_end = idx_space.memory_kji.GetFlatIdx(ke, je, ie);
                             Kokkos::parallel_for(Kokkos::TeamThreadRange(member, flat_start,
                                                                          flat_end + 1),
                                                  KOKKOS_LAMBDA(const int idx) { f(idx); });
                           } else {
                             Kokkos::parallel_for(
                                 Kokkos::TeamThreadRange(member, logical_start, logical_end + 1),
                                 KOKKOS_LAMBDA(const int idx) {
                                   const auto [k, j, i] = idx_space.logical_kji(idx);
                                   f(idx_space.memory_kji.GetFlatIdx(k, j, i));
                                 });
                           }
                         });
  }
}

} // namespace loop_abstraction::impl

} // namespace plb2
