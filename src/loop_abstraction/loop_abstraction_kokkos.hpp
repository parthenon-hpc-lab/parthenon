#pragma once

#include "loop_abstraction_base.hpp"

namespace loop_abstraction::impl {

template <class IndexSpaceType, class F>
void outer_kokkos(IndexSpaceType idx_space, F &&f) {
  using InnerIndexRangeType = InnerIndexRange<IndexSpaceType>;
  const std::size_t scratch_size_in_bytes = idx_space.GetPerTeamScratchSizeInBytes();
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
          idx_range.ks = k;
          idx_range.js = j;
          idx_range.is = i;
          f(idx_range, b);
        });
  } else if constexpr (IndexSpaceType::loop_tag_v == loop_tag::bovi) {
    const int nouter = GetNOuter(idx_space);
    const int league_size = idx_space.GetNBlocks() * nouter;
    auto policy = Kokkos::TeamPolicy<parthenon::DevExecSpace>(league_size, Kokkos::AUTO);
    if (scratch_size_in_bytes > 0)
      policy.set_scratch_size(1, Kokkos::PerTeam(scratch_size_in_bytes), Kokkos::PerThread(0));
    Kokkos::parallel_for(
        "loop_abstraction::outer_kokkos_team", policy,
        KOKKOS_LAMBDA(const device_team_member_t &member) {
          const int league = member.league_rank();
          const int b = league / nouter;
          const int o = league % nouter;
          const int logical_start = o * idx_space.GetNInner();
          const int logical_end =
              std::min((o + 1) * idx_space.GetNInner() - 1,
                       static_cast<int>(idx_space.GetLogicalIndexer().size()) - 1);
          InnerIndexRangeType idx_range(idx_space, idx_space.GetLogicalIndexer(), b, logical_start, logical_end, &member);
          f(idx_range, b);
        });
  } else if constexpr (IndexSpaceType::loop_tag_v == loop_tag::bvoi) {
    auto policy =
        Kokkos::TeamPolicy<parthenon::DevExecSpace>(idx_space.GetNBlocks(), Kokkos::AUTO);
    if (scratch_size_in_bytes > 0)
      policy.set_scratch_size(1, Kokkos::PerTeam(scratch_size_in_bytes), Kokkos::PerThread(0));
    Kokkos::parallel_for(
        "loop_abstraction::outer_kokkos_bvoi", policy,
        KOKKOS_LAMBDA(const device_team_member_t &member) {
          const int b = member.league_rank();
          InnerIndexRangeType idx_range(idx_space, idx_space.GetLogicalIndexer(), b, &member);
          f(idx_range, b);
        });
  }
}

template <class InnerIndexRangeType, class F>
KOKKOS_FORCEINLINE_FUNCTION void inner_kokkos(const InnerIndexRangeType &idx_range,
                                              F &&f) {
  using IndexSpaceType =
      std::remove_cv_t<std::remove_reference_t<decltype(*idx_range.pidx_space)>>;
  const auto &idx_space = *(idx_range.pidx_space);
  if constexpr (IndexSpaceType::loop_tag_v == loop_tag::boiv) {
    using halo_t = typename InnerIndexRangeType::halo_t;
    if constexpr (IndexSpaceType::inner_tag_v == inner_tag::logical_flat) {
      if constexpr (std::is_invocable_v<F, int, int, int>) {
        for (int n = 0; n < halo_t::npoints; ++n)
          f(idx_range.ks + halo_t::dk(n), idx_range.js + halo_t::dj(n),
            idx_range.is + halo_t::di(n));
      } else {
        static_assert(!impl::has_explicit_unary_int_call_v<F>,
                      "boiv/logical_flat inner loops require auto or MemoryOffset "
                      "single-argument bodies; explicit int bodies lose halo "
                      "offset coordinates.");
        for (int n = 0; n < halo_t::npoints; ++n) {
          f(idx_space.GetMemoryOffsetIndex(halo_t::dk(n), halo_t::dj(n),
                                           halo_t::di(n)));
        }
      }
    } else if constexpr (IndexSpaceType::inner_tag_v == inner_tag::logical_coords) {
      if constexpr (std::is_invocable_v<F, int, int, int>) {
        for (int n = 0; n < halo_t::npoints; ++n)
          f(idx_range.ks + halo_t::dk(n), idx_range.js + halo_t::dj(n),
            idx_range.is + halo_t::di(n));
      } else {
        for (int n = 0; n < halo_t::npoints; ++n)
          f(Index3{idx_range.ks + halo_t::dk(n), idx_range.js + halo_t::dj(n),
                   idx_range.is + halo_t::di(n)});
      }
    }
  } else if constexpr (IndexSpaceType::loop_tag_v == loop_tag::bovi) {
    const auto *team_member = idx_range.team_member;
    KOKKOS_ASSERT(team_member != nullptr);
    const auto &member = *team_member;
    const int mem_start =
        idx_space.GetMemoryIndexer().GetFlatIdx(idx_range.ks, idx_range.js, idx_range.is);
    for (int r = 0; r < idx_range.nregions; ++r) {
      const int start = idx_range.flat_start[r];
      const int end_exclusive = idx_range.flat_end[r] + 1 - start;
      Kokkos::parallel_for(
          Kokkos::TeamThreadRange(member, 0, end_exclusive),
          KOKKOS_LAMBDA(const int idx) {
            if constexpr (std::is_invocable_v<F, int, int, int>) {
              if constexpr (IndexSpaceType::inner_tag_v == inner_tag::memory) {
                const auto [k, j, i] = idx_space.GetMemoryIndexer()(idx + start);
                f(k, j, i);
              } else {
                const auto [k, j, i] = idx_space.GetLogicalIndexer()(idx + start);
                f(k, j, i);
              }
            } else if constexpr (IndexSpaceType::inner_tag_v == inner_tag::memory) {
              f(idx + start - mem_start);
            } else if constexpr (IndexSpaceType::inner_tag_v == inner_tag::logical_flat) {
              const auto [k, j, i] = idx_space.GetLogicalIndexer()(idx + start);
              f(idx_space.GetMemoryIndexer().GetFlatIdx(k, j, i) - mem_start);
            } else {
              const auto [k, j, i] = idx_space.GetLogicalIndexer()(idx + start);
              f(Index3{k, j, i});
            }
          });
    }
  } else if constexpr (IndexSpaceType::loop_tag_v == loop_tag::bvoi) {
    const auto &idx_space = *(idx_range.pidx_space);
    const auto *team_member = idx_range.team_member;
    KOKKOS_ASSERT(team_member != nullptr);
    const auto &member = *team_member;
    const int nouter = GetNOuter(idx_space);
    const auto &logical_kji = idx_range.logical_kji;
    const int mem_start =
        idx_space.GetMemoryIndexer().GetFlatIdx(idx_range.ks, idx_range.js, idx_range.is);
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(member, 0, nouter), KOKKOS_LAMBDA(const int o) {
          const int logical_start = o * idx_space.GetNInner();
          const int logical_end =
              std::min((o + 1) * idx_space.GetNInner() - 1,
                       static_cast<int>(idx_space.GetLogicalIndexer().size()) - 1);
          const InnerIndexRangeType inner_range(idx_space, idx_range.logical_kji,
                                                idx_range.block, logical_start,
                                                logical_end, team_member);
          for (int r = 0; r < inner_range.nregions; ++r) {
            const int start = inner_range.flat_start[r];
            const int end_exclusive = inner_range.flat_end[r] + 1 - start;
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(member, 0, end_exclusive),
                KOKKOS_LAMBDA(const int idx) {
                    if constexpr (std::is_invocable_v<F, int, int, int>) {
                      if constexpr (IndexSpaceType::inner_tag_v == inner_tag::memory) {
                        const auto [k, j, i] = idx_space.GetMemoryIndexer()(idx + start);
                        f(k, j, i);
                      } else {
                        const auto [k, j, i] = logical_kji(idx + start);
                        f(k, j, i);
                      }
                    } else if constexpr (IndexSpaceType::inner_tag_v == inner_tag::memory) {
                      f(idx + start - mem_start);
                    } else if constexpr (IndexSpaceType::inner_tag_v == inner_tag::logical_flat) {
                      const auto [k, j, i] = logical_kji(idx + start);
                      f(idx_space.GetMemoryIndexer().GetFlatIdx(k, j, i) - mem_start);
                    } else {
                      const auto [k, j, i] = logical_kji(idx + start);
                      f(Index3{k, j, i});
                    }
                });
          }
        });
  }
}

} // namespace loop_abstraction::impl
