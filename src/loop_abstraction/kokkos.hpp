//========================================================================================
// (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
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
#ifndef LOOP_ABSTRACTION_KOKKOS_HPP_
#define LOOP_ABSTRACTION_KOKKOS_HPP_

// This file was made in part with generative AI.
#include <algorithm>

#include "base.hpp"

namespace parthenon::loop_abstraction::impl {

// An extended __host__ __device__ lambda (nvcc) fixes its capture set before resolving
// any `if constexpr` in its body, so a variable first used only inside a constexpr-if
// branch is never captured. Calling ForceCapture with those variables odr-uses them
// unconditionally at the top of the lambda, forcing them into the capture set. Compiles
// to nothing.
template <class... Ts>
KOKKOS_FORCEINLINE_FUNCTION void ForceCapture(const Ts &...) {}

// The abstraction exposes three logical levels -- blocks, outer (kji) chunks, and the
// inner traversal. Here only two are mapped onto Kokkos parallelism: the league (over
// blocks, or blocks x chunks for bovi) and the team/vector inner loop. Where a raw
// `for` walks chunks inside a team (e.g. the bvoi/memory chunk loop in inner_kokkos),
// that middle level could instead become another level of Kokkos parallelism; it is a
// plain loop for now because we don't really expect to use this in production. If we do
// for some reason, we need to look at the performance implications.
template <class IndexSpaceType, class F>
void outer_kokkos(IndexSpaceType idx_space, F &&f) {
  using InnerIndexRangeType = InnerIndexRange<IndexSpaceType>;
  const std::size_t scratch_size_in_bytes = idx_space.GetPerTeamScratchSizeInBytes();
  if constexpr (IndexSpaceType::loop_tag_v == loop_tag::boiv) {
    const std::int64_t cells_per_block =
        static_cast<std::int64_t>(idx_space.GetLogicalIndexer().size());
    const std::int64_t total = idx_space.GetNBlocks() * cells_per_block;
    Kokkos::parallel_for(
        "loop_abstraction::outer_kokkos_boiv",
        Kokkos::RangePolicy<parthenon::DevExecSpace>(0, total),
        KOKKOS_LAMBDA(const std::int64_t flat) {
          const int b = static_cast<int>(flat / cells_per_block);
          const int local = static_cast<int>(flat % cells_per_block);
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
      policy.set_scratch_size(1, Kokkos::PerTeam(scratch_size_in_bytes),
                              Kokkos::PerThread(0));
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
          InnerIndexRangeType idx_range(idx_space, idx_space.GetLogicalIndexer(), b,
                                        logical_start, logical_end, &member);
          f(idx_range, b);
        });
  } else if constexpr (IndexSpaceType::loop_tag_v == loop_tag::bvoi) {
    auto policy =
        Kokkos::TeamPolicy<parthenon::DevExecSpace>(idx_space.GetNBlocks(), Kokkos::AUTO);
    if (scratch_size_in_bytes > 0)
      policy.set_scratch_size(1, Kokkos::PerTeam(scratch_size_in_bytes),
                              Kokkos::PerThread(0));
    Kokkos::parallel_for(
        "loop_abstraction::outer_kokkos_bvoi", policy,
        KOKKOS_LAMBDA(const device_team_member_t &member) {
          const int b = member.league_rank();
          InnerIndexRangeType idx_range(idx_space, idx_space.GetLogicalIndexer(), b,
                                        &member);
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
    // In a reduced-dimension run, visit only the [begin, end) run of halo offsets that
    // do not point into a degenerate direction (see HaloReducedRange).
    const HaloRange hrange = HaloReducedRange<halo_t>(idx_space.GetNdim());
    if constexpr (IndexSpaceType::inner_tag_v == inner_tag::logical_flat) {
      if constexpr (std::is_invocable_v<F, int, int, int>) {
        for (int n = hrange.begin; n < hrange.end; ++n)
          f(idx_range.ks + halo_t::dk(n), idx_range.js + halo_t::dj(n),
            idx_range.is + halo_t::di(n));
      } else {
        static_assert(!impl::has_explicit_unary_int_call_v<F>,
                      "boiv/logical_flat inner loops require auto or MemoryOffset "
                      "single-argument bodies; explicit int bodies lose halo "
                      "offset coordinates.");
        for (int n = hrange.begin; n < hrange.end; ++n) {
          f(idx_space.GetMemoryOffsetIndex(halo_t::dk(n), halo_t::dj(n), halo_t::di(n)));
        }
      }
    } else if constexpr (IndexSpaceType::inner_tag_v == inner_tag::logical_coords) {
      if constexpr (std::is_invocable_v<F, int, int, int>) {
        for (int n = hrange.begin; n < hrange.end; ++n)
          f(idx_range.ks + halo_t::dk(n), idx_range.js + halo_t::dj(n),
            idx_range.is + halo_t::di(n));
      } else {
        for (int n = hrange.begin; n < hrange.end; ++n)
          f(Index3{idx_range.ks + halo_t::dk(n), idx_range.js + halo_t::dj(n),
                   idx_range.is + halo_t::di(n)});
      }
    }
  } else if constexpr (IndexSpaceType::loop_tag_v == loop_tag::bovi) {
    const auto *team_member = idx_range.team_member;
    PARTHENON_DEBUG_REQUIRE(team_member != nullptr,
                            "Should not be here with a nullptr team member.");
    const auto &member = *team_member;
    const auto &logical_kji = idx_range.logical_kji;
    const int mem_start =
        idx_space.GetMemoryIndexer().GetFlatIdx(idx_range.ks, idx_range.js, idx_range.is);
    for (int r = 0; r < idx_range.nregions; ++r) {
      const int start = idx_range.flat_start[r];
      const int end_exclusive = idx_range.flat_end[r] + 1 - start;
      Kokkos::parallel_for(
          Kokkos::TeamThreadRange(member, 0, end_exclusive), [&](const int idx) {
            ForceCapture(f, start, logical_kji, idx_space, mem_start);
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
  } else if constexpr (IndexSpaceType::loop_tag_v == loop_tag::bvoi) {
    const auto &idx_space = *(idx_range.pidx_space);
    const auto *team_member = idx_range.team_member;
    PARTHENON_DEBUG_REQUIRE(team_member != nullptr,
                            "Should not be here with a nullptr team member.");
    const auto &member = *team_member;
    const auto &logical_kji = idx_range.logical_kji;
    const int mem_start =
        idx_space.GetMemoryIndexer().GetFlatIdx(idx_range.ks, idx_range.js, idx_range.is);
    if constexpr (IndexSpaceType::inner_tag_v == inner_tag::memory) {
      // Chunk the halo-extended logical space directly (see the raw backend's
      // bvoi/memory branch for the full rationale): chunking the base space and
      // re-applying the halo per chunk double-counts chunk-boundary cells when the
      // halo aligns with the chunk-iteration direction. Each chunk of the extended
      // space maps to one contiguous memory span, and consecutive spans are strictly
      // increasing, so every cell is visited exactly once. ninner still bounds the
      // chunk size.
      const auto &ext_logical_kji = idx_range.logical_kji;
      // Resolve the chunk shape against the *extended* indexer (see raw backend).
      const int ninner = idx_space.GetNInner(ext_logical_kji);
      const int ext_size = static_cast<int>(ext_logical_kji.size());
      const int nouter = ext_size / ninner + (ext_size % ninner != 0);
      for (int o = 0; o < nouter; ++o) {
        const int logical_start = o * ninner;
        const int logical_end = std::min((o + 1) * ninner - 1, ext_size - 1);
        const auto [ks, js, is] = ext_logical_kji(logical_start);
        const auto [ke, je, ie] = ext_logical_kji(logical_end);
        const int mem_first = idx_space.GetMemoryIndexer().GetFlatIdx(ks, js, is);
        const int mem_last = idx_space.GetMemoryIndexer().GetFlatIdx(ke, je, ie);
        const int end_exclusive = mem_last + 1 - mem_first;
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(member, 0, end_exclusive), [&](const int idx) {
              ForceCapture(f, mem_first, idx_space, mem_start);
              if constexpr (std::is_invocable_v<F, int, int, int>) {
                const auto [k, j, i] = idx_space.GetMemoryIndexer()(idx + mem_first);
                f(k, j, i);
              } else {
                f(idx + mem_first - mem_start);
              }
            });
      }
    } else {
      for (int r = 0; r < idx_range.nregions; ++r) {
        const int start = idx_range.flat_start[r];
        const int end_exclusive = idx_range.flat_end[r] + 1 - start;
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(member, 0, end_exclusive), [&](const int idx) {
              ForceCapture(f, start, logical_kji, idx_space, mem_start);
              if constexpr (std::is_invocable_v<F, int, int, int>) {
                const auto [k, j, i] = logical_kji(idx + start);
                f(k, j, i);
              } else if constexpr ( // NOLINT(readability/braces)
                  IndexSpaceType::inner_tag_v == inner_tag::logical_flat) {
                const auto [k, j, i] = logical_kji(idx + start);
                f(idx_space.GetMemoryIndexer().GetFlatIdx(k, j, i) - mem_start);
              } else {
                const auto [k, j, i] = logical_kji(idx + start);
                f(Index3{k, j, i});
              }
            });
      }
    }
  }
}

// ---------------------------------------------------------------------------------------
// Reductions
//
// outer_reduce/inner_reduce mirror outer/inner but fold a single Kokkos reducer over a
// reduction index space (one carrying a reducer type; see ReductionIndexSpace). The
// reducer instance (e.g. Kokkos::Min<double>(result)) is bound to a host result and
// passed to outer_reduce; its type must match the space's reduction_t. The enclosing
// parallel_reduce accumulator is stored on the InnerIndexRange (reduce_.update); the
// reducer *type* comes from the index space, so inner_reduce needs no handle and does not
// restate the join op. Multiple inner_reduce calls (interleaved with plain inner calls
// that only fill scratch) all contribute to the same result. Reductions never touch
// ghost/halo cells: halo ranges are rejected at compile time and the memory inner tag
// degenerates to logical_flat so no ghost cell swept by a contiguous span is ever visited.
// ---------------------------------------------------------------------------------------

template <class IndexSpaceType, class F, class Reducer>
void outer_kokkos_reduce(IndexSpaceType idx_space, F &&f, Reducer reducer) {
  using InnerIndexRangeType = InnerIndexRange<IndexSpaceType>;
  using value_type = typename Reducer::value_type;
  static_assert(std::is_same_v<Reducer, typename IndexSpaceType::reduction_t>,
                "Reducer type must match the reduction index space's reduction_t.");
  const std::size_t scratch_size_in_bytes = idx_space.GetPerTeamScratchSizeInBytes();
  if constexpr (IndexSpaceType::loop_tag_v == loop_tag::boiv) {
    const std::int64_t cells_per_block =
        static_cast<std::int64_t>(idx_space.GetLogicalIndexer().size());
    const std::int64_t total = idx_space.GetNBlocks() * cells_per_block;
    Kokkos::parallel_reduce(
        "loop_abstraction::outer_kokkos_reduce_boiv",
        Kokkos::RangePolicy<parthenon::DevExecSpace>(0, total),
        KOKKOS_LAMBDA(const std::int64_t flat, value_type &update) {
          const int b = static_cast<int>(flat / cells_per_block);
          const int local = static_cast<int>(flat % cells_per_block);
          const auto [k, j, i] = idx_space.GetLogicalIndexer()(local);
          InnerIndexRangeType idx_range;
          idx_range.pidx_space = &idx_space;
          idx_range.block = b;
          idx_range.ks = k;
          idx_range.js = j;
          idx_range.is = i;
          idx_range.reduce_.update = &update;
          f(idx_range, b);
        },
        reducer);
  } else if constexpr (IndexSpaceType::loop_tag_v == loop_tag::bovi) {
    const int nouter = GetNOuter(idx_space);
    const int league_size = idx_space.GetNBlocks() * nouter;
    auto policy = Kokkos::TeamPolicy<parthenon::DevExecSpace>(league_size, Kokkos::AUTO);
    if (scratch_size_in_bytes > 0)
      policy.set_scratch_size(1, Kokkos::PerTeam(scratch_size_in_bytes),
                              Kokkos::PerThread(0));
    Kokkos::parallel_reduce(
        "loop_abstraction::outer_kokkos_reduce_team", policy,
        KOKKOS_LAMBDA(const device_team_member_t &member, value_type &update) {
          const int league = member.league_rank();
          const int b = league / nouter;
          const int o = league % nouter;
          const int logical_start = o * idx_space.GetNInner();
          const int logical_end =
              std::min((o + 1) * idx_space.GetNInner() - 1,
                       static_cast<int>(idx_space.GetLogicalIndexer().size()) - 1);
          InnerIndexRangeType idx_range(idx_space, idx_space.GetLogicalIndexer(), b,
                                        logical_start, logical_end, &member);
          idx_range.reduce_.update = &update;
          f(idx_range, b);
        },
        reducer);
  } else if constexpr (IndexSpaceType::loop_tag_v == loop_tag::bvoi) {
    auto policy =
        Kokkos::TeamPolicy<parthenon::DevExecSpace>(idx_space.GetNBlocks(), Kokkos::AUTO);
    if (scratch_size_in_bytes > 0)
      policy.set_scratch_size(1, Kokkos::PerTeam(scratch_size_in_bytes),
                              Kokkos::PerThread(0));
    Kokkos::parallel_reduce(
        "loop_abstraction::outer_kokkos_reduce_bvoi", policy,
        KOKKOS_LAMBDA(const device_team_member_t &member, value_type &update) {
          const int b = member.league_rank();
          InnerIndexRangeType idx_range(idx_space, idx_space.GetLogicalIndexer(), b,
                                        &member);
          idx_range.reduce_.update = &update;
          f(idx_range, b);
        },
        reducer);
  }
}

template <class InnerIndexRangeType, class F>
KOKKOS_FORCEINLINE_FUNCTION void
inner_kokkos_reduce(const InnerIndexRangeType &idx_range, F &&f) {
  using IndexSpaceType =
      std::remove_cv_t<std::remove_reference_t<decltype(*idx_range.pidx_space)>>;
  static_assert(IndexSpaceType::is_reduction_v,
                "inner_reduce requires a reduction index space (see ReductionIndexSpace "
                "/ IndexSpace::WithReducer).");
  using reducer_t = typename IndexSpaceType::reduction_t;
  using value_type = typename reducer_t::value_type;
  static_assert(
      std::is_same_v<typename InnerIndexRangeType::halo_t, halo::none_t>,
      "Reductions over halo ranges are not allowed: extend the range only for "
      "producer (scratch) inner loops, and reduce over the base (halo-free) range.");
  const auto &idx_space = *(idx_range.pidx_space);
  value_type &update = *idx_range.reduce_.update;
  if constexpr (IndexSpaceType::loop_tag_v == loop_tag::boiv) {
    // Single logical point, no team. halo is none_t (asserted above), so the identity
    // offset {0,0,0} is the only cell -- reduce it straight into the work-item's update.
    if constexpr (std::is_invocable_v<F, int, int, int, value_type &>) {
      f(idx_range.ks, idx_range.js, idx_range.is, update);
    } else if constexpr (IndexSpaceType::inner_tag_v == inner_tag::logical_coords) {
      f(Index3{idx_range.ks, idx_range.js, idx_range.is}, update);
    } else {
      f(idx_space.GetMemoryOffsetIndex(0, 0, 0), update);
    }
  } else {
    // bvoi / bovi team case. Unlike inner_kokkos -- whose parallel_for path needs
    // separate bvoi/bovi handling (a raw in-team loop over nregions, plus a memory-tag
    // chunk loop to bound swept ghosts) -- the reduce path merges the two. It can,
    // *only because reductions forbid halos*: with halo == none_t there is exactly one
    // region (nregions == 1) and no ghost cells to bound. What remains for both tags is
    // a single TeamThreadRange reduction over one contiguous logical span. The span
    // differs by tag purely through chunk_logical_start/end: bvoi's range spans the
    // whole block (league = blocks) while bovi's spans one chunk (league =
    // blocks x chunks); Kokkos then combines each league entry's update via the reducer.
    // Do NOT add halo support here without restoring the multi-region logic.
    //
    // Reduce this chunk's logical cells with a nested TeamThreadRange reduction (same
    // idiom as par_reduce_inner in kokkos_abstraction.hpp), then join the team result
    // into the enclosing accumulator once. We iterate the logical span for every inner
    // tag, so the memory tag (degenerated to logical_flat) never sweeps ghost cells.
    const auto &member = *idx_range.team_member;
    const auto &logical_kji = idx_range.logical_kji;
    const int start = idx_range.chunk_logical_start;
    const int n = idx_range.chunk_logical_end - start + 1;
    const int mem_start =
        idx_space.GetMemoryIndexer().GetFlatIdx(idx_range.ks, idx_range.js, idx_range.is);
    // A fresh reducer of the same type bound to a team-local result. Kokkos reducers are
    // cheap value types (a reference to the target + an optional space tag); join(a, b)
    // takes explicit refs and does not read the bound target, so any instance can perform
    // the final join into the enclosing accumulator.
    value_type team_result;
    reducer_t team_reducer(team_result);
    team_reducer.init(team_result);
    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(member, 0, n),
        [&](const int idx, value_type &partial) {
          ForceCapture(f, start, logical_kji, idx_space, mem_start);
          const auto [k, j, i] = logical_kji(idx + start);
          if constexpr (std::is_invocable_v<F, int, int, int, value_type &>) {
            f(k, j, i, partial);
          } else if constexpr (IndexSpaceType::inner_tag_v == inner_tag::logical_coords) {
            f(Index3{k, j, i}, partial);
          } else {
            // logical_flat and (degenerated) memory: memory-relative flat index.
            f(idx_space.GetMemoryIndexer().GetFlatIdx(k, j, i) - mem_start, partial);
          }
        },
        team_reducer);
    Kokkos::single(Kokkos::PerTeam(member),
                   [&]() { team_reducer.join(update, team_result); });
  }
}

} // namespace parthenon::loop_abstraction::impl

#endif // LOOP_ABSTRACTION_KOKKOS_HPP_
