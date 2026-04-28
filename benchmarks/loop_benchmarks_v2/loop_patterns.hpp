#pragma once

#include <algorithm>

#include <Kokkos_Core.hpp>

#include "dataset.hpp"
#include "kernels.hpp"

namespace plb2 {

namespace {

using TeamPolicy = Kokkos::TeamPolicy<>;
using TeamMember = TeamPolicy::member_type;

struct FlatSpan {
  int start = 0;
  int end = -1;
  int size = 0;
};

template <typename IndexerType>
KOKKOS_INLINE_FUNCTION FlatSpan MakeFlatSpan(const IndexerType &indexer, int outer,
                                             int ninner) {
  const int start = outer * ninner;
  const int end = std::min(static_cast<int>(indexer.size()) - 1, start + ninner - 1);
  return {start, end, end - start + 1};
}

KOKKOS_INLINE_FUNCTION int CeilDiv(int numer, int denom) {
  return (numer + denom - 1) / denom;
}

inline int SelectNvarsForBlock(const Dataset &dataset, int block) {
  const auto &spec = dataset.problem;
  if (block >= 0 && block < static_cast<int>(spec.vars_per_block.size())) {
    return std::min(spec.vars_per_block[block], spec.nvars);
  }
  return spec.nvars;
}

template <typename Body>
inline void RunCpuFlatGhosts(const Dataset &dataset, Body body) {
  const auto &spec = dataset.problem;
  const auto &data = dataset.data;
  const auto memory_indexer = spec.memory_indexer;
  const int nmem = static_cast<int>(memory_indexer.size());

  for (int b = 0; b < spec.nblocks; ++b) {
    const int nvars = SelectNvarsForBlock(dataset, b);
    for (int v = 0; v < nvars; ++v) {
#pragma omp simd
      for (int flat = 0; flat < nmem; ++flat) {
        const auto [k, j, i] = memory_indexer(flat);
        data.out(b, v, k, j, i) = body(data, b, v, k, j, i);
      }
    }
  }
}

// cpu_boiv_contiguous: direct-view form for block/outer/inner/var over memory-order spans.
template <typename Body>
inline void RunCpuBoivContiguous(const Dataset &dataset, int logical_inner_size, Body body) {
  const auto &spec = dataset.problem;
  const auto &data = dataset.data;
  const auto memory_indexer = spec.memory_indexer;
  const int outer_points = CeilDiv(static_cast<int>(memory_indexer.size()), logical_inner_size);

  for (int b = 0; b < spec.nblocks; ++b) {
    const int nvars = SelectNvarsForBlock(dataset, b);
    for (int outer = 0; outer < outer_points; ++outer) {
      const FlatSpan span = MakeFlatSpan(memory_indexer, outer, logical_inner_size);
      for (int idx = 0; idx < span.size; ++idx) {
        const auto [k, j, i] = memory_indexer(span.start + idx);
        for (int v = 0; v < nvars; ++v) {
          data.out(b, v, k, j, i) = body(data, b, v, k, j, i);
        }
      }
    }
  }
}

// cpu_bovi_contiguous: hoisted-pointer form for block/outer/var/inner over memory-order spans.
template <typename AccessBuilder, typename Body>
inline void RunCpuBoviContiguous(const Dataset &dataset, int logical_inner_size,
                                 AccessBuilder build_access, Body body) {
  const auto &spec = dataset.problem;
  const auto &data = dataset.data;
  const auto memory_indexer = spec.memory_indexer;
  const int outer_points = CeilDiv(static_cast<int>(memory_indexer.size()), logical_inner_size);

  for (int b = 0; b < spec.nblocks; ++b) {
    const int nvars = SelectNvarsForBlock(dataset, b);
    for (int outer = 0; outer < outer_points; ++outer) {
      const FlatSpan span = MakeFlatSpan(memory_indexer, outer, logical_inner_size);
      const auto [k, j, i] = memory_indexer(span.start);

      for (int v = 0; v < nvars; ++v) {
        const auto access = build_access(data, b, v, k, j, i);
        double *const out = &data.out(b, v, k, j, i);
#pragma omp simd
        for (int idx = 0; idx < span.size; ++idx) {
          out[idx] = body(access, idx);
        }
      }
    }
  }
}

// cpu_boiv_logical: direct-view form for block/outer/inner/var over active logical spans.
template <typename Body>
inline void RunCpuBoivLogical(const Dataset &dataset, int logical_inner_size, Body body) {
  const auto &spec = dataset.problem;
  const auto &data = dataset.data;
  const auto logical_indexer = spec.logical_indexer;
  const int outer_points = CeilDiv(static_cast<int>(logical_indexer.size()), logical_inner_size);

  for (int b = 0; b < spec.nblocks; ++b) {
    const int nvars = SelectNvarsForBlock(dataset, b);
    for (int outer = 0; outer < outer_points; ++outer) {
      const FlatSpan span = MakeFlatSpan(logical_indexer, outer, logical_inner_size);
      for (int idx = 0; idx < span.size; ++idx) {
        const auto [k, j, i] = logical_indexer(span.start + idx);
        for (int v = 0; v < nvars; ++v) {
          data.out(b, v, k, j, i) = body(data, b, v, k, j, i);
        }
      }
    }
  }
}

// cpu_bovi_logical: direct-view form for block/outer/var/inner over active logical spans.
template <typename Body>
inline void RunCpuBoviLogical(const Dataset &dataset, int logical_inner_size, Body body) {
  const auto &spec = dataset.problem;
  const auto &data = dataset.data;
  const auto logical_indexer = spec.logical_indexer;
  const int outer_points = CeilDiv(static_cast<int>(logical_indexer.size()), logical_inner_size);

  for (int b = 0; b < spec.nblocks; ++b) {
    const int nvars = SelectNvarsForBlock(dataset, b);
    for (int outer = 0; outer < outer_points; ++outer) {
      const FlatSpan span = MakeFlatSpan(logical_indexer, outer, logical_inner_size);

      for (int v = 0; v < nvars; ++v) {
#pragma omp simd
        for (int idx = 0; idx < span.size; ++idx) {
          const auto [k, j, i] = logical_indexer(span.start + idx);
          data.out(b, v, k, j, i) = body(data, b, v, k, j, i);
        }
      }
    }
  }
}

// cpu_bvoi_contiguous: hoisted-pointer form for block/var/outer/inner over memory-order spans.
template <typename AccessBuilder, typename Body>
inline void RunCpuBvoiContiguous(const Dataset &dataset, int logical_inner_size,
                                 AccessBuilder build_access, Body body) {
  const auto &spec = dataset.problem;
  const auto &data = dataset.data;
  const auto memory_indexer = spec.memory_indexer;
  const int outer_points = CeilDiv(static_cast<int>(memory_indexer.size()), logical_inner_size);

  for (int b = 0; b < spec.nblocks; ++b) {
    const int nvars = SelectNvarsForBlock(dataset, b);
    for (int v = 0; v < nvars; ++v) {
      for (int outer = 0; outer < outer_points; ++outer) {
        const FlatSpan span = MakeFlatSpan(memory_indexer, outer, logical_inner_size);
        const auto [k, j, i] = memory_indexer(span.start);
        const auto access = build_access(data, b, v, k, j, i);
        double *const out = &data.out(b, v, k, j, i);

#pragma omp simd
        for (int idx = 0; idx < span.size; ++idx) {
          out[idx] = body(access, idx);
        }
      }
    }
  }
}

// cpu_bvoi_logical: direct-view form for block/var/outer/inner over active logical spans.
template <typename Body>
inline void RunCpuBvoiLogical(const Dataset &dataset, int logical_inner_size, Body body) {
  const auto &spec = dataset.problem;
  const auto &data = dataset.data;
  const auto logical_indexer = spec.logical_indexer;
  const int outer_points = CeilDiv(static_cast<int>(logical_indexer.size()), logical_inner_size);

  for (int b = 0; b < spec.nblocks; ++b) {
    const int nvars = SelectNvarsForBlock(dataset, b);
    for (int v = 0; v < nvars; ++v) {
      for (int outer = 0; outer < outer_points; ++outer) {
        const FlatSpan span = MakeFlatSpan(logical_indexer, outer, logical_inner_size);

#pragma omp simd
        for (int idx = 0; idx < span.size; ++idx) {
          const auto [k, j, i] = logical_indexer(span.start + idx);
          data.out(b, v, k, j, i) = body(data, b, v, k, j, i);
        }
      }
    }
  }
}

// kokkos_boiv_flat: single RangePolicy launch over the active logical index space.
template <typename Body>
inline void RunKokkosBoivFlat(const Dataset &dataset, Body body) {
  const auto &spec = dataset.problem;
  const auto &data = dataset.data;
  const auto logical_indexer = spec.logical_indexer;
  const int cells_per_block = static_cast<int>(logical_indexer.size());
  const int total = spec.nblocks * cells_per_block;

  Kokkos::parallel_for(
      "KokkosBoivFlat", Kokkos::RangePolicy<>(0, total), KOKKOS_LAMBDA(const int flat) {
        const int b = flat / cells_per_block;
        const int local = flat % cells_per_block;
        const auto [k, j, i] = logical_indexer(local);
        const int nvars = data.active_counts(b);
        for (int v = 0; v < nvars; ++v) {
          data.out(b, v, k, j, i) = body(data, b, v, k, j, i);
        }
      });
}

// kokkos_bovi_team_contiguous: TeamPolicy launch with a hoisted contiguous span per outer chunk.
template <typename AccessBuilder, typename Body>
inline void RunKokkosBoviTeamContiguous(const Dataset &dataset, int logical_inner_size,
                                        AccessBuilder build_access, Body body) {
  const auto &spec = dataset.problem;
  const auto &data = dataset.data;
  const auto memory_indexer = spec.memory_indexer;
  const int cells_per_block = static_cast<int>(memory_indexer.size());
  const int outer_points = CeilDiv(cells_per_block, logical_inner_size);
  const int league_size = spec.nblocks * outer_points;
  const TeamPolicy policy(league_size, Kokkos::AUTO);

  Kokkos::parallel_for(
      "KokkosBoviTeamContiguous", policy, KOKKOS_LAMBDA(const TeamMember &member) {
        const int league = member.league_rank();
        const int b = league / outer_points;
        const int outer = league % outer_points;
        const FlatSpan span = MakeFlatSpan(memory_indexer, outer, logical_inner_size);
        const auto [k, j, i] = memory_indexer(span.start);
        const int nvars = data.active_counts(b);

        for (int v = 0; v < nvars; ++v) {
          const auto access = build_access(data, b, v, k, j, i);
          double *const out = &data.out(b, v, k, j, i);
          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(member, 0, span.size),
              KOKKOS_LAMBDA(const int idx) { out[idx] = body(access, idx); });
        }
      });
}

// kokkos_bovi_team_logical: TeamPolicy launch over logical active spans, direct-view inside.
template <typename Body>
inline void RunKokkosBoviTeamLogical(const Dataset &dataset, int logical_inner_size, Body body) {
  const auto &spec = dataset.problem;
  const auto &data = dataset.data;
  const auto logical_indexer = spec.logical_indexer;
  const int cells_per_block = static_cast<int>(logical_indexer.size());
  const int outer_points = CeilDiv(cells_per_block, logical_inner_size);
  const int league_size = spec.nblocks * outer_points;
  const TeamPolicy policy(league_size, Kokkos::AUTO);

  Kokkos::parallel_for(
      "KokkosBoviTeamLogical", policy, KOKKOS_LAMBDA(const TeamMember &member) {
        const int league = member.league_rank();
        const int b = league / outer_points;
        const int outer = league % outer_points;
        const FlatSpan span = MakeFlatSpan(logical_indexer, outer, logical_inner_size);
        const int nvars = data.active_counts(b);

        for (int v = 0; v < nvars; ++v) {
          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(member, 0, span.size),
              KOKKOS_LAMBDA(const int idx) {
                const auto [k, j, i] = logical_indexer(span.start + idx);
                data.out(b, v, k, j, i) = body(data, b, v, k, j, i);
              });
        }
      });
}

}  // namespace

}  // namespace plb2
