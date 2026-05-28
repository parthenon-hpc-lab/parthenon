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

struct ChunkSpan {
  int logical_start = 0;
  int logical_end = -1;
  int memory_start = 0;
  int memory_end = -1;
  int size = 0;
};

template <typename LogicalIndexer, typename MemoryIndexer>
KOKKOS_INLINE_FUNCTION ChunkSpan MakeChunkSpan(const LogicalIndexer &logical_indexer,
                                               const MemoryIndexer &memory_indexer,
                                               int outer, int logical_inner_size) {
  const int logical_start = outer * logical_inner_size;
  const int logical_end = std::min(static_cast<int>(logical_indexer.size()) - 1,
                                   logical_start + logical_inner_size - 1);
  const auto [ks, js, is] = logical_indexer(logical_start);
  const auto [ke, je, ie] = logical_indexer(logical_end);
  const int memory_start = static_cast<int>(memory_indexer.GetFlatIdx(ks, js, is));
  const int memory_end = static_cast<int>(memory_indexer.GetFlatIdx(ke, je, ie));
  return {logical_start, logical_end, memory_start, memory_end,
          memory_end - memory_start + 1};
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

// cpu_flat_ghosts: hoisted-pointer form for the full memory-order span.
template <typename AccessBuilder, typename Body>
inline void RunCpuFlatGhosts(const Dataset &dataset, AccessBuilder build_access,
                             Body body) {
  const auto &spec = dataset.problem;
  const auto &data = dataset.data;
  const auto memory_indexer = spec.memory_indexer;
  const int nmem = static_cast<int>(memory_indexer.size());

  for (int b = 0; b < spec.nblocks; ++b) {
    const int nvars = SelectNvarsForBlock(dataset, b);
    for (int v = 0; v < nvars; ++v) {
      const auto [k, j, i] = memory_indexer(0);
      const auto access = build_access(data, b, v, k, j, i);
      double *const out = &data.out(b, v, k, j, i);
#pragma omp simd
      for (int flat = 0; flat < nmem; ++flat) {
        out[flat] = body(access, flat);
      }
    }
  }
}

// cpu_boiv_contiguous: direct-view form for block/outer/inner/var over memory-order
// spans.
template <typename Body>
inline void RunCpuBoivContiguous(const Dataset &dataset, int logical_inner_size,
                                 Body body) {
  const auto &spec = dataset.problem;
  const auto &data = dataset.data;
  const auto logical_indexer = spec.logical_indexer;
  const auto memory_indexer = spec.memory_indexer;
  const int outer_points =
      CeilDiv(static_cast<int>(logical_indexer.size()), logical_inner_size);

  for (int b = 0; b < spec.nblocks; ++b) {
    const int nvars = SelectNvarsForBlock(dataset, b);
    for (int outer = 0; outer < outer_points; ++outer) {
      const ChunkSpan span =
          MakeChunkSpan(logical_indexer, memory_indexer, outer, logical_inner_size);
      for (int idx = 0; idx < span.size; ++idx) {
        const auto [k, j, i] = memory_indexer(span.memory_start + idx);
        for (int v = 0; v < nvars; ++v) {
          data.out(b, v, k, j, i) = body(data, b, v, k, j, i);
        }
      }
    }
  }
}

// cpu_bovi_contiguous: hoisted-pointer form for block/outer/var/inner over memory-order
// spans.
template <typename AccessBuilder, typename Body>
inline void RunCpuBoviContiguous(const Dataset &dataset, int logical_inner_size,
                                 AccessBuilder build_access, Body body) {
  const auto &spec = dataset.problem;
  const auto &data = dataset.data;
  const auto logical_indexer = spec.logical_indexer;
  const auto memory_indexer = spec.memory_indexer;
  const int outer_points =
      CeilDiv(static_cast<int>(logical_indexer.size()), logical_inner_size);

  for (int b = 0; b < spec.nblocks; ++b) {
    const int nvars = SelectNvarsForBlock(dataset, b);
    for (int outer = 0; outer < outer_points; ++outer) {
      const ChunkSpan span =
          MakeChunkSpan(logical_indexer, memory_indexer, outer, logical_inner_size);
      const auto [k, j, i] = logical_indexer(span.logical_start);

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

// cpu_bovi_contiguous: direct-view form for block/outer/var/inner over memory-order
// spans.
template <typename Body>
inline void RunCpuBoviContiguousDirect(const Dataset &dataset, int logical_inner_size,
                                       Body body) {
  const auto &spec = dataset.problem;
  const auto &data = dataset.data;
  const auto logical_indexer = spec.logical_indexer;
  const auto memory_indexer = spec.memory_indexer;
  const int outer_points =
      CeilDiv(static_cast<int>(logical_indexer.size()), logical_inner_size);

  for (int b = 0; b < spec.nblocks; ++b) {
    const int nvars = SelectNvarsForBlock(dataset, b);
    for (int outer = 0; outer < outer_points; ++outer) {
      const ChunkSpan span =
          MakeChunkSpan(logical_indexer, memory_indexer, outer, logical_inner_size);
      for (int v = 0; v < nvars; ++v) {
#pragma omp simd
        for (int idx = 0; idx < span.size; ++idx) {
          const auto [k, j, i] = memory_indexer(span.memory_start + idx);
          data.out(b, v, k, j, i) = body(data, b, v, k, j, i);
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
  const int outer_points =
      CeilDiv(static_cast<int>(logical_indexer.size()), logical_inner_size);

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
  const int outer_points =
      CeilDiv(static_cast<int>(logical_indexer.size()), logical_inner_size);

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

// cpu_bvoi_contiguous: hoisted-pointer form for block/var/outer/inner over memory-order
// spans.
template <typename AccessBuilder, typename Body>
inline void RunCpuBvoiContiguous(const Dataset &dataset, int logical_inner_size,
                                 AccessBuilder build_access, Body body) {
  const auto &spec = dataset.problem;
  const auto &data = dataset.data;
  const auto logical_indexer = spec.logical_indexer;
  const auto memory_indexer = spec.memory_indexer;
  const int outer_points =
      CeilDiv(static_cast<int>(logical_indexer.size()), logical_inner_size);

  for (int b = 0; b < spec.nblocks; ++b) {
    const int nvars = SelectNvarsForBlock(dataset, b);
    for (int v = 0; v < nvars; ++v) {
      for (int outer = 0; outer < outer_points; ++outer) {
        const ChunkSpan span =
            MakeChunkSpan(logical_indexer, memory_indexer, outer, logical_inner_size);
        const auto [k, j, i] = logical_indexer(span.logical_start);
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

// cpu_bvoi_contiguous: direct-view form for block/var/outer/inner over memory-order
// spans.
template <typename Body>
inline void RunCpuBvoiContiguousDirect(const Dataset &dataset, int logical_inner_size,
                                       Body body) {
  const auto &spec = dataset.problem;
  const auto &data = dataset.data;
  const auto logical_indexer = spec.logical_indexer;
  const auto memory_indexer = spec.memory_indexer;
  const int outer_points =
      CeilDiv(static_cast<int>(logical_indexer.size()), logical_inner_size);

  for (int b = 0; b < spec.nblocks; ++b) {
    const int nvars = SelectNvarsForBlock(dataset, b);
    for (int v = 0; v < nvars; ++v) {
      for (int outer = 0; outer < outer_points; ++outer) {
        const ChunkSpan span =
            MakeChunkSpan(logical_indexer, memory_indexer, outer, logical_inner_size);

#pragma omp simd
        for (int idx = 0; idx < span.size; ++idx) {
          const auto [k, j, i] = memory_indexer(span.memory_start + idx);
          data.out(b, v, k, j, i) = body(data, b, v, k, j, i);
        }
      }
    }
  }
}

// cpu_bvoi_logical: direct-view form for block/var/outer/inner over active logical spans.
template <typename Body>
inline void RunCpuBvoiLogical(const Dataset &dataset, int logical_inner_size, Body body) {
  (void)logical_inner_size;
  const auto &spec = dataset.problem;
  const auto &data = dataset.data;
  const auto logical_indexer = spec.logical_indexer;
  const int kstart = static_cast<int>(logical_indexer.template StartIdx<0>());
  const int kend = static_cast<int>(logical_indexer.template EndIdx<0>());
  const int jstart = static_cast<int>(logical_indexer.template StartIdx<1>());
  const int jend = static_cast<int>(logical_indexer.template EndIdx<1>());
  const int istart = static_cast<int>(logical_indexer.template StartIdx<2>());
  const int iend = static_cast<int>(logical_indexer.template EndIdx<2>());

  for (int b = 0; b < spec.nblocks; ++b) {
    const int nvars = SelectNvarsForBlock(dataset, b);
    for (int v = 0; v < nvars; ++v) {
      for (int k = kstart; k <= kend; ++k) {
        for (int j = jstart; j <= jend; ++j) {
#pragma omp simd
          for (int i = istart; i <= iend; ++i) {
            data.out(b, v, k, j, i) = body(data, b, v, k, j, i);
          }
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

// kokkos_bovi_team_contiguous: TeamPolicy launch with a hoisted contiguous span per outer
// chunk.
template <typename AccessBuilder, typename Body>
inline void RunKokkosBoviTeamContiguous(const Dataset &dataset, int logical_inner_size,
                                        AccessBuilder build_access, Body body) {
  const auto &spec = dataset.problem;
  const auto &data = dataset.data;
  const auto logical_indexer = spec.logical_indexer;
  const auto memory_indexer = spec.memory_indexer;
  const int outer_points =
      CeilDiv(static_cast<int>(logical_indexer.size()), logical_inner_size);
  const int league_size = spec.nblocks * outer_points;
  const TeamPolicy policy(league_size, Kokkos::AUTO);

  Kokkos::parallel_for(
      "KokkosBoviTeamContiguous", policy, KOKKOS_LAMBDA(const TeamMember &member) {
        const int league = member.league_rank();
        const int b = league / outer_points;
        const int outer = league % outer_points;
        const ChunkSpan span =
            MakeChunkSpan(logical_indexer, memory_indexer, outer, logical_inner_size);
        const auto [k, j, i] = logical_indexer(span.logical_start);
        const int nvars = data.active_counts(b);

        for (int v = 0; v < nvars; ++v) {
          const auto access = build_access(data, b, v, k, j, i);
          double *const out = &data.out(b, v, k, j, i);
          Kokkos::parallel_for(Kokkos::TeamThreadRange(member, 0, span.size),
                               [&](const int idx) { out[idx] = body(access, idx); });
        }
      });
}

// kokkos_bovi_team_contiguous: direct-view TeamPolicy launch over contiguous spans.
template <typename Body>
inline void RunKokkosBoviTeamContiguousDirect(const Dataset &dataset,
                                              int logical_inner_size, Body body) {
  const auto &spec = dataset.problem;
  const auto &data = dataset.data;
  const auto logical_indexer = spec.logical_indexer;
  const auto memory_indexer = spec.memory_indexer;
  const int outer_points =
      CeilDiv(static_cast<int>(logical_indexer.size()), logical_inner_size);
  const int league_size = spec.nblocks * outer_points;
  const TeamPolicy policy(league_size, Kokkos::AUTO);

  Kokkos::parallel_for(
      "KokkosBoviTeamContiguousDirect", policy, KOKKOS_LAMBDA(const TeamMember &member) {
        const int league = member.league_rank();
        const int b = league / outer_points;
        const int outer = league % outer_points;
        const ChunkSpan span =
            MakeChunkSpan(logical_indexer, memory_indexer, outer, logical_inner_size);
        const int nvars = data.active_counts(b);

        for (int v = 0; v < nvars; ++v) {
          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(member, 0, span.size), [&](const int idx) {
                const auto [k, j, i] = memory_indexer(span.memory_start + idx);
                data.out(b, v, k, j, i) = body(data, b, v, k, j, i);
              });
        }
      });
}

// kokkos_bovi_team_logical: TeamPolicy launch over logical active spans, direct-view
// inside.
template <typename Body>
inline void RunKokkosBoviTeamLogical(const Dataset &dataset, int logical_inner_size,
                                     Body body) {
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
          Kokkos::parallel_for(Kokkos::TeamThreadRange(member, 0, span.size),
                               [&](const int idx) {
                                 const auto [k, j, i] = logical_indexer(span.start + idx);
                                 data.out(b, v, k, j, i) = body(data, b, v, k, j, i);
                               });
        }
      });
}

// kokkos_bovi_team_logical: same logical TeamPolicy shape, but reserve team scratch to
// measure the overhead of enabling scratch-backed storage.
template <typename Body>
inline void RunKokkosBoviTeamLogicalScratch(const Dataset &dataset,
                                            int logical_inner_size, Body body) {
  const auto &spec = dataset.problem;
  const auto &data = dataset.data;
  const auto logical_indexer = spec.logical_indexer;
  const int cells_per_block = static_cast<int>(logical_indexer.size());
  const int outer_points = CeilDiv(cells_per_block, logical_inner_size);
  const int league_size = spec.nblocks * outer_points;
  TeamPolicy policy(league_size, Kokkos::AUTO);
  policy.set_scratch_size(
      0, Kokkos::PerTeam(sizeof(double) * static_cast<std::size_t>(logical_inner_size)));

  Kokkos::parallel_for(
      "KokkosBoviTeamLogicalScratch", policy,
      KOKKOS_LAMBDA(const TeamMember &member) {
        const int league = member.league_rank();
        const int b = league / outer_points;
        const int outer = league % outer_points;
        const FlatSpan span = MakeFlatSpan(logical_indexer, outer, logical_inner_size);
        const int nvars = data.active_counts(b);

        for (int v = 0; v < nvars; ++v) {
          Kokkos::parallel_for(Kokkos::TeamThreadRange(member, 0, span.size),
                               [&](const int idx) {
                                 const auto [k, j, i] =
                                     logical_indexer(span.start + idx);
                                 data.out(b, v, k, j, i) = body(data, b, v, k, j, i);
                               });
        }
      });
}

} // namespace

} // namespace plb2
