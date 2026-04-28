#pragma once

#include <algorithm>

#include <Kokkos_Core.hpp>

#include "dataset.hpp"
#include "kernels.hpp"

namespace plb2 {

namespace {

using TeamPolicy = Kokkos::TeamPolicy<>;
using TeamMember = TeamPolicy::member_type;

struct BlockSpan {
  int start = 0;
  int end = -1;
  int size = 0;
};

KOKKOS_INLINE_FUNCTION BlockSpan GetBlockSpan(const parthenon::Indexer4D &indexer, int b) {
  const int start =
      indexer.GetFlatIdx(b, indexer.StartIdx<1>(), indexer.StartIdx<2>(), indexer.StartIdx<3>());
  const int end =
      indexer.GetFlatIdx(b, indexer.EndIdx<1>(), indexer.EndIdx<2>(), indexer.EndIdx<3>());
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
// cpu_flat_ghosts: flat walk over the full 5D memory space, including ghosts.
inline void RunCpuFlatGhosts(const Dataset &dataset, Body body) {
  const auto &spec = dataset.problem;
  const auto &data = dataset.data;
  const auto memory_indexer = spec.memory_indexer;

#pragma omp simd
  for (int flat = 0; flat < static_cast<int>(memory_indexer.size()); ++flat) {
    const auto [b, v, k, j, i] = memory_indexer(flat);
    data.out(b, v, k, j, i) = body(data, b, v, k, j, i);
  }
}

template <typename Body>
// cpu_boiv_contiguous: direct-view form for the block/outer/inner/var order.
// The inner span is contiguous, but the kernel still reads through the view.
inline void RunCpuBoivContiguous(const Dataset &dataset, int logical_inner_size, Body body) {
  const auto &spec = dataset.problem;
  const auto &data = dataset.data;
  const auto logical_indexer = spec.logical_indexer;

  for (int b = 0; b < spec.nblocks; ++b) {
    const BlockSpan block_span = GetBlockSpan(logical_indexer, b);
    const int outer_points = CeilDiv(block_span.size, logical_inner_size);
    const int nvars = SelectNvarsForBlock(dataset, b);

    for (int outer = 0; outer < outer_points; ++outer) {
      const int logical_start = block_span.start + outer * logical_inner_size;
      const int logical_end =
          std::min(block_span.end, logical_start + logical_inner_size - 1);
      const int ninner = logical_end - logical_start + 1;

      for (int idx = 0; idx < ninner; ++idx) {
        const auto [bb, k, j, i] = logical_indexer(logical_start + idx);
        for (int v = 0; v < nvars; ++v) {
          data.out(bb, v, k, j, i) = body(data, bb, v, k, j, i);
        }
      }
    }
  }
}

template <typename AccessBuilder, typename Body>
// cpu_bovi_contiguous: hoisted-pointer form for block/outer/var/inner.
// build_access() is called once per outer span and the inner loop indexes the hoisted span.
inline void RunCpuBoviContiguous(const Dataset &dataset, int logical_inner_size,
                                 AccessBuilder build_access, Body body) {
  const auto &spec = dataset.problem;
  const auto &data = dataset.data;
  const auto logical_indexer = spec.logical_indexer;

  for (int b = 0; b < spec.nblocks; ++b) {
    const BlockSpan block_span = GetBlockSpan(logical_indexer, b);
    const int outer_points = CeilDiv(block_span.size, logical_inner_size);
    const int nvars = SelectNvarsForBlock(dataset, b);

    for (int outer = 0; outer < outer_points; ++outer) {
      const int logical_start = block_span.start + outer * logical_inner_size;
      const int logical_end =
          std::min(block_span.end, logical_start + logical_inner_size - 1);
      const int ninner = logical_end - logical_start + 1;
      const auto [bs, ks, js, is] = logical_indexer(logical_start);

      for (int v = 0; v < nvars; ++v) {
        const auto access = build_access(data, b, v, ks, js, is);
        double *const out = &data.out(b, v, ks, js, is);
#pragma omp simd
        for (int idx = 0; idx < ninner; ++idx) {
          out[idx] = body(access, idx);
        }
      }
    }
  }
}

template <typename Body>
// cpu_boiv_logical: direct-view form for block/outer/inner/var over logical active cells.
inline void RunCpuBoivLogical(const Dataset &dataset, int logical_inner_size, Body body) {
  const auto &spec = dataset.problem;
  const auto &data = dataset.data;
  const auto logical_indexer = spec.logical_indexer;

  for (int b = 0; b < spec.nblocks; ++b) {
    const BlockSpan block_span = GetBlockSpan(logical_indexer, b);
    const int outer_points = CeilDiv(block_span.size, logical_inner_size);
    const int nvars = SelectNvarsForBlock(dataset, b);

    for (int outer = 0; outer < outer_points; ++outer) {
      const int logical_start = block_span.start + outer * logical_inner_size;
      const int logical_end =
          std::min(block_span.end, logical_start + logical_inner_size - 1);

      for (int idx = logical_start; idx <= logical_end; ++idx) {
        const auto [bb, k, j, i] = logical_indexer(idx);
        for (int v = 0; v < nvars; ++v) {
          data.out(bb, v, k, j, i) = body(data, bb, v, k, j, i);
        }
      }
    }
  }
}

template <typename Body>
// cpu_bovi_logical: direct-view form for block/outer/var/inner over logical active cells.
inline void RunCpuBoviLogical(const Dataset &dataset, int logical_inner_size, Body body) {
  const auto &spec = dataset.problem;
  const auto &data = dataset.data;
  const auto logical_indexer = spec.logical_indexer;

  for (int b = 0; b < spec.nblocks; ++b) {
    const BlockSpan block_span = GetBlockSpan(logical_indexer, b);
    const int outer_points = CeilDiv(block_span.size, logical_inner_size);
    const int nvars = SelectNvarsForBlock(dataset, b);

    for (int outer = 0; outer < outer_points; ++outer) {
      const int logical_start = block_span.start + outer * logical_inner_size;
      const int logical_end =
          std::min(block_span.end, logical_start + logical_inner_size - 1);

      for (int v = 0; v < nvars; ++v) {
#pragma omp simd
        for (int idx = logical_start; idx <= logical_end; ++idx) {
          const auto [bb, k, j, i] = logical_indexer(idx);
          data.out(bb, v, k, j, i) = body(data, bb, v, k, j, i);
        }
      }
    }
  }
}

template <typename AccessBuilder, typename Body>
// cpu_bvoi_contiguous: hoisted-pointer form for block/var/outer/inner.
// This swaps the variable loop ahead of the outer chunk loop, then walks a contiguous inner span.
inline void RunCpuBvoiContiguous(const Dataset &dataset, int logical_inner_size,
                                 AccessBuilder build_access, Body body) {
  const auto &spec = dataset.problem;
  const auto &data = dataset.data;
  const auto logical_indexer = spec.logical_indexer;

  for (int b = 0; b < spec.nblocks; ++b) {
    const BlockSpan block_span = GetBlockSpan(logical_indexer, b);
    const int outer_points = CeilDiv(block_span.size, logical_inner_size);
    const int nvars = SelectNvarsForBlock(dataset, b);

    for (int v = 0; v < nvars; ++v) {
      for (int outer = 0; outer < outer_points; ++outer) {
        const int logical_start = block_span.start + outer * logical_inner_size;
        const int logical_end =
            std::min(block_span.end, logical_start + logical_inner_size - 1);
        const int ninner = logical_end - logical_start + 1;
        const auto [bs, ks, js, is] = logical_indexer(logical_start);
        const auto access = build_access(data, b, v, ks, js, is);
        double *const out = &data.out(b, v, ks, js, is);

#pragma omp simd
        for (int idx = 0; idx < ninner; ++idx) {
          out[idx] = body(access, idx);
        }
      }
    }
  }
}

template <typename Body>
// cpu_bvoi_logical: direct-view form for block/var/outer/inner over logical active cells.
inline void RunCpuBvoiLogical(const Dataset &dataset, int logical_inner_size, Body body) {
  const auto &spec = dataset.problem;
  const auto &data = dataset.data;
  const auto logical_indexer = spec.logical_indexer;

  for (int b = 0; b < spec.nblocks; ++b) {
    const BlockSpan block_span = GetBlockSpan(logical_indexer, b);
    const int outer_points = CeilDiv(block_span.size, logical_inner_size);
    const int nvars = SelectNvarsForBlock(dataset, b);

    for (int v = 0; v < nvars; ++v) {
      for (int outer = 0; outer < outer_points; ++outer) {
        const int logical_start = block_span.start + outer * logical_inner_size;
        const int logical_end =
            std::min(block_span.end, logical_start + logical_inner_size - 1);

#pragma omp simd
        for (int idx = logical_start; idx <= logical_end; ++idx) {
          const auto [bb, k, j, i] = logical_indexer(idx);
          data.out(bb, v, k, j, i) = body(data, bb, v, k, j, i);
        }
      }
    }
  }
}

template <typename Body>
// kokkos_boiv_flat: single RangePolicy launch over logical (b,k,j,i), with a serial variable loop.
inline void RunKokkosBoivFlat(const Dataset &dataset, Body body) {
  const auto &spec = dataset.problem;
  const auto &data = dataset.data;
  const auto logical_indexer = spec.logical_indexer;

  Kokkos::parallel_for(
      "KokkosBoivFlat",
      Kokkos::RangePolicy<>(0, static_cast<int>(logical_indexer.size())),
      KOKKOS_LAMBDA(const int flat) {
        const auto [b, k, j, i] = logical_indexer(flat);
        const int nvars = data.active_counts(b);
        for (int v = 0; v < nvars; ++v) {
          data.out(b, v, k, j, i) = body(data, b, v, k, j, i);
        }
      });
}

template <typename AccessBuilder, typename Body>
// kokkos_bovi_team_contiguous: TeamPolicy launch with a hoisted contiguous span per outer chunk.
inline void RunKokkosBoviTeamContiguous(const Dataset &dataset, int logical_inner_size,
                                        AccessBuilder build_access, Body body) {
  const auto &spec = dataset.problem;
  const auto &data = dataset.data;
  const auto logical_indexer = spec.logical_indexer;
  const int cells_per_block = static_cast<int>(logical_indexer.size()) / spec.nblocks;
  const int outer_points = CeilDiv(cells_per_block, logical_inner_size);
  const int league_size = spec.nblocks * outer_points;
  const TeamPolicy policy(league_size, Kokkos::AUTO);

  Kokkos::parallel_for(
      "KokkosBoviTeamContiguous", policy, KOKKOS_LAMBDA(const TeamMember &member) {
        const int league = member.league_rank();
        const int b = league / outer_points;
        const int outer = league % outer_points;
        const int block_base = b * cells_per_block;
        const int logical_start = block_base + outer * logical_inner_size;
        const int logical_end =
            std::min(block_base + cells_per_block - 1, logical_start + logical_inner_size - 1);
        const int ninner = logical_end - logical_start + 1;
        const auto [bs, ks, js, is] = logical_indexer(logical_start);
        const int nvars = data.active_counts(b);

        for (int v = 0; v < nvars; ++v) {
          const auto access = build_access(data, b, v, ks, js, is);
          double *const out = &data.out(b, v, ks, js, is);
          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(member, 0, ninner),
              KOKKOS_LAMBDA(const int idx) { out[idx] = body(access, idx); });
        }
      });
}

template <typename Body>
// kokkos_bovi_team_logical: TeamPolicy launch over logical active cells, direct-view inside the team.
inline void RunKokkosBoviTeamLogical(const Dataset &dataset, int logical_inner_size, Body body) {
  const auto &spec = dataset.problem;
  const auto &data = dataset.data;
  const auto logical_indexer = spec.logical_indexer;
  const int cells_per_block = static_cast<int>(logical_indexer.size()) / spec.nblocks;
  const int outer_points = CeilDiv(cells_per_block, logical_inner_size);
  const int league_size = spec.nblocks * outer_points;
  const TeamPolicy policy(league_size, Kokkos::AUTO);

  Kokkos::parallel_for(
      "KokkosBoviTeamLogical", policy, KOKKOS_LAMBDA(const TeamMember &member) {
        const int league = member.league_rank();
        const int b = league / outer_points;
        const int outer = league % outer_points;
        const int block_base = b * cells_per_block;
        const int logical_start = block_base + outer * logical_inner_size;
        const int logical_end =
            std::min(block_base + cells_per_block - 1, logical_start + logical_inner_size - 1);
        const int nvars = data.active_counts(b);

        for (int v = 0; v < nvars; ++v) {
          Kokkos::parallel_for(Kokkos::TeamThreadRange(member, logical_start, logical_end + 1),
                               KOKKOS_LAMBDA(const int idx) {
                                 const auto [bb, k, j, i] = logical_indexer(idx);
                                 data.out(bb, v, k, j, i) = body(data, bb, v, k, j, i);
                               });
        }
      });
}

}  // namespace

}  // namespace plb2
