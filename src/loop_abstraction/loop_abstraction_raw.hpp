//========================================================================================
// (C) (or copyright) 2024-2026. Triad National Security, LLC. All rights reserved.
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
#ifndef LOOP_ABSTRACTION_LOOP_ABSTRACTION_RAW_HPP_
#define LOOP_ABSTRACTION_LOOP_ABSTRACTION_RAW_HPP_

// This file was made in part with generative AI.

#include "loop_abstraction_base.hpp"
#include "utils/bump_arena.hpp"

namespace parthenon::loop_abstraction::impl {

template <class IndexSpaceType, class F>
void outer_raw_for(IndexSpaceType idx_space, F &&f) {
  using InnerIndexRangeType = InnerIndexRange<IndexSpaceType>;
  if constexpr (IndexSpaceType::loop_tag_v == loop_tag::bvoi) {
    const auto &logical_kji = idx_space.GetLogicalIndexer();
    for (int b = 0; b < idx_space.GetNBlocks(); ++b) {
      // Reclaim last iteration's per-point scratch (see ThreadLocalBumpArena).
      parthenon::GetThreadLocalBumpArena().reset();
      InnerIndexRangeType idx_range(idx_space, idx_space.GetLogicalIndexer(), b);
      f(idx_range, b);
    }
  } else if constexpr (IndexSpaceType::loop_tag_v == loop_tag::bovi) {
    const int nouter = GetNOuter(idx_space);
    for (int b = 0; b < idx_space.GetNBlocks(); ++b) {
      for (int o = 0; o < nouter; ++o) {
        // Reclaim last iteration's per-point scratch (see ThreadLocalBumpArena).
        parthenon::GetThreadLocalBumpArena().reset();
        const int logical_start = o * idx_space.GetNInner();
        const int logical_end =
            std::min((o + 1) * idx_space.GetNInner() - 1,
                     static_cast<int>(idx_space.GetLogicalIndexer().size()) - 1);
        InnerIndexRangeType idx_range(idx_space, idx_space.GetLogicalIndexer(), b, logical_start, logical_end);
        f(idx_range, b);
      }
    }
  } else if constexpr (IndexSpaceType::loop_tag_v == loop_tag::boiv) {
    const auto &logical_kji = idx_space.GetLogicalIndexer();
    const int ks = logical_kji.template StartIdx<0>();
    const int ke = logical_kji.template EndIdx<0>();
    const int js = logical_kji.template StartIdx<1>();
    const int je = logical_kji.template EndIdx<1>();
    const int is = logical_kji.template StartIdx<2>();
    const int ie = logical_kji.template EndIdx<2>();
    InnerIndexRangeType idx_range;
    idx_range.pidx_space = &idx_space;
    for (idx_range.block = 0; idx_range.block < idx_space.GetNBlocks();
         ++idx_range.block) {
      for (int k = ks; k <= ke; ++k) {
        for (int j = js; j <= je; ++j) {
#pragma omp simd
          for (int i = is; i <= ie; ++i) {
            idx_range.ks = k;
            idx_range.js = j;
            idx_range.is = i;
            f(idx_range, idx_range.block);
          }
        }
      }
    }
  }
}

template <class InnerIndexRangeType, class F>
KOKKOS_FORCEINLINE_FUNCTION void inner_raw_for(const InnerIndexRangeType &idx_range,
                                               F &&f) {
  using IndexSpaceType =
      std::remove_cv_t<std::remove_reference_t<decltype(*idx_range.pidx_space)>>;
  const auto &idx_space = *(idx_range.pidx_space);
  const auto &memory_kji = idx_space.GetMemoryIndexer(); 
  if constexpr (IndexSpaceType::loop_tag_v == loop_tag::bvoi) {
    const auto &logical_kji = idx_range.logical_kji; 
    if constexpr (IndexSpaceType::inner_tag_v == inner_tag::logical_flat ||
                  IndexSpaceType::inner_tag_v == inner_tag::logical_coords) {
      const int ks = logical_kji.template StartIdx<0>();
      const int ke = logical_kji.template EndIdx<0>();
      const int js = logical_kji.template StartIdx<1>();
      const int je = logical_kji.template EndIdx<1>();
      const int is = logical_kji.template StartIdx<2>();
      const int ie = logical_kji.template EndIdx<2>();
      const int mem_start = memory_kji.GetFlatIdx(idx_range.ks, idx_range.js, idx_range.is);
      for (int k = ks; k <= ke; ++k) {
        for (int j = js; j <= je; ++j) {
#pragma omp simd
          for (int i = is; i <= ie; ++i) {
            if constexpr (std::is_invocable_v<F, int, int, int>) {
              f(k, j, i);
            } else if constexpr (IndexSpaceType::inner_tag_v == inner_tag::logical_flat) {
              f(memory_kji.GetFlatIdx(k, j, i) - mem_start);
            } else {
              f(Index3{k, j, i});
            }
          }
        }
      }
    } else if constexpr (IndexSpaceType::inner_tag_v == inner_tag::memory) {
      // Chunk the *halo-extended* logical space directly (idx_range.logical_kji is
      // already the extended indexer), exactly as the logical-tag branch above
      // iterates it -- rather than chunking the base space and re-applying the halo
      // per chunk. The latter double-counts cells on chunk boundaries when the halo
      // is aligned with the chunk-iteration direction (adjacent chunks' halo images
      // overlap). Here each chunk of the extended space maps to one contiguous
      // memory span [memflat(start), memflat(end)]; because memory-flat order agrees
      // with the extended lexicographic order, consecutive chunks' inclusive spans
      // are strictly increasing and cannot overlap, so every extended cell (and any
      // ghost cell swept inside a span) is touched exactly once. ninner still bounds
      // the chunk size, keeping the swept ghost work small.
      const int mem_start = memory_kji.GetFlatIdx(idx_range.ks, idx_range.js, idx_range.is);
      const auto &ext_logical_kji = idx_range.logical_kji;
      // Resolve the chunk shape against the *extended* indexer so, e.g., an ij_slab
      // is one extended plane and chunks land on clean plane boundaries.
      const int ninner = idx_space.GetNInner(ext_logical_kji);
      const int ext_size = static_cast<int>(ext_logical_kji.size());
      const int nouter = ext_size / ninner + (ext_size % ninner != 0);
      for (int o = 0; o < nouter; ++o) {
        const int logical_start = o * ninner;
        const int logical_end = std::min((o + 1) * ninner - 1, ext_size - 1);
        const auto [ks, js, is] = ext_logical_kji(logical_start);
        const auto [ke, je, ie] = ext_logical_kji(logical_end);
        const int mem_first = memory_kji.GetFlatIdx(ks, js, is);
        const int mem_last = memory_kji.GetFlatIdx(ke, je, ie);
#pragma omp simd
        for (int idx = mem_first; idx <= mem_last; ++idx) {
          if constexpr (std::is_invocable_v<F, int, int, int>) {
            const auto [k, j, i] = memory_kji(idx);
            f(k, j, i);
          } else {
            f(idx - mem_start);
          }
        }
      }
    }
  } else if constexpr (IndexSpaceType::loop_tag_v == loop_tag::bovi) {
    const auto &logical_kji = idx_range.logical_kji; 
    const int mem_base = memory_kji.GetFlatIdx(idx_range.ks, idx_range.js, idx_range.is);
    for (int r = 0; r < idx_range.nregions; ++r) {
      const int start = idx_range.flat_start[r];
      const int end_exclusive = idx_range.flat_end[r] + 1 - start;
#pragma omp simd
      for (int idx = 0; idx < end_exclusive; ++idx) {
        if constexpr (std::is_invocable_v<F, int, int, int>) {
          if constexpr (IndexSpaceType::inner_tag_v == inner_tag::memory) {
            const auto [k, j, i] = memory_kji(idx + start);
            f(k, j, i);
          } else {
            const auto [k, j, i] = logical_kji(idx + start);
            f(k, j, i);
          }
        } else if constexpr (IndexSpaceType::inner_tag_v == inner_tag::memory) {
          f(idx + start - mem_base);
        } else if constexpr (IndexSpaceType::inner_tag_v == inner_tag::logical_flat) {
          const auto [k, j, i] = logical_kji(idx + start);
          f(memory_kji.GetFlatIdx(k, j, i) - mem_base);
        } else {
          const auto [k, j, i] = logical_kji(idx + start);
          f(Index3{k, j, i});
        }
      }
    }
  } else if constexpr (IndexSpaceType::loop_tag_v == loop_tag::boiv) {
    using halo_t = InnerIndexRangeType::halo_t;
    if constexpr (IndexSpaceType::inner_tag_v == inner_tag::logical_flat) {
      if constexpr (std::is_invocable_v<F, int, int, int>) {
        for (int n = 0; n < halo_t::npoints; ++n)
          f(idx_range.ks + halo_t::dk(n), idx_range.js + halo_t::dj(n), idx_range.is + halo_t::di(n));
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
          f(idx_range.ks + halo_t::dk(n), idx_range.js + halo_t::dj(n), idx_range.is + halo_t::di(n));
      } else {
        for (int n = 0; n < halo_t::npoints; ++n)
          f(Index3{idx_range.ks + halo_t::dk(n), idx_range.js + halo_t::dj(n), idx_range.is + halo_t::di(n)});
      }
    }
  }
}

} // namespace parthenon::loop_abstraction::impl

#endif // LOOP_ABSTRACTION_LOOP_ABSTRACTION_RAW_HPP_
