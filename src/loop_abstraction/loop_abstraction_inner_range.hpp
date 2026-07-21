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
#ifndef LOOP_ABSTRACTION_LOOP_ABSTRACTION_INNER_RANGE_HPP_
#define LOOP_ABSTRACTION_LOOP_ABSTRACTION_INNER_RANGE_HPP_

// This file was made in part with generative AI.

// InnerIndexRange: one slice of an IndexSpace passed into inner(...). It carries the
// block index and current slice state, builds the (possibly disjoint) merged flat
// spans for a halo-extended range, and exposes the index conversions (GetKJI,
// ScratchIndex, ...) that bodies and pack/view helpers use to translate between flat,
// memory, and logical coordinates. The boiv loop tag has its own thin specialization
// whose range is a single logical point.

#include <array>
#include <tuple>

#include "utils/indexer.hpp"

#include "loop_abstraction/loop_abstraction_halo.hpp"
#include "loop_abstraction/loop_abstraction_index_space.hpp"
#include "loop_abstraction/loop_abstraction_types.hpp"

namespace parthenon::loop_abstraction {

template <class IndexSpaceType, class Halo = halo::none_t>
class InnerIndexRange {
 public:
  using index_space_t = IndexSpaceType;
  using halo_t = Halo;
  static_assert(impl::HaloSatisfiesContract<Halo>(),
                "Halo offsets must include exactly one identity offset {0,0,0} "
                "and be strictly sorted lexicographically by (dk,dj,di).");

  const IndexSpaceType *pidx_space = nullptr;
  parthenon::Indexer3D logical_kji;
  int block = 0;
  std::array<int, Halo::npoints> flat_start{};
  std::array<int, Halo::npoints> flat_end{};
  int nregions = 1;
  int cached_size = 0;
  int scratch_flat_start = 0;
  int scratch_index_start = 0;
  int scratch_span_size = 0;
  int ks = 0;
  int js = 0;
  int is = 0;
  const device_team_member_t *team_member = nullptr;

  KOKKOS_FORCEINLINE_FUNCTION
  void TeamBarrier() const {
    if (team_member) team_member->team_barrier();
  }

  // Constructor relevant for bvoi
  KOKKOS_INLINE_FUNCTION
  InnerIndexRange(const IndexSpaceType &idx_space,
                  const parthenon::Indexer3D &logical_kji_in, int b,
                  const device_team_member_t *team_member_in = nullptr)
      : pidx_space(&idx_space), logical_kji(logical_kji_in), block(b),
        ks(logical_kji.template StartIdx<0>()), js(logical_kji.template StartIdx<1>()),
        is(logical_kji.template StartIdx<2>()), team_member(team_member_in) {
    const Index3 start{logical_kji.template StartIdx<0>(),
                       logical_kji.template StartIdx<1>(),
                       logical_kji.template StartIdx<2>()};

    const Index3 end{logical_kji.template EndIdx<0>(), logical_kji.template EndIdx<1>(),
                     logical_kji.template EndIdx<2>()};

    BuildRegionsFromEndpoints(start, end);
  }

  // Constructor relevant for bovi
  KOKKOS_INLINE_FUNCTION
  InnerIndexRange(const IndexSpaceType &idx_space,
                  const parthenon::Indexer3D &logical_kji_in, int b, Index3 start,
                  Index3 end, const device_team_member_t *team_member_in = nullptr)
      : pidx_space(&idx_space), logical_kji(logical_kji_in), block(b), ks(start.k),
        js(start.j), is(start.i), team_member(team_member_in) {
    BuildRegionsFromEndpoints(start, end);
  }

  KOKKOS_INLINE_FUNCTION
  InnerIndexRange(const IndexSpaceType &idx_space,
                  const parthenon::Indexer3D &logical_kji_in, int b, int flat_start,
                  int flat_end, const device_team_member_t *team_member_in = nullptr)
      : pidx_space(&idx_space), logical_kji(logical_kji_in), block(b),
        team_member(team_member_in) {
    const auto [ks_, js_, is_] = logical_kji(flat_start);
    ks = ks_;
    js = js_;
    is = is_;
    BuildRegionsFromEndpoints({ks, js, is}, logical_kji(flat_end));
  }

  template <class Halo_in>
  KOKKOS_INLINE_FUNCTION InnerIndexRange<IndexSpaceType, Halo_in> AddHalo() const {
    static_assert(std::is_same_v<Halo, halo::none_t>,
                  "Halo composition is currently not supported.");
    parthenon::Indexer3D halo_kji = AddHaloToIndexer<Halo_in>(logical_kji);
    const auto [ke, je, ie] = GetKJIFromFlatIdx(flat_end[0]);
    return InnerIndexRange<IndexSpaceType, Halo_in>(
        *pidx_space, halo_kji, block, {ks, js, is}, {ke, je, ie}, team_member);
  }

  KOKKOS_INLINE_FUNCTION void BuildRegionsFromEndpoints(const Index3 start,
                                                        const Index3 end) {
    const auto &memory = pidx_space->GetMemoryIndexer();
    flat_start[0] = GetFlatIdxFromKJI(start.k + Halo::dk(0), start.j + Halo::dj(0),
                                      start.i + Halo::di(0));
    flat_end[0] =
        GetFlatIdxFromKJI(end.k + Halo::dk(0), end.j + Halo::dj(0), end.i + Halo::di(0));
    const int memory_base = memory.GetFlatIdx(start.k, start.j, start.i);
    scratch_flat_start = memory.GetFlatIdx(start.k + Halo::dk(0), start.j + Halo::dj(0),
                                           start.i + Halo::di(0));
    int scratch_flat_end =
        memory.GetFlatIdx(end.k + Halo::dk(0), end.j + Halo::dj(0), end.i + Halo::di(0));
    nregions = 1;
    // Create possibly disjoint ranges, this algorithm relies on the start and end points
    // of the ranges being sorted by flat start
    for (int n = 1; n < Halo::npoints; ++n) {
      const int fstart = GetFlatIdxFromKJI(start.k + Halo::dk(n), start.j + Halo::dj(n),
                                           start.i + Halo::di(n));
      const int fend = GetFlatIdxFromKJI(end.k + Halo::dk(n), end.j + Halo::dj(n),
                                         end.i + Halo::di(n));
      const int scratch_start = memory.GetFlatIdx(
          start.k + Halo::dk(n), start.j + Halo::dj(n), start.i + Halo::di(n));
      const int scratch_end = memory.GetFlatIdx(end.k + Halo::dk(n), end.j + Halo::dj(n),
                                                end.i + Halo::di(n));
      scratch_flat_start = std::min(scratch_flat_start, scratch_start);
      scratch_flat_end = std::max(scratch_flat_end, scratch_end);
      if (fstart <= flat_end[nregions - 1] + 1) {
        if (fend > flat_end[nregions - 1]) flat_end[nregions - 1] = fend;
      } else {
        flat_start[nregions] = fstart;
        flat_end[nregions] = fend;
        ++nregions;
      }
    }
    cached_size = 0;
    for (int r = 0; r < nregions; ++r) {
      cached_size += flat_end[r] - flat_start[r] + 1;
    }
    scratch_index_start = scratch_flat_start - memory_base;
    scratch_span_size = scratch_flat_end - scratch_flat_start + 1;
  }

  KOKKOS_FORCEINLINE_FUNCTION auto GetKJIFromFlatIdx(int flat_idx) const {
    if constexpr (IndexSpaceType::inner_tag_v == inner_tag::memory) {
      return pidx_space->GetMemoryIndexer()(flat_idx);
    } else {
      return logical_kji(flat_idx);
    }
  }

  KOKKOS_FORCEINLINE_FUNCTION auto GetFlatIdxFromKJI(int k, int j, int i) const {
    if constexpr (IndexSpaceType::inner_tag_v == inner_tag::memory) {
      return pidx_space->GetMemoryIndexer().GetFlatIdx(k, j, i);
    } else {
      return logical_kji.GetFlatIdx(k, j, i);
    }
  }

  KOKKOS_FORCEINLINE_FUNCTION auto GetFlatIdxFromMemoryIdx(int mem_idx) const {
    if constexpr (IndexSpaceType::inner_tag_v == inner_tag::memory) {
      const int mem_shift = pidx_space->GetMemoryIndexer().GetFlatIdx(ks, js, is);
      return mem_idx + mem_shift;
    } else {
      const auto [k, j, i] = GetKJI(mem_idx);
      return logical_kji.GetFlatIdx(k, j, i);
    }
  }

  KOKKOS_INLINE_FUNCTION std::tuple<int, int, int> GetKJI(int mem_idx) const {
    const int mem_shift = pidx_space->GetMemoryIndexer().GetFlatIdx(ks, js, is);
    return pidx_space->GetMemoryIndexer()(mem_idx + mem_shift);
  }

  KOKKOS_INLINE_FUNCTION std::tuple<int, int, int> GetKJI(MemoryOffset idx) const {
    return GetKJI(idx.flat);
  }

  KOKKOS_INLINE_FUNCTION std::tuple<int, int, int> GetKJI(Index3 idx) const {
    return {idx.k, idx.j, idx.i};
  }

  KOKKOS_INLINE_FUNCTION
  int size() const { return cached_size; }

  KOKKOS_FORCEINLINE_FUNCTION
  int ScratchSize() const { return scratch_span_size; }

  KOKKOS_FORCEINLINE_FUNCTION
  int ScratchIndex(int mem_idx) const { return mem_idx - scratch_index_start; }

  KOKKOS_FORCEINLINE_FUNCTION
  int ScratchIndex(MemoryOffset idx) const { return ScratchIndex(idx.flat); }

  KOKKOS_FORCEINLINE_FUNCTION
  int ScratchIndex(Index3 idx) const {
    return pidx_space->GetMemoryIndexer().GetFlatIdx(idx.k, idx.j, idx.i) -
           scratch_flat_start;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  int ScratchIndex(const int k, const int j, const int i) const {
    return pidx_space->GetMemoryIndexer().GetFlatIdx(k, j, i) - scratch_flat_start;
  }

  // CompactIndex maps the possibly disjoint merged halo spans onto a dense
  // zero-based index space. This is the minimal-footprint scratch indexing model.
  // Scratch currently uses the enclosing memory-flat span instead, so these helpers
  // are unused but kept here as a reference path and possible future option.
  KOKKOS_INLINE_FUNCTION
  int CompactIndexFromFlat(int flat_idx) const {
    int offset = 0;

    for (int r = 0; r < nregions; ++r) {
      if (flat_idx >= flat_start[r] && flat_idx <= flat_end[r]) {
        return offset + (flat_idx - flat_start[r]);
      }

      offset += flat_end[r] - flat_start[r] + 1;
    }

    return -1;
  }

  KOKKOS_INLINE_FUNCTION
  int CompactIndex(int mem_idx) const {
    return CompactIndexFromFlat(GetFlatIdxFromMemoryIdx(mem_idx));
  }

  KOKKOS_INLINE_FUNCTION
  int CompactIndex(Index3 idx) const {
    return CompactIndexFromFlat(GetFlatIdxFromKJI(idx.k, idx.j, idx.i));
  }

  KOKKOS_INLINE_FUNCTION
  int CompactIndex(const int k, const int j, const int i) const {
    return CompactIndexFromFlat(GetFlatIdxFromKJI(k, j, i));
  }
};

template <inner_tag INNER_TAG, loop_backend BACKEND, class Halo>
class InnerIndexRange<IndexSpace<loop_tag::boiv, INNER_TAG, BACKEND>, Halo> {
 public:
  using index_space_t = IndexSpace<loop_tag::boiv, INNER_TAG, BACKEND>;
  using halo_t = Halo;
  static_assert(impl::HaloSatisfiesContract<Halo>(),
                "Halo offsets must include exactly one identity offset {0,0,0} "
                "and be strictly sorted lexicographically by (dk,dj,di).");
  const IndexSpace<loop_tag::boiv, INNER_TAG, BACKEND> *pidx_space = nullptr;
  int block = 0;
  int ks = 0;
  int js = 0;
  int is = 0;

  template <class Halo_in>
  KOKKOS_INLINE_FUNCTION
      InnerIndexRange<IndexSpace<loop_tag::boiv, INNER_TAG, BACKEND>, Halo_in>
      AddHalo() const {
    static_assert(std::is_same_v<Halo, halo::none_t>,
                  "Halo composition is currently not supported.");
    InnerIndexRange<IndexSpace<loop_tag::boiv, INNER_TAG, BACKEND>, Halo_in> out;
    out.pidx_space = pidx_space;
    out.block = block;
    out.ks = ks;
    out.js = js;
    out.is = is;
    return out;
  }

  KOKKOS_INLINE_FUNCTION std::tuple<int, int, int> GetKJI(int idx) const {
    const int shift = pidx_space->GetMemoryIndexer().GetFlatIdx(ks, js, is);
    return pidx_space->GetMemoryIndexer()(idx + shift);
  }

  KOKKOS_INLINE_FUNCTION std::tuple<int, int, int> GetKJI(MemoryOffset idx) const {
    return {ks + idx.dk, js + idx.dj, is + idx.di};
  }

  KOKKOS_INLINE_FUNCTION std::tuple<int, int, int> GetKJI(Index3 idx) const {
    return {idx.k, idx.j, idx.i};
  }

  KOKKOS_INLINE_FUNCTION
  int size() const { return halo_t::npoints; }

  KOKKOS_FORCEINLINE_FUNCTION
  void TeamBarrier() const {}
};

} // namespace parthenon::loop_abstraction

#endif // LOOP_ABSTRACTION_LOOP_ABSTRACTION_INNER_RANGE_HPP_
