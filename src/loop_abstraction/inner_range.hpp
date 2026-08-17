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
#ifndef LOOP_ABSTRACTION_INNER_RANGE_HPP_
#define LOOP_ABSTRACTION_INNER_RANGE_HPP_

// This file was made in part with generative AI.

// InnerIndexRange: one slice of an IndexSpace passed into inner(...). It carries the
// block index and current slice state, builds the (possibly disjoint) merged flat
// spans for a halo-extended range, and exposes the index conversions (GetKJI,
// ScratchIndex, ...) that bodies and pack/view helpers use to translate between flat,
// memory, and logical coordinates. The boiv loop tag has its own thin specialization
// whose range is a single logical point.

#include <algorithm>
#include <array>
#include <tuple>

#include "utils/indexer.hpp"

#include "loop_abstraction/halo.hpp"
#include "loop_abstraction/index_space.hpp"
#include "loop_abstraction/types.hpp"

namespace parthenon::loop_abstraction {

// Result of merging the shifted copies of a [start, end] rectangle (one copy per halo
// offset) into contiguous flat spans, expressed in some indexer's flat space.
// `span_start`/`span_end` are the enclosing flat interval (min flat start and max flat
// end over all offsets), used for scratch sizing.
template <class Halo>
struct RegionMerge {
  std::array<int, Halo::npoints> flat_start{};
  std::array<int, Halo::npoints> flat_end{};
  int nregions = 1;
  int cached_size = 0;
  int span_start = 0;
  int span_end = 0;
};

// Merge the Halo-shifted copies of the [start, end] rectangle into contiguous flat
// spans, using `idxer` to flatten coordinates. A pure operation on (idxer, Halo, ndim,
// start, end): iteration regions come from calling it with the inner-tag indexer, and
// the bovi scratch extent from calling it with the memory indexer. `ndim` selects the
// contiguous run of halo offsets kept in a reduced-dimension run (see HaloReducedRange);
// the run stays sorted, so the single-pass merge below is valid.
template <class Halo>
KOKKOS_INLINE_FUNCTION RegionMerge<Halo>
BuildRegions(const parthenon::Indexer3D &idxer, int ndim, Index3 start, Index3 end) {
  const HaloRange hrange = HaloReducedRange<Halo>(ndim);
  const int hbegin = hrange.begin;
  RegionMerge<Halo> out;
  out.flat_start[0] = idxer.GetFlatIdx(start.k + Halo::dk(hbegin),
                                       start.j + Halo::dj(hbegin),
                                       start.i + Halo::di(hbegin));
  out.flat_end[0] = idxer.GetFlatIdx(end.k + Halo::dk(hbegin), end.j + Halo::dj(hbegin),
                                     end.i + Halo::di(hbegin));
  out.span_start = out.flat_start[0];
  out.span_end = out.flat_end[0];
  out.nregions = 1;
  // Create possibly disjoint ranges, this algorithm relies on the start and end points
  // of the ranges being sorted by flat start
  for (int n = hbegin + 1; n < hrange.end; ++n) {
    const int fstart = idxer.GetFlatIdx(start.k + Halo::dk(n), start.j + Halo::dj(n),
                                        start.i + Halo::di(n));
    const int fend = idxer.GetFlatIdx(end.k + Halo::dk(n), end.j + Halo::dj(n),
                                      end.i + Halo::di(n));
    out.span_start = std::min(out.span_start, fstart);
    out.span_end = std::max(out.span_end, fend);
    if (fstart <= out.flat_end[out.nregions - 1] + 1) {
      if (fend > out.flat_end[out.nregions - 1]) out.flat_end[out.nregions - 1] = fend;
    } else {
      out.flat_start[out.nregions] = fstart;
      out.flat_end[out.nregions] = fend;
      ++out.nregions;
    }
  }
  out.cached_size = 0;
  for (int r = 0; r < out.nregions; ++r) {
    out.cached_size += out.flat_end[r] - out.flat_start[r] + 1;
  }
  return out;
}

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
  // Logical-flat bounds of this slice's chunk (inclusive), in the block-wide logical
  // indexer. Used by the reduction path to iterate logical cells even for the memory
  // inner tag (which otherwise sweeps a contiguous memory span that includes ghosts).
  // See inner_kokkos_reduce in kokkos.hpp.
  int chunk_logical_start = 0;
  int chunk_logical_end = 0;
  const device_team_member_t *team_member = nullptr;
  // Reduction accumulator, set by outer_reduce and joined into by inner_reduce. Empty
  // (zero bytes, via [[no_unique_address]]) for a non-reduction index space.
  [[no_unique_address]] impl::ReduceState<typename IndexSpaceType::reduction_t> reduce_;

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

    chunk_logical_start = 0;
    chunk_logical_end = static_cast<int>(logical_kji.size()) - 1;
    InitFromEndpoints(start, end);
  }

  // Constructor relevant for bovi
  KOKKOS_INLINE_FUNCTION
  InnerIndexRange(const IndexSpaceType &idx_space,
                  const parthenon::Indexer3D &logical_kji_in, int b, Index3 start,
                  Index3 end, const device_team_member_t *team_member_in = nullptr)
      : pidx_space(&idx_space), logical_kji(logical_kji_in), block(b), ks(start.k),
        js(start.j), is(start.i), team_member(team_member_in) {
    InitFromEndpoints(start, end);
  }

  KOKKOS_INLINE_FUNCTION
  InnerIndexRange(const IndexSpaceType &idx_space,
                  const parthenon::Indexer3D &logical_kji_in, int b, int flat_start,
                  int flat_end, const device_team_member_t *team_member_in = nullptr)
      : pidx_space(&idx_space), logical_kji(logical_kji_in), block(b),
        chunk_logical_start(flat_start), chunk_logical_end(flat_end),
        team_member(team_member_in) {
    const auto [ks_, js_, is_] = logical_kji(flat_start);
    ks = ks_;
    js = js_;
    is = is_;
    InitFromEndpoints({ks, js, is}, Index3(logical_kji(flat_end)));
  }

  template <class Halo_in>
  KOKKOS_INLINE_FUNCTION InnerIndexRange<IndexSpaceType, Halo_in> AddHalo() const {
    static_assert(std::is_same_v<Halo, halo::none_t>,
                  "Halo composition is currently not supported.");
    static_assert(impl::HaloIsProjectionClosed<Halo_in>(),
                  "Halo is not closed under projection onto the active dimensions. In a "
                  "reduced-dimension (2D/1D) run, offsets pointing into a degenerate "
                  "direction are dropped; this is only correct when every offset's "
                  "projection is itself a declared offset. Add the missing projection "
                  "point(s) to the halo (filling an unused scratch cell is a no-op).");
    const HaloRange hrange = HaloReducedRange<Halo_in>(pidx_space->GetNdim());
    parthenon::Indexer3D halo_kji = AddHaloToIndexer<Halo_in>(logical_kji, hrange);
    const auto [ke, je, ie] = GetKJIFromFlatIdx(flat_end[0]);
    return InnerIndexRange<IndexSpaceType, Halo_in>(
        *pidx_space, halo_kji, block, {ks, js, is}, {ke, je, ie}, team_member);
  }

  // Initialize the iteration regions and scratch bookkeeping from the range's endpoints.
  // Runs from every constructor after the endpoints are known; not called elsewhere.
  KOKKOS_INLINE_FUNCTION void InitFromEndpoints(const Index3 start, const Index3 end) {
    const auto &memory = pidx_space->GetMemoryIndexer();
    const int ndim = pidx_space->GetNdim();
    // Iteration regions live in the inner-tag indexer's flat space (memory span for the
    // memory tag, logical otherwise).
    const parthenon::Indexer3D &iter_indexer =
        (IndexSpaceType::inner_tag_v == inner_tag::memory) ? memory : logical_kji;
    const RegionMerge<Halo> iter = BuildRegions<Halo>(iter_indexer, ndim, start, end);
    flat_start = iter.flat_start;
    flat_end = iter.flat_end;
    nregions = iter.nregions;
    cached_size = iter.cached_size;

    if constexpr (IndexSpaceType::loop_tag_v == loop_tag::bvoi) {
      // bvoi enumerates contiguous flat spans and converts each flat index back to
      // (k,j,i), so it sweeps the whole rectangular (halo-extended) box -- including the
      // multi-axis corner cells that lie in no single shifted copy -- regardless of the
      // inner tag. Scratch must therefore cover the full box, whose extent in memory is
      // bounded by the box's low and high corners (logical_kji is that box here). Sizing
      // from the shifted-copy union instead would under-allocate and under-run the
      // buffer at those corners (e.g. a 7-point + (-2i) stencil).
      scratch_flat_start = memory.GetFlatIdx(logical_kji.template StartIdx<0>(),
                                             logical_kji.template StartIdx<1>(),
                                             logical_kji.template StartIdx<2>());
      const int scratch_flat_end = memory.GetFlatIdx(logical_kji.template EndIdx<0>(),
                                                     logical_kji.template EndIdx<1>(),
                                                     logical_kji.template EndIdx<2>());
      scratch_span_size = scratch_flat_end - scratch_flat_start + 1;
    } else {
      // bovi visits only the union of shifted copies (flat spans in the memory indexer),
      // so the enclosing memory interval suffices.
      const RegionMerge<Halo> scr = BuildRegions<Halo>(memory, ndim, start, end);
      scratch_flat_start = scr.span_start;
      scratch_span_size = scr.span_end - scr.span_start + 1;
    }
    const int memory_base = memory.GetFlatIdx(start.k, start.j, start.i);
    scratch_index_start = scratch_flat_start - memory_base;
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

template <inner_tag INNER_TAG, loop_backend BACKEND, class Reduction, class Halo>
class InnerIndexRange<IndexSpace<loop_tag::boiv, INNER_TAG, BACKEND, Reduction>, Halo> {
 public:
  using index_space_t = IndexSpace<loop_tag::boiv, INNER_TAG, BACKEND, Reduction>;
  using halo_t = Halo;
  static_assert(impl::HaloSatisfiesContract<Halo>(),
                "Halo offsets must include exactly one identity offset {0,0,0} "
                "and be strictly sorted lexicographically by (dk,dj,di).");
  const index_space_t *pidx_space = nullptr;
  int block = 0;
  int ks = 0;
  int js = 0;
  int is = 0;
  // Reduction accumulator (see the primary template). Empty for a non-reduction space.
  [[no_unique_address]] impl::ReduceState<Reduction> reduce_;

  template <class Halo_in>
  KOKKOS_INLINE_FUNCTION InnerIndexRange<index_space_t, Halo_in> AddHalo() const {
    static_assert(std::is_same_v<Halo, halo::none_t>,
                  "Halo composition is currently not supported.");
    static_assert(impl::HaloIsProjectionClosed<Halo_in>(),
                  "Halo is not closed under projection onto the active dimensions. In a "
                  "reduced-dimension (2D/1D) run, offsets pointing into a degenerate "
                  "direction are dropped; this is only correct when every offset's "
                  "projection is itself a declared offset. Add the missing projection "
                  "point(s) to the halo (filling an unused scratch cell is a no-op).");
    InnerIndexRange<index_space_t, Halo_in> out;
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
  int size() const {
    const HaloRange r = HaloReducedRange<halo_t>(pidx_space->GetNdim());
    return r.end - r.begin;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  void TeamBarrier() const {}
};

} // namespace parthenon::loop_abstraction

#endif // LOOP_ABSTRACTION_INNER_RANGE_HPP_
