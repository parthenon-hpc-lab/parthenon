#pragma once

#include <algorithm>
#include <array>
#include <concepts>
#include <optional>
#include <typeindex>
#include <unordered_map>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "basic_types.hpp"
#include "interface/mesh_data.hpp"
#include "kokkos_types.hpp"
#include "mesh/mesh.hpp"
#include "utils/indexer.hpp"
#include "utils/concepts_lite.hpp"


namespace loop_abstraction {

using device_team_member_t =
    typename Kokkos::TeamPolicy<parthenon::DevExecSpace>::member_type;

namespace halo {
// Halo types enumerate the shifted copies of an inner range that should be
// visited, including the identity offset {0,0,0}. Offsets must be ordered by
// increasing flat index in the halo-extended logical indexer.
//
// This ordering lets the bovi implementation build the halo range with a
// single linear merge pass: each candidate span only needs to be compared with
// the last emitted span. In other words, halo construction stays device-friendly
// and avoids sorting or dynamic storage before every inner loop.
struct none_t {
  static constexpr int npoints = 1;
  KOKKOS_INLINE_FUNCTION static constexpr int dk(int) { return 0; }
  KOKKOS_INLINE_FUNCTION static constexpr int dj(int) { return 0; }
  KOKKOS_INLINE_FUNCTION static constexpr int di(int) { return 0; }
};

struct plus_j_t {
  static constexpr int npoints = 2;
  // Sorted by flat offset: identity, then +j.
  KOKKOS_INLINE_FUNCTION static constexpr int dk(int) { return 0; }
  KOKKOS_INLINE_FUNCTION static constexpr int dj(int n) { return n==0 ? 0 : 1; }
  KOKKOS_INLINE_FUNCTION static constexpr int di(int) { return 0; }
};

struct minus_i_t {
  static constexpr int npoints = 2;
  KOKKOS_INLINE_FUNCTION static constexpr int dk(int) { return 0; }
  KOKKOS_INLINE_FUNCTION static constexpr int dj(int) { return 0; }
  KOKKOS_INLINE_FUNCTION static constexpr int di(int n) { return -1 * (n == 0); }
};

struct minus_j_t {
  static constexpr int npoints = 2;
  KOKKOS_INLINE_FUNCTION static constexpr int dk(int) { return 0; }
  KOKKOS_INLINE_FUNCTION static constexpr int dj(int n) { return -1 * (n == 0); }
  KOKKOS_INLINE_FUNCTION static constexpr int di(int) { return 0; }
};

struct minus_k_t {
  static constexpr int npoints = 2;
  KOKKOS_INLINE_FUNCTION static constexpr int dk(int n) { return -1 * (n == 0); }
  KOKKOS_INLINE_FUNCTION static constexpr int dj(int) { return 0; }
  KOKKOS_INLINE_FUNCTION static constexpr int di(int) { return 0; }
};

}

enum class loop_backend { raw, kokkos };

inline constexpr loop_backend default_loop_backend_v =
    std::is_same_v<parthenon::DevExecSpace, parthenon::HostExecSpace>
        ? loop_backend::raw
        : loop_backend::kokkos;

namespace impl {
template <class IndexSpaceType>
KOKKOS_INLINE_FUNCTION int GetNOuter(const IndexSpaceType &idx_space) {
  return idx_space.GetNOuter();
}

template <class>
struct ExplicitUnaryIntCall : std::false_type {};

template <class R, class C, class Arg>
struct ExplicitUnaryIntCall<R (C::*)(Arg)>
    : std::is_same<std::remove_cvref_t<Arg>, int> {};

template <class R, class C, class Arg>
struct ExplicitUnaryIntCall<R (C::*)(Arg) const>
    : std::is_same<std::remove_cvref_t<Arg>, int> {};

template <class F, class = void>
struct HasExplicitUnaryIntCall : std::false_type {};

template <class F>
struct HasExplicitUnaryIntCall<
    F, std::void_t<decltype(&std::remove_reference_t<F>::operator())>>
    : ExplicitUnaryIntCall<decltype(&std::remove_reference_t<F>::operator())> {};

template <class F>
inline constexpr bool has_explicit_unary_int_call_v =
    HasExplicitUnaryIntCall<F>::value;

constexpr bool HaloOffsetLess(const int dk0, const int dj0, const int di0,
                              const int dk1, const int dj1, const int di1) {
  if (dk0 != dk1) return dk0 < dk1;
  if (dj0 != dj1) return dj0 < dj1;
  return di0 < di1;
}

template <class Halo>
constexpr bool HaloHasUniqueIdentity() {
  if constexpr (Halo::npoints <= 0) {
    return false;
  } else {
    int count = 0;
    for (int n = 0; n < Halo::npoints; ++n) {
      if (Halo::dk(n) == 0 && Halo::dj(n) == 0 && Halo::di(n) == 0) {
        ++count;
      }
    }
    return count == 1;
  }
}

template <class Halo>
constexpr bool HaloOffsetsAreStrictlySorted() {
  if constexpr (Halo::npoints <= 0) {
    return false;
  } else {
    for (int n = 1; n < Halo::npoints; ++n) {
      if (!HaloOffsetLess(Halo::dk(n - 1), Halo::dj(n - 1), Halo::di(n - 1),
                          Halo::dk(n), Halo::dj(n), Halo::di(n))) {
        return false;
      }
    }
    return true;
  }
}

template <class Halo>
constexpr bool HaloSatisfiesContract() {
  return HaloHasUniqueIdentity<Halo>() && HaloOffsetsAreStrictlySorted<Halo>();
}

} // namespace impl

enum class loop_tag { bvoi, bovi, boiv };
enum class inner_tag { logical_flat, logical_coords, memory };

struct Index3 {
  int k, j, i;
  KOKKOS_INLINE_FUNCTION
  constexpr Index3() = default;

  KOKKOS_INLINE_FUNCTION
  constexpr Index3(int k_, int j_, int i_)
      : k(k_), j(j_), i(i_) {}

  KOKKOS_INLINE_FUNCTION
  constexpr Index3(const std::tuple<int, int, int> &t)
      : k(std::get<0>(t)), j(std::get<1>(t)), i(std::get<2>(t)) {}
};

KOKKOS_INLINE_FUNCTION
constexpr Index3 operator+(Index3 a, Index3 b) {
  return {a.k + b.k, a.j + b.j, a.i + b.i};
}

KOKKOS_INLINE_FUNCTION
constexpr Index3 operator-(Index3 a, Index3 b) {
  return {a.k - b.k, a.j - b.j, a.i - b.i};
}

KOKKOS_INLINE_FUNCTION
constexpr Index3 operator-(Index3 a) { return {-a.k, -a.j, -a.i}; }

KOKKOS_INLINE_FUNCTION
constexpr Index3 operator*(int n, Index3 a) { return {n * a.k, n * a.j, n * a.i}; }

KOKKOS_INLINE_FUNCTION
constexpr Index3 operator*(Index3 a, int n) { return n * a; }

struct MemoryOffset {
  int dk = 0;
  int dj = 0;
  int di = 0;
  int flat = 0;

  KOKKOS_INLINE_FUNCTION constexpr operator int() const { return flat; }
};

KOKKOS_INLINE_FUNCTION
constexpr MemoryOffset operator+(MemoryOffset a, MemoryOffset b) {
  return {a.dk + b.dk, a.dj + b.dj, a.di + b.di, a.flat + b.flat};
}

KOKKOS_INLINE_FUNCTION
constexpr MemoryOffset operator-(MemoryOffset a, MemoryOffset b) {
  return {a.dk - b.dk, a.dj - b.dj, a.di - b.di, a.flat - b.flat};
}

KOKKOS_INLINE_FUNCTION
constexpr MemoryOffset operator-(MemoryOffset a) {
  return {-a.dk, -a.dj, -a.di, -a.flat};
}

KOKKOS_INLINE_FUNCTION
constexpr MemoryOffset operator*(int n, MemoryOffset a) {
  return {n * a.dk, n * a.dj, n * a.di, n * a.flat};
}

KOKKOS_INLINE_FUNCTION
constexpr MemoryOffset operator*(MemoryOffset a, int n) { return n * a; }

template <class Halo>
inline auto AddHaloToIndexer(const parthenon::Indexer3D &idxer) { 
  std::array<int, 3> extend_low{0, 0, 0}, extend_up{0, 0, 0};
  for (int p = 0; p < Halo::npoints; ++p) { 
    extend_low[0] = std::max(extend_low[0], -Halo::dk(p)); 
    extend_low[1] = std::max(extend_low[1], -Halo::dj(p)); 
    extend_low[2] = std::max(extend_low[2], -Halo::di(p)); 

    extend_up[0] = std::max(extend_up[0], Halo::dk(p)); 
    extend_up[1] = std::max(extend_up[1], Halo::dj(p)); 
    extend_up[2] = std::max(extend_up[2], Halo::di(p)); 
  }

  return parthenon::Indexer3D({idxer.template StartIdx<0>() - extend_low[0], idxer.template EndIdx<0>() + extend_up[0]},
                   {idxer.template StartIdx<1>() - extend_low[1], idxer.template EndIdx<1>() + extend_up[1]},
                   {idxer.template StartIdx<2>() - extend_low[2], idxer.template EndIdx<2>() + extend_up[2]});
} 

template <class T, class Halo, std::size_t... Dims, class IndexSpaceType>
std::size_t GetPerTeamScratchSize(const IndexSpaceType &idx_space);

template <loop_tag LOOP_TAG, inner_tag INNER_TAG,
          loop_backend BACKEND = default_loop_backend_v>
class IndexSpace {
  static_assert(!(LOOP_TAG == loop_tag::boiv && INNER_TAG == inner_tag::memory),
                "IndexSpace: This tag combination is not supported and will not be.");
 public:
  static constexpr loop_tag loop_tag_v = LOOP_TAG;
  static constexpr inner_tag inner_tag_v = INNER_TAG;
  static constexpr loop_backend backend_v = BACKEND;

  KOKKOS_INLINE_FUNCTION int GetMemoryOffset(const int dk, const int dj,
                                             const int di) const {
    const int nj =
        memory_kji.template EndIdx<1>() - memory_kji.template StartIdx<1>() + 1;
    const int ni =
        memory_kji.template EndIdx<2>() - memory_kji.template StartIdx<2>() + 1;
    return dk * nj * ni + dj * ni + di;
  }

  KOKKOS_INLINE_FUNCTION MemoryOffset GetMemoryOffsetIndex(const int dk, const int dj,
                                                           const int di) const {
    return {dk, dj, di, GetMemoryOffset(dk, dj, di)};
  }

  auto GetDelta(parthenon::CoordinateDirection dir) {
    const int nk =
        memory_kji.template EndIdx<0>() - memory_kji.template StartIdx<0>() + 1;
    const int nj =
        memory_kji.template EndIdx<1>() - memory_kji.template StartIdx<1>() + 1;
    const int ni =
        memory_kji.template EndIdx<2>() - memory_kji.template StartIdx<2>() + 1;
    if constexpr (inner_tag_v == inner_tag::logical_coords) {
      if (dir == parthenon::X1DIR) return Index3{0, 0, 1};
      if (dir == parthenon::X2DIR) return Index3{0, nj > 1, 0};
      if (dir == parthenon::X3DIR) return Index3{nk > 1, 0, 0};
      return Index3{0, 0, 0};
    } else {
      if (dir == parthenon::X1DIR) return GetMemoryOffsetIndex(0, 0, 1);
      if (dir == parthenon::X2DIR)
        return nj > 1 ? GetMemoryOffsetIndex(0, 1, 0) : MemoryOffset{};
      if (dir == parthenon::X3DIR)
        return nk > 1 ? GetMemoryOffsetIndex(1, 0, 0) : MemoryOffset{};
      return MemoryOffset{};
    }
  }

  IndexSpace(int nblocks, int nx, int ny, int nz, int nghost,
             std::optional<int> ninner = std::nullopt)
      : nblocks(nblocks), ninner(ninner.value_or(nx * ny)) {
    logical_kji = parthenon::Indexer3D(
        {nghost, nghost + nz - 1}, {nghost, nghost + ny - 1}, {nghost, nghost + nx - 1});
    memory_kji = parthenon::Indexer3D({0, 2 * nghost + nz - 1}, {0, 2 * nghost + ny - 1},
                                      {0, 2 * nghost + nx - 1});
  }

  using ID = parthenon::IndexDomain;
  using TE = parthenon::TopologicalElement;
  IndexSpace(int ninner, ID domain, int halo, int nblocks,
             const parthenon::MeshData<parthenon::Real> *md, TE domain_te,
             TE memory_te = TE::CC)
      : nblocks(nblocks), ninner(ninner),
        memory_kji(md->GetBoundsK(ID::entire, memory_te),
                   md->GetBoundsJ(ID::entire, memory_te),
                   md->GetBoundsI(ID::entire, memory_te)) {
    auto ib = md->GetBoundsI(domain, domain_te);
    auto jb = md->GetBoundsJ(domain, domain_te);
    auto kb = md->GetBoundsK(domain, domain_te);
    if (md->GetMeshPointer()) {
      const int ndim = md->GetMeshPointer()->ndim;
      if (ndim > 0) {
        ib.s -= halo;
        ib.e += halo;
      }
      if (ndim > 1) {
        jb.s -= halo;
        jb.e += halo;
      }
      if (ndim > 2) {
        kb.s -= halo;
        kb.e += halo;
      }
    } else if (halo != 0) {
      PARTHENON_FAIL(
          "Asking for a halo with no mesh object. No way to determine dimension.");
    }
    logical_kji = parthenon::Indexer3D({kb.s, kb.e}, {jb.s, jb.e}, {ib.s, ib.e});
    PARTHENON_REQUIRE(memory_te == TE::CC || memory_te == TE::NN,
                      "Only two kinds of memory layouts for topological elements.");
  }

  KOKKOS_INLINE_FUNCTION int GetNOuter() const {
    return logical_kji.size() / ninner + (logical_kji.size() % ninner != 0);
  }

  KOKKOS_INLINE_FUNCTION const parthenon::Indexer3D &GetLogicalIndexer() const {
    return logical_kji;
  }

  KOKKOS_INLINE_FUNCTION const parthenon::Indexer3D &GetMemoryIndexer() const {
    return memory_kji;
  }

  KOKKOS_INLINE_FUNCTION int GetNBlocks() const { return nblocks; }

  KOKKOS_INLINE_FUNCTION int GetNInner() const { return ninner; }

  template <class T, class Halo = halo::none_t>
  void AddPerPointScratch(std::size_t count = 1) {
    per_team_scratch_size_in_bytes += count * GetPerTeamScratchSize<T, Halo>(*this);
  }

  template <class T, std::size_t... Dims>
    requires(sizeof...(Dims) > 0)
  void AddPerPointScratch(std::size_t count = 1) {
    per_team_scratch_size_in_bytes += count * GetPerTeamScratchSize<T, Dims...>(*this);
  }

  template <class T, class Halo, std::size_t... Dims>
    requires(sizeof...(Dims) > 0)
  void AddPerPointScratch(std::size_t count = 1) {
    per_team_scratch_size_in_bytes +=
        count * GetPerTeamScratchSize<T, Halo, Dims...>(*this);
  }

  std::size_t GetPerTeamScratchSizeInBytes() const {
    return per_team_scratch_size_in_bytes;
  }

 private:
  parthenon::Indexer3D logical_kji, memory_kji;
  int nblocks;
  int ninner;
  std::size_t per_team_scratch_size_in_bytes = 0;
};



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
                  const parthenon::Indexer3D &logical_kji_in,
                  int b,
                  const device_team_member_t *team_member_in = nullptr)
    : pidx_space(&idx_space),
      logical_kji(logical_kji_in),
      block(b),
      ks(logical_kji.template StartIdx<0>()),
      js(logical_kji.template StartIdx<1>()),
      is(logical_kji.template StartIdx<2>()),
      team_member(team_member_in) {
    const Index3 start{
      logical_kji.template StartIdx<0>(),
      logical_kji.template StartIdx<1>(),
      logical_kji.template StartIdx<2>()};

    const Index3 end{
        logical_kji.template EndIdx<0>(),
        logical_kji.template EndIdx<1>(),
        logical_kji.template EndIdx<2>()};

    BuildRegionsFromEndpoints(start, end);
  }
  
  // Constructor relevant for bovi
  KOKKOS_INLINE_FUNCTION
  InnerIndexRange(const IndexSpaceType &idx_space,
                  const parthenon::Indexer3D &logical_kji_in,
                  int b, Index3 start, Index3 end,
                  const device_team_member_t *team_member_in = nullptr)
    : pidx_space(&idx_space),
      logical_kji(logical_kji_in),
      block(b),
      ks(start.k),
      js(start.j),
      is(start.i),
      team_member(team_member_in) {
    BuildRegionsFromEndpoints(start, end);
  }

  KOKKOS_INLINE_FUNCTION
  InnerIndexRange(const IndexSpaceType &idx_space,
                  const parthenon::Indexer3D &logical_kji_in,
                  int b, int flat_start, int flat_end,
                  const device_team_member_t *team_member_in = nullptr)
    : pidx_space(&idx_space),
      logical_kji(logical_kji_in),
      block(b),
      team_member(team_member_in) {
    const auto [ks_, js_, is_] = logical_kji(flat_start);
    ks = ks_;
    js = js_;
    is = is_;
    BuildRegionsFromEndpoints({ks, js, is}, logical_kji(flat_end));
  }


  template <class Halo_in>
  KOKKOS_INLINE_FUNCTION
  InnerIndexRange<IndexSpaceType, Halo_in> AddHalo() const {
    static_assert(std::is_same_v<Halo, halo::none_t>, "Halo composition is currently not supported.");
    parthenon::Indexer3D halo_kji = AddHaloToIndexer<Halo_in>(logical_kji);
    const auto [ke, je, ie] = GetKJIFromFlatIdx(flat_end[0]);
    return InnerIndexRange<IndexSpaceType, Halo_in>(*pidx_space, halo_kji, block,
                                                    {ks, js, is}, {ke, je, ie},
                                                    team_member);
  }

  KOKKOS_INLINE_FUNCTION void BuildRegionsFromEndpoints(const Index3 start, const Index3 end) {
    flat_start[0] = GetFlatIdxFromKJI(start.k + Halo::dk(0), start.j + Halo::dj(0), start.i + Halo::di(0));
    flat_end[0]   = GetFlatIdxFromKJI(end.k + Halo::dk(0), end.j + Halo::dj(0), end.i + Halo::di(0));
    const int memory_base =
        pidx_space->GetMemoryIndexer().GetFlatIdx(start.k, start.j, start.i);
    scratch_flat_start = pidx_space->GetMemoryIndexer().GetFlatIdx(
        start.k + Halo::dk(0), start.j + Halo::dj(0), start.i + Halo::di(0));
    int scratch_flat_end = pidx_space->GetMemoryIndexer().GetFlatIdx(
        end.k + Halo::dk(0), end.j + Halo::dj(0), end.i + Halo::di(0));
    nregions = 1;
    // Create possibly disjoint ranges, this algorithm relies on the start and end points of the ranges 
    // being sorted by flat start
    for (int n = 1; n < Halo::npoints; ++n) {
      const int fstart = GetFlatIdxFromKJI(start.k + Halo::dk(n), start.j + Halo::dj(n), start.i + Halo::di(n));
      const int fend   = GetFlatIdxFromKJI(end.k + Halo::dk(n), end.j + Halo::dj(n), end.i + Halo::di(n));
      const int scratch_start = pidx_space->GetMemoryIndexer().GetFlatIdx(
          start.k + Halo::dk(n), start.j + Halo::dj(n), start.i + Halo::di(n));
      const int scratch_end = pidx_space->GetMemoryIndexer().GetFlatIdx(
          end.k + Halo::dk(n), end.j + Halo::dj(n), end.i + Halo::di(n));
      scratch_flat_start = std::min(scratch_flat_start, scratch_start);
      scratch_flat_end = std::max(scratch_flat_end, scratch_end);
      if (fstart <= flat_end[nregions - 1] + 1) {
        if (fend > flat_end[nregions - 1])
          flat_end[nregions - 1] = fend;  
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
  int size() const {
    return cached_size;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  int ScratchSize() const {
    return scratch_span_size;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  int ScratchIndex(int mem_idx) const {
    return mem_idx - scratch_index_start;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  int ScratchIndex(MemoryOffset idx) const {
    return ScratchIndex(idx.flat);
  }

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
  int size() const {
    return halo_t::npoints;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  void TeamBarrier() const {}
};



} // namespace loop_abstraction
