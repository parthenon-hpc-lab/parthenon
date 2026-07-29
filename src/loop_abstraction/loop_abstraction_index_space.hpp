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
#ifndef LOOP_ABSTRACTION_LOOP_ABSTRACTION_INDEX_SPACE_HPP_
#define LOOP_ABSTRACTION_LOOP_ABSTRACTION_INDEX_SPACE_HPP_

// This file was made in part with generative AI.

// The chunk-shape descriptor (NInner) and the central IndexSpace class. An
// IndexSpace fixes the loop shape, inner traversal, and backend at compile time and
// carries the logical and memory indexers plus per-team scratch bookkeeping for a
// block. It is the object passed into outer(...). The definition of
// GetPerTeamScratchSize lives in the scratch header; only a forward declaration is
// needed here so IndexSpace::AddPerPointScratch can call it.

#include <optional>

#include "basic_types.hpp"
#include "interface/mesh_data.hpp"
#include "mesh/mesh.hpp"
#include "utils/indexer.hpp"

#include "loop_abstraction/loop_abstraction_halo.hpp"
#include "loop_abstraction/loop_abstraction_types.hpp"

namespace parthenon::loop_abstraction {

// Expressive inner-chunk shapes for the memory-tag chunking. Rather than a bare
// cell count (which the caller must compute from the logical dimensions before the
// halo is known, and which then misaligns once a halo is added), a chunk_shape is
// resolved to a cell count against whichever indexer is actually being chunked --
// including the halo-extended indexer in the bvoi/memory path -- so chunks land on
// clean row/plane boundaries by construction.
enum class chunk_shape {
  i_pencil, // one (extended) i-row per chunk
  ij_slab,  // one (extended) ij-plane per chunk
  kji_cube, // the whole (extended) block in one chunk
};

// Chunk-size descriptor: either an explicit cell count or a chunk_shape resolved
// lazily against a given indexer. Implicitly constructible from int so existing
// call sites that pass a raw count keep working unchanged.
class NInner {
 public:
  KOKKOS_FUNCTION NInner() : explicit_(true), cells_(0), shape_(chunk_shape::ij_slab) {}
  KOKKOS_FUNCTION NInner(int cells)
      : explicit_(true), cells_(cells), shape_(chunk_shape::ij_slab) {}
  KOKKOS_FUNCTION NInner(chunk_shape shape)
      : explicit_(false), cells_(0), shape_(shape) {}

  KOKKOS_INLINE_FUNCTION int resolve(const parthenon::Indexer3D &idxer) const {
    if (explicit_) return cells_;
    const int nk = idxer.template EndIdx<0>() - idxer.template StartIdx<0>() + 1;
    const int nj = idxer.template EndIdx<1>() - idxer.template StartIdx<1>() + 1;
    const int ni = idxer.template EndIdx<2>() - idxer.template StartIdx<2>() + 1;
    switch (shape_) {
    case chunk_shape::i_pencil:
      return ni;
    case chunk_shape::ij_slab:
      return ni * nj;
    case chunk_shape::kji_cube:
      return ni * nj * nk;
    }
    return ni * nj;
  }

 private:
  bool explicit_;
  int cells_ = 0;
  chunk_shape shape_ = chunk_shape::ij_slab;
};

namespace impl {
template <class IndexSpaceType>
KOKKOS_INLINE_FUNCTION int GetNOuter(const IndexSpaceType &idx_space) {
  return idx_space.GetNOuter();
}
} // namespace impl

// Forward declarations
template <class T, class Halo, std::size_t... Dims, class IndexSpaceType>
std::size_t GetPerTeamScratchSize(const IndexSpaceType &idx_space);
template <class IndexSpaceType, class Halo>
class InnerIndexRange;

template <loop_tag LOOP_TAG, inner_tag INNER_TAG,
          loop_backend BACKEND = default_loop_backend_v>
class IndexSpace {
  static_assert(!(LOOP_TAG == loop_tag::boiv && INNER_TAG == inner_tag::memory),
                "IndexSpace: This tag combination is not supported and will not be.");

 public:
  static constexpr loop_tag loop_tag_v = LOOP_TAG;
  static constexpr inner_tag inner_tag_v = INNER_TAG;
  static constexpr loop_backend backend_v = BACKEND;

  // The (base, no-halo) inner range that outer() hands to a loop body. Naming it lets
  // an outer body spell its parameter type without `auto` (which nvcc rejects for
  // extended lambdas): outer(idx_space, KOKKOS_LAMBDA(const IST::idx_range_t &r, ...)).
  using idx_range_t = InnerIndexRange<IndexSpace, halo::none_t>;

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
             std::optional<NInner> ninner = std::nullopt)
      : nblocks(nblocks), ninner(ninner.value_or(NInner(chunk_shape::ij_slab))) {
    logical_kji = parthenon::Indexer3D(
        {nghost, nghost + nz - 1}, {nghost, nghost + ny - 1}, {nghost, nghost + nx - 1});
    memory_kji = parthenon::Indexer3D({0, 2 * nghost + nz - 1}, {0, 2 * nghost + ny - 1},
                                      {0, 2 * nghost + nx - 1});
  }

  using ID = parthenon::IndexDomain;
  using TE = parthenon::TopologicalElement;
  template <class MeshDataOrMeshBlockData>
  IndexSpace(NInner ninner, ID domain, int halo, int nblocks,
             const MeshDataOrMeshBlockData *md, TE domain_te, TE memory_te = TE::CC)
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
    const int ni = GetNInner();
    KOKKOS_ASSERT(ni > 0 && "IndexSpace: inner chunk size must be positive.");
    return logical_kji.size() / ni + (logical_kji.size() % ni != 0);
  }

  KOKKOS_INLINE_FUNCTION const parthenon::Indexer3D &GetLogicalIndexer() const {
    return logical_kji;
  }

  KOKKOS_INLINE_FUNCTION const parthenon::Indexer3D &GetMemoryIndexer() const {
    return memory_kji;
  }

  KOKKOS_INLINE_FUNCTION int GetNBlocks() const { return nblocks; }

  // Default: resolve the chunk shape against the base logical indexer. Used by the
  // logical-tag paths and scratch sizing, which chunk the un-extended space.
  KOKKOS_INLINE_FUNCTION int GetNInner() const { return ninner.resolve(logical_kji); }

  // Resolve against an explicit indexer -- e.g. the halo-extended indexer that the
  // bvoi/memory path chunks, so an ij_slab means one *extended* plane.
  KOKKOS_INLINE_FUNCTION int GetNInner(const parthenon::Indexer3D &idxer) const {
    return ninner.resolve(idxer);
  }

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
  NInner ninner;
  std::size_t per_team_scratch_size_in_bytes = 0;
};

} // namespace parthenon::loop_abstraction

#endif // LOOP_ABSTRACTION_LOOP_ABSTRACTION_INDEX_SPACE_HPP_
