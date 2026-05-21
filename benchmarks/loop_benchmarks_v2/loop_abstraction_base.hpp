#pragma once

#include <algorithm>
#include <array>
#include <concepts>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

#include <Kokkos_Core.hpp>

#include "basic_types.hpp"
#include "interface/mesh_data.hpp"
#include "kokkos_types.hpp"
#include "mesh/mesh.hpp"
#include "utils/indexer.hpp"

namespace plb2 {

namespace loop_abstraction {

using device_team_member_t =
    typename Kokkos::TeamPolicy<parthenon::DevExecSpace>::member_type;

namespace impl {
template <class IndexSpaceType>
KOKKOS_INLINE_FUNCTION int GetNOuter(const IndexSpaceType &idx_space) {
  return idx_space.GetNOuter();
}

inline constexpr bool use_raw_for_v =
    std::is_same_v<parthenon::DevExecSpace, parthenon::HostExecSpace>;

} // namespace impl

enum class loop_tag { bvoi, bovi, boiv };
enum class inner_tag { logical_flat, logical_coords, memory };

struct Index3 {
  int k, j, i;
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

template <loop_tag LOOP_TAG, inner_tag INNER_TAG>
class IndexSpace {
 public:
  static constexpr loop_tag loop_tag_v = LOOP_TAG;
  static constexpr inner_tag inner_tag_v = INNER_TAG;

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
      if (dir == parthenon::X1DIR) return 1;
      if (dir == parthenon::X2DIR) return nj > 1 ? ni : 0;
      if (dir == parthenon::X3DIR) return nk > 1 ? ni * nj : 0;
      return 0;
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

 private:
  parthenon::Indexer3D logical_kji, memory_kji;
  int nblocks;
  int ninner;
};

template <class IndexSpaceType>
class InnerIndexRange {
 public:
  const IndexSpaceType *pidx_space = nullptr;
  int block = 0;
  int flat_start = 0;
  int flat_end = -1;
  int ks = 0;
  int js = 0;
  int is = 0;
  const device_team_member_t *team_member = nullptr;

  KOKKOS_INLINE_FUNCTION std::tuple<int, int, int> GetKJI(int idx) const {
    const int shift = pidx_space->GetMemoryIndexer().GetFlatIdx(ks, js, is);
    return pidx_space->GetMemoryIndexer()(idx + shift);
  }
  KOKKOS_INLINE_FUNCTION std::tuple<int, int, int> GetKJI(Index3 idx) const {
    return {idx.k, idx.j, idx.i};
  }
};

template <inner_tag INNER_TAG>
class InnerIndexRange<IndexSpace<loop_tag::boiv, INNER_TAG>> {
 public:
  const IndexSpace<loop_tag::boiv, INNER_TAG> *pidx_space = nullptr;
  int block = 0;
  int ks = 0;
  int js = 0;
  int is = 0;

  KOKKOS_INLINE_FUNCTION std::tuple<int, int, int> GetKJI(int idx) const {
    const int shift = pidx_space->GetMemoryIndexer().GetFlatIdx(ks, js, is);
    return pidx_space->GetMemoryIndexer()(idx + shift);
  }

  KOKKOS_INLINE_FUNCTION std::tuple<int, int, int> GetKJI(Index3 idx) const {
    return {idx.k, idx.j, idx.i};
  }
};

} // namespace loop_abstraction

} // namespace plb2
