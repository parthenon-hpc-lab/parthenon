#pragma once

#include <algorithm>
#include <array>
#include <concepts>
#include <optional>
#include <type_traits>
#include <utility>

#include <Kokkos_Core.hpp>

#include "basic_types.hpp"
#include "kokkos_types.hpp"
#include "utils/indexer.hpp"

namespace plb2 {

namespace loop_abstraction {

using device_team_member_t = typename Kokkos::TeamPolicy<parthenon::DevExecSpace>::member_type;

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

template <loop_tag LOOP_TAG, inner_tag INNER_TAG>
class IndexSpace {
 public:
  static constexpr loop_tag loop_tag_v = LOOP_TAG;
  static constexpr inner_tag inner_tag_v = INNER_TAG;

  IndexSpace(int nblocks, int nx, int ny, int nz, int nghost,
             std::optional<int> ninner = std::nullopt)
      : nblocks(nblocks), ninner(ninner.value_or(nx * ny)) {
    logical_kji = parthenon::Indexer3D({nghost, nghost + nz - 1}, {nghost, nghost + ny - 1},
                                       {nghost, nghost + nx - 1});
    memory_kji = parthenon::Indexer3D({0, 2 * nghost + nz - 1},
                                      {0, 2 * nghost + ny - 1},
                                      {0, 2 * nghost + nx - 1});
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
};

template <inner_tag INNER_TAG>
class InnerIndexRange<IndexSpace<loop_tag::boiv, INNER_TAG>> {
 public:
  const IndexSpace<loop_tag::boiv, INNER_TAG> *pidx_space = nullptr;
  int block = 0;
  int k = 0;
  int j = 0;
  int i = 0;
};

} // namespace loop_abstraction

} // namespace plb2
