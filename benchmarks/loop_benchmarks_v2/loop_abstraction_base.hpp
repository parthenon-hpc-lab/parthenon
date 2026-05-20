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

template <class IndexSpaceType, class F>
KOKKOS_INLINE_FUNCTION void outer_raw_for(IndexSpaceType idx_space, F &&f);

template <class InnerIndexRangeType, class F>
KOKKOS_FORCEINLINE_FUNCTION void inner_raw_for(const InnerIndexRangeType &idx_range, F &&f);

template <class IndexSpaceType, class F>
void outer_kokkos(IndexSpaceType idx_space, F &&f);

template <class InnerIndexRangeType, class F>
KOKKOS_FORCEINLINE_FUNCTION void inner_kokkos(const InnerIndexRangeType &idx_range, F &&f);

template <class IndexSpaceType>
KOKKOS_INLINE_FUNCTION int GetNOuter(const IndexSpaceType &idx_space) {
  return idx_space.GetNOuter();
}

inline constexpr bool use_raw_for_v =
    std::is_same_v<parthenon::DevExecSpace, parthenon::HostExecSpace>;

} // namespace impl

template <class IndexSpaceType, class F>
void outer(IndexSpaceType idx_space, F &&f);

template <class InnerIndexRangeType, class F>
KOKKOS_FORCEINLINE_FUNCTION void inner(const InnerIndexRangeType &idx_range, F &&f);

template <class IndexSpace>
class InnerIndexRange;

template <class IndexSpaceType>
struct var_view_t;

enum class loop_tag { bvoi, bovi, boiv };
enum class inner_tag { logical, memory };

template <loop_tag LOOP_TAG, inner_tag INNER_TAG>
class IndexSpace;

template <loop_tag LOOP_TAG>
struct inner_index_range_payload_t;

template <loop_tag LOOP_TAG>
struct inner_index_range_payload_t {
  int flat_start = 0;
  int flat_end = -1;
  int ks, js, is;
  const device_team_member_t *team_member = nullptr;
};

template <>
struct inner_index_range_payload_t<loop_tag::boiv> {
  int k = 0;
  int j = 0;
  int i = 0;
};

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
  using payload_t = inner_index_range_payload_t<IndexSpaceType::loop_tag_v>;

  const IndexSpaceType *pidx_space = nullptr;
  int block = 0;
  payload_t payload_{};
};

} // namespace loop_abstraction

} // namespace plb2
