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

template <class IndexSpaceType>
struct var_view_t {
 public:
  parthenon::Real *data = nullptr;
  int shift;
  const IndexSpaceType *pidx_space = nullptr;

  KOKKOS_FUNCTION
  parthenon::Real &operator()(int idx) const { return data[idx + shift]; }

  KOKKOS_FUNCTION
  parthenon::Real &operator()(Index3 in) const {
    return data[pidx_space->memory_kji.GetFlatIdx(in.k, in.j, in.i) + shift];
  }
};

template <>
struct var_view_t<IndexSpace<loop_tag::boiv, inner_tag::logical>> {
 public:
  parthenon::Real *data = nullptr;

  KOKKOS_FUNCTION
  parthenon::Real &operator()(Index3 in) const {
    (void)in;
    return *data;
  }

  KOKKOS_FUNCTION
  parthenon::Real &operator()(int idx) const {
    (void)idx;
    return *data;
  }
};

template <loop_tag LOOP_TAG, inner_tag INNER_TAG>
class IndexSpace {
 public:
  template <class IndexSpaceType, class F>
  KOKKOS_INLINE_FUNCTION friend void impl::outer_raw_for(IndexSpaceType idx_space, F &&f);
  template <class InnerIndexRangeType, class F>
  KOKKOS_FORCEINLINE_FUNCTION
  friend void impl::inner_raw_for(const InnerIndexRangeType &idx_range, F &&f);
  template <class IndexSpaceType, class F>
  friend void impl::outer_kokkos(IndexSpaceType idx_space, F &&f);
  template <class InnerIndexRangeType, class F>
  KOKKOS_FORCEINLINE_FUNCTION
  friend void impl::inner_kokkos(const InnerIndexRangeType &idx_range, F &&f);
  template <class>
  friend struct var_view_t;
  template <class>
  friend class InnerIndexRange;

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

  template <class ViewType>
  KOKKOS_INLINE_FUNCTION auto GetInnerView(ViewType &in, int block, int var,
                                           std::array<int, 3> offset = {0, 0, 0}) const {
    return var_view_t<IndexSpace>{&in(block, var, 0, 0, 0),
                                  static_cast<int>(
                                      memory_kji.GetFlatIdx(offset[0], offset[1], offset[2])),
                                  this};
  }

  KOKKOS_INLINE_FUNCTION int GetNOuter() const {
    return logical_kji.size() / ninner + (logical_kji.size() % ninner != 0);
  }

 private:
  parthenon::Indexer3D logical_kji, memory_kji;
  int nblocks;
  int ninner;
};

template <class IndexSpaceType>
class InnerIndexRange {
 public:
  template <class ViewType>
  KOKKOS_INLINE_FUNCTION auto view(ViewType &in, int var,
                                   std::array<int, 3> offset = {0, 0, 0}) const {
    if constexpr (IndexSpaceType::loop_tag_v == loop_tag::boiv) {
      static_assert(IndexSpaceType::inner_tag_v == inner_tag::logical,
                    "boiv currently expects logical inner coordinates");
      return var_view_t<IndexSpaceType>{&in(block, var, payload_.k + offset[0],
                                            payload_.j + offset[1], payload_.i + offset[2])};
    } else if constexpr (IndexSpaceType::loop_tag_v == loop_tag::bovi &&
                         IndexSpaceType::inner_tag_v == inner_tag::memory) {
      return var_view_t<IndexSpaceType>{&in(block, var, payload_.ks + offset[0],
                                            payload_.js + offset[1], payload_.is + offset[2]),
                                        0, pidx_space};
    } else {
      return pidx_space->GetInnerView(in, block, var, offset);
    }
  }

 private:
  using payload_t = inner_index_range_payload_t<IndexSpaceType::loop_tag_v>;

  template <class AnyIndexSpaceType, class F>
  KOKKOS_INLINE_FUNCTION friend void impl::outer_raw_for(AnyIndexSpaceType idx_space, F &&f);
  template <class InnerIndexRangeType, class F>
  KOKKOS_FORCEINLINE_FUNCTION
  friend void impl::inner_raw_for(const InnerIndexRangeType &idx_range, F &&f);
  template <class AnyIndexSpaceType, class F>
  friend void impl::outer_kokkos(AnyIndexSpaceType idx_space, F &&f);
  template <class InnerIndexRangeType, class F>
  KOKKOS_FORCEINLINE_FUNCTION
  friend void impl::inner_kokkos(const InnerIndexRangeType &idx_range, F &&f);

  KOKKOS_FUNCTION static InnerIndexRange FlatRange(
      const IndexSpaceType &idx_space, int b, int logical_start, int logical_end,
      const device_team_member_t *team_member = nullptr) {
    InnerIndexRange out;
    out.pidx_space = &idx_space;
    const auto [ks, js, is] = idx_space.logical_kji(logical_start);
    out.block = b;
    const auto [ke, je, ie] = idx_space.logical_kji(logical_end);
    if constexpr (IndexSpaceType::inner_tag_v == inner_tag::memory) {
      out.payload_.flat_start = idx_space.memory_kji.GetFlatIdx(ks, js, is);
      out.payload_.flat_end = idx_space.memory_kji.GetFlatIdx(ke, je, ie);
    } else if constexpr (IndexSpaceType::inner_tag_v == inner_tag::logical) {
      out.payload_.flat_start = logical_start;
      out.payload_.flat_end = logical_end;
    }
    if constexpr (IndexSpaceType::loop_tag_v != loop_tag::boiv) {
      out.payload_.team_member = team_member;
      out.payload_.ks = ks;
      out.payload_.js = js;
      out.payload_.is = is;
    }
    return out;
  }

 public:
  const IndexSpaceType *pidx_space = nullptr;
  int block = 0;
  payload_t payload_{};
};

} // namespace loop_abstraction

} // namespace plb2
