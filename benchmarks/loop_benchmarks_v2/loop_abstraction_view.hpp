#pragma once

#include "pack/sparse_pack/sparse_pack.hpp"
#include "utils/type_list.hpp"

#include "loop_abstraction_base.hpp"

namespace plb2 {

namespace loop_abstraction {

namespace impl {
struct SumAllTypes {};

template <class TL, std::size_t... Is>
constexpr std::size_t SumSizesImpl(std::index_sequence<Is...>) {
  return (std::size_t{0} + ... + TL::template type<Is>::size());
}
} // namespace impl

template <class TL, class StopT = impl::SumAllTypes>
constexpr std::size_t SumSizesBefore() {
  if constexpr (std::is_same_v<StopT, impl::SumAllTypes>) {
    return impl::SumSizesImpl<TL>(std::make_index_sequence<TL::n_types>{});
  } else {
    constexpr std::size_t stop_idx = TL::template GetIdx<StopT>();
    return impl::SumSizesImpl<TL>(std::make_index_sequence<stop_idx>{});
  }
}

template <class IndexSpaceType>
struct var_view_t {
 public:
  parthenon::Real *data = nullptr;
  int flattened_offset = 0;
  const IndexSpaceType *pidx_space = nullptr;

  KOKKOS_FUNCTION
  parthenon::Real &operator()(int idx) const { return data[idx + flattened_offset]; }

  KOKKOS_FUNCTION
  parthenon::Real &operator()(Index3 in) const {
    return data[pidx_space->GetMemoryIndexer().GetFlatIdx(in.k, in.j, in.i) +
                flattened_offset];
  }

  KOKKOS_FUNCTION
  parthenon::Real &operator()(int k, int j, int i) const { return (*this)(Index3{k, j, i}); }
};

template <inner_tag INNER_TAG>
struct var_view_t<IndexSpace<loop_tag::bovi, INNER_TAG>> {
 public:
  parthenon::Real *data = nullptr;
  int shift = 0;
  const IndexSpace<loop_tag::bovi, INNER_TAG> *pidx_space = nullptr;

  KOKKOS_FUNCTION
  parthenon::Real &operator()(int idx) const { return data[idx]; }

  KOKKOS_FUNCTION
  parthenon::Real &operator()(Index3 in) const {
    return data[pidx_space->GetMemoryIndexer().GetFlatIdx(in.k, in.j, in.i) - shift];
  }

  KOKKOS_FUNCTION
  parthenon::Real &operator()(int k, int j, int i) const { return (*this)(Index3{k, j, i}); }
};

template <inner_tag INNER_TAG>
struct var_view_t<IndexSpace<loop_tag::boiv, INNER_TAG>> {
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

  KOKKOS_FUNCTION
  parthenon::Real &operator()(int k, int j, int i) const { return (*this)(Index3{k, j, i}); }
};

template <class IndexSpaceType, class ViewType>
KOKKOS_INLINE_FUNCTION auto GetView(const InnerIndexRange<IndexSpaceType> &idx_range,
                                    ViewType &in, int var,
                                    std::array<int, 3> offset = {0, 0, 0}) {
  if constexpr (IndexSpaceType::loop_tag_v == loop_tag::boiv) {
    static_assert(IndexSpaceType::inner_tag_v == inner_tag::logical_flat ||
                  IndexSpaceType::inner_tag_v == inner_tag::logical_coords,
                  "boiv currently expects logical inner coordinates");
    return var_view_t<IndexSpaceType>{
        &in(idx_range.block, var, idx_range.k + offset[0], idx_range.j + offset[1],
            idx_range.i + offset[2])};
  } else if constexpr (IndexSpaceType::loop_tag_v == loop_tag::bovi &&
                       IndexSpaceType::inner_tag_v == inner_tag::memory) {
    const int shift = idx_range.pidx_space->GetMemoryIndexer().GetFlatIdx(
        idx_range.ks + offset[0], idx_range.js + offset[1], idx_range.is + offset[2]);
    return var_view_t<IndexSpaceType>{
        &in(idx_range.block, var, idx_range.ks + offset[0], idx_range.js + offset[1],
            idx_range.is + offset[2]),
        shift, idx_range.pidx_space};
  } else {
    const auto &idx_space = *idx_range.pidx_space;
    return var_view_t<IndexSpaceType>{
        &in(idx_range.block, var, 0, 0, 0),
        static_cast<int>(idx_space.GetMemoryIndexer().GetFlatIdx(offset[0], offset[1],
                                                                 offset[2])),
        &idx_space};
  }
}

template <class IndexSpaceType, class PackType, class... Ts>
struct pack_view_t {
  using TL = parthenon::TypeList<Ts...>;
  KOKKOS_DEFAULTED_FUNCTION
  pack_view_t() = default;

  template <class var_t>
  KOKKOS_INLINE_FUNCTION parthenon::Real &operator()(var_t v, int idx) const {
    static_assert(TL::template IsIn<var_t>(), "Type must be in pack view type list.");
    return data_[SumSizesBefore<TL, var_t>() + v.idx][idx];
  }

  template <class var_t>
  KOKKOS_INLINE_FUNCTION parthenon::Real &operator()(var_t v, Index3 in) const {
    static_assert(TL::template IsIn<var_t>(), "Type must be in pack view type list.");
    return data_[SumSizesBefore<TL, var_t>() + v.idx]
        [pidx_space->GetMemoryIndexer().GetFlatIdx(in.k, in.j, in.i) - shift_];
  }

  template <class var_t>
  KOKKOS_INLINE_FUNCTION parthenon::Real &operator()(var_t v, int k, int j, int i) const {
    return (*this)(v, Index3{k, j, i});
  }

  std::array<parthenon::Real *, SumSizesBefore<TL>()> data_{};
  int shift_ = 0;
  const IndexSpaceType *pidx_space = nullptr;
};

template <loop_tag LOOP_TAG, class PackType, class... Ts>
struct pack_view_t<IndexSpace<LOOP_TAG, inner_tag::logical_coords>, PackType, Ts...> {
  using IndexSpaceType = IndexSpace<LOOP_TAG, inner_tag::logical_coords>;

  const PackType *pack = nullptr;
  int b = 0;
  int s = 0;

  KOKKOS_DEFAULTED_FUNCTION
  pack_view_t() = default;

  KOKKOS_INLINE_FUNCTION
  pack_view_t(const PackType *pack_in, int block, int sparse) : pack(pack_in), b(block), s(sparse) {}

  template <class var_t>
  KOKKOS_FORCEINLINE_FUNCTION parthenon::Real &operator()(var_t v, Index3 in) const {
    static_assert(parthenon::TypeList<Ts...>::template IsIn<var_t>(),
                  "Type must be in pack view type list.");
    return (*pack)(b, var_t(v.idx + s * var_t::size()), in.k, in.j, in.i);
  }

  template <class var_t>
  KOKKOS_FORCEINLINE_FUNCTION parthenon::Real &operator()(var_t v, int k, int j, int i) const {
    static_assert(parthenon::TypeList<Ts...>::template IsIn<var_t>(),
                  "Type must be in pack view type list.");
    return (*pack)(b, var_t(v.idx + s * var_t::size()), k, j, i);
  }
};

template <class IndexSpaceType, class sparse_pack_t, class... Ts>
KOKKOS_INLINE_FUNCTION auto make_pack_view_impl(const InnerIndexRange<IndexSpaceType> &idx_range,
                                                const sparse_pack_t &pack_in, const int b,
                                                const int s, parthenon::TypeList<Ts...>) {
  using TL = parthenon::TypeList<Ts...>;
  if constexpr (IndexSpaceType::inner_tag_v == inner_tag::logical_coords) {
    return pack_view_t<IndexSpaceType, sparse_pack_t, Ts...>{&pack_in, b, s};
  } else {
    pack_view_t<IndexSpaceType, sparse_pack_t, Ts...> out;
    out.pidx_space = idx_range.pidx_space;
    out.shift_ = idx_range.pidx_space->GetMemoryIndexer().GetFlatIdx(idx_range.ks, idx_range.js, idx_range.is);
    ([&] {
      constexpr std::size_t vstart = SumSizesBefore<TL, Ts>();
      const std::size_t sparse_offset = s * Ts::size();
      for (std::size_t v = 0; v < Ts::size(); ++v) {
        auto var = pack_in(b, Ts(v + sparse_offset));
        out.data_[vstart + v] = var.data() + out.shift_;
      }
    }(), ...);
    return out;
  }
}

template <class T>
using check_not_sparse_type = std::bool_constant<!T::is_sparse()>;

template <class T>
using check_sparse_type = std::bool_constant<T::is_sparse()>;

template <class T>
using check_fixed_size = std::bool_constant<(T::size() > 0)>;

template <class IndexSpaceType, class... Ts>
KOKKOS_INLINE_FUNCTION auto make_pack_view(const InnerIndexRange<IndexSpaceType> &idx_range,
                                           const parthenon::SparsePack<Ts...> &pack_in,
                                           const int b) {
  using full_tl = parthenon::TypeList<Ts...>;
  using no_sparse_tl = parthenon::filter_type_list_t<full_tl, check_not_sparse_type>;
  using filtered_tl = parthenon::filter_type_list_t<no_sparse_tl, check_fixed_size>;
  return make_pack_view_impl(idx_range, pack_in, b, 0, filtered_tl{});
}

template <class IndexSpaceType, class... Ts>
KOKKOS_INLINE_FUNCTION auto make_sparse_pack_view(const InnerIndexRange<IndexSpaceType> &idx_range,
                                                  const parthenon::SparsePack<Ts...> &pack_in,
                                                  const int b, const int s) {
  using full_tl = parthenon::TypeList<Ts...>;
  using no_sparse_tl = parthenon::filter_type_list_t<full_tl, check_sparse_type>;
  using filtered_tl = parthenon::filter_type_list_t<no_sparse_tl, check_fixed_size>;
  return make_pack_view_impl(idx_range, pack_in, b, s, filtered_tl{});
}

} // namespace loop_abstraction

} // namespace plb2
