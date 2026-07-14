#pragma once

#include "pack/sparse_pack/sparse_pack.hpp"
#include "utils/type_list.hpp"

#include "loop_abstraction_base.hpp"

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
  KOKKOS_INLINE_FUNCTION parthenon::Real &operator()(var_t v,
                                                     MemoryOffset idx) const {
    return (*this)(v, idx.flat);
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

template <loop_tag LOOP_TAG, loop_backend BACKEND, class PackType, class... Ts>
struct pack_view_t<IndexSpace<LOOP_TAG, inner_tag::logical_coords, BACKEND>, PackType,
                   Ts...> {
  using IndexSpaceType = IndexSpace<LOOP_TAG, inner_tag::logical_coords, BACKEND>;

  const PackType *pack = nullptr;
  int b = 0;
  int s = 0;

  KOKKOS_DEFAULTED_FUNCTION
  pack_view_t() = default;

  KOKKOS_INLINE_FUNCTION
  pack_view_t(const PackType *pack_in, int block, int sparse)
      : pack(pack_in), b(block), s(sparse) {}

  template <class var_t>
  KOKKOS_FORCEINLINE_FUNCTION parthenon::Real &operator()(var_t v, Index3 in) const {
    static_assert(parthenon::TypeList<Ts...>::template IsIn<var_t>(),
                  "Type must be in pack view type list.");
    return (*pack)(b, var_t(v.idx + s * var_t::size()), in.k, in.j, in.i);
  }

  template <class var_t>
  KOKKOS_FORCEINLINE_FUNCTION parthenon::Real &operator()(var_t v, int k, int j,
                                                          int i) const {
    static_assert(parthenon::TypeList<Ts...>::template IsIn<var_t>(),
                  "Type must be in pack view type list.");
    return (*pack)(b, var_t(v.idx + s * var_t::size()), k, j, i);
  }
};

template <class IndexSpaceType, class sparse_pack_t, class... Ts>
KOKKOS_INLINE_FUNCTION auto
make_pack_view_impl(const InnerIndexRange<IndexSpaceType> &idx_range,
                    const sparse_pack_t &pack_in, const int s,
                    parthenon::TypeList<Ts...>) {
  using TL = parthenon::TypeList<Ts...>;
  if constexpr (IndexSpaceType::inner_tag_v == inner_tag::logical_coords) {
    return pack_view_t<IndexSpaceType, sparse_pack_t, Ts...>{&pack_in, idx_range.block,
                                                             s};
  } else {
    pack_view_t<IndexSpaceType, sparse_pack_t, Ts...> out;
    out.pidx_space = idx_range.pidx_space;
    out.shift_ = idx_range.pidx_space->GetMemoryIndexer().GetFlatIdx(
        idx_range.ks, idx_range.js, idx_range.is);
    (
        [&] {
          constexpr std::size_t vstart = SumSizesBefore<TL, Ts>();
          const std::size_t sparse_offset = s * Ts::size();
          for (std::size_t v = 0; v < Ts::size(); ++v) {
            if (pack_in.GetSize(idx_range.block, Ts()) > 0) {
              const auto &var = pack_in(idx_range.block, Ts(v + sparse_offset));
              out.data_[vstart + v] = var.data() + out.shift_;
            } else { 
              out.data_[vstart + v] = nullptr;
            }
          }
        }(),
        ...);
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
KOKKOS_INLINE_FUNCTION auto
make_pack_view(const InnerIndexRange<IndexSpaceType> &idx_range,
               const parthenon::SparsePack<Ts...> &pack_in) {
  using full_tl = parthenon::TypeList<Ts...>;
  using no_sparse_tl = parthenon::filter_type_list_t<full_tl, check_not_sparse_type>;
  using filtered_tl = parthenon::filter_type_list_t<no_sparse_tl, check_fixed_size>;
  return make_pack_view_impl(idx_range, pack_in, 0, filtered_tl{});
}

template <class IndexSpaceType, class... Ts>
KOKKOS_INLINE_FUNCTION auto
make_sparse_pack_view(const InnerIndexRange<IndexSpaceType> &idx_range,
                      const parthenon::SparsePack<Ts...> &pack_in, const int s) {
  using full_tl = parthenon::TypeList<Ts...>;
  using no_sparse_tl = parthenon::filter_type_list_t<full_tl, check_sparse_type>;
  using filtered_tl = parthenon::filter_type_list_t<no_sparse_tl, check_fixed_size>;
  return make_pack_view_impl(idx_range, pack_in, s, filtered_tl{});
}

// A view over the *flux* arrays of a pack for one sweep direction, mirroring
// pack_view_t (which views state). Constructed with a fixed direction; index
// contracts (int / MemoryOffset / Index3 / k,j,i) match pack_view_t.
//
// NOTE: not every variable in a with_fluxes pack has a flux array -- a variable
// present in the pack but without Metadata::WithFluxes leaves its flux slot
// default-constructed (null data, size 0). make_flux_view detects this and stores
// nullptr for such variables; accessing one is a bug caught (debug builds only) by
// the DEBUG_REQUIRE below.
template <class IndexSpaceType, class PackType, class... Ts>
struct flux_view_t {
  using TL = parthenon::TypeList<Ts...>;
  KOKKOS_DEFAULTED_FUNCTION
  flux_view_t() = default;

  template <class var_t>
  KOKKOS_INLINE_FUNCTION parthenon::Real &operator()(var_t v, int idx) const {
    static_assert(TL::template IsIn<var_t>(), "Type must be in flux view type list.");
    parthenon::Real *base = data_[SumSizesBefore<TL, var_t>() + v.idx];
    PARTHENON_DEBUG_REQUIRE(base != nullptr,
                            "flux view accessed for a variable with no flux array");
    return base[idx];
  }

  template <class var_t>
  KOKKOS_INLINE_FUNCTION parthenon::Real &operator()(var_t v, MemoryOffset idx) const {
    return (*this)(v, idx.flat);
  }

  template <class var_t>
  KOKKOS_INLINE_FUNCTION parthenon::Real &operator()(var_t v, Index3 in) const {
    static_assert(TL::template IsIn<var_t>(), "Type must be in flux view type list.");
    parthenon::Real *base = data_[SumSizesBefore<TL, var_t>() + v.idx];
    PARTHENON_DEBUG_REQUIRE(base != nullptr,
                            "flux view accessed for a variable with no flux array");
    return base[pidx_space->GetMemoryIndexer().GetFlatIdx(in.k, in.j, in.i) - shift_];
  }

  template <class var_t>
  KOKKOS_INLINE_FUNCTION parthenon::Real &operator()(var_t v, int k, int j, int i) const {
    return (*this)(v, Index3{k, j, i});
  }

  std::array<parthenon::Real *, SumSizesBefore<TL>()> data_{};
  int shift_ = 0;
  const IndexSpaceType *pidx_space = nullptr;
};

// logical_coords specialization: forward straight to pack.flux() with coordinates,
// no cached pointers (mirrors pack_view_t's logical_coords specialization).
template <loop_tag LOOP_TAG, loop_backend BACKEND, class PackType, class... Ts>
struct flux_view_t<IndexSpace<LOOP_TAG, inner_tag::logical_coords, BACKEND>, PackType,
                   Ts...> {
  using IndexSpaceType = IndexSpace<LOOP_TAG, inner_tag::logical_coords, BACKEND>;

  const PackType *pack = nullptr;
  int b = 0;
  int s = 0;
  int dir = 0;

  KOKKOS_DEFAULTED_FUNCTION
  flux_view_t() = default;

  KOKKOS_INLINE_FUNCTION
  flux_view_t(const PackType *pack_in, int block, int sparse, int dir_in)
      : pack(pack_in), b(block), s(sparse), dir(dir_in) {}

  template <class var_t>
  KOKKOS_FORCEINLINE_FUNCTION parthenon::Real &operator()(var_t v, Index3 in) const {
    static_assert(parthenon::TypeList<Ts...>::template IsIn<var_t>(),
                  "Type must be in flux view type list.");
    return pack->flux(b, dir, var_t(v.idx + s * var_t::size()), in.k, in.j, in.i);
  }

  template <class var_t>
  KOKKOS_FORCEINLINE_FUNCTION parthenon::Real &operator()(var_t v, int k, int j,
                                                          int i) const {
    static_assert(parthenon::TypeList<Ts...>::template IsIn<var_t>(),
                  "Type must be in flux view type list.");
    return pack->flux(b, dir, var_t(v.idx + s * var_t::size()), k, j, i);
  }
};

template <class IndexSpaceType, class sparse_pack_t, class... Ts>
KOKKOS_INLINE_FUNCTION auto
make_flux_view_impl(const InnerIndexRange<IndexSpaceType> &idx_range,
                    const sparse_pack_t &pack_in, const int dir, const int s,
                    parthenon::TypeList<Ts...>) {
  using TL = parthenon::TypeList<Ts...>;
  if constexpr (IndexSpaceType::inner_tag_v == inner_tag::logical_coords) {
    return flux_view_t<IndexSpaceType, sparse_pack_t, Ts...>{&pack_in, idx_range.block,
                                                             s, dir};
  } else {
    flux_view_t<IndexSpaceType, sparse_pack_t, Ts...> out;
    out.pidx_space = idx_range.pidx_space;
    out.shift_ = idx_range.pidx_space->GetMemoryIndexer().GetFlatIdx(
        idx_range.ks, idx_range.js, idx_range.is);
    (
        [&] {
          constexpr std::size_t vstart = SumSizesBefore<TL, Ts>();
          const std::size_t sparse_offset = s * Ts::size();
          for (std::size_t v = 0; v < Ts::size(); ++v) {
            // Cache the flux base pointer only if this variable actually has a flux
            // array; a with_fluxes pack may contain non-WithFluxes variables whose
            // flux slot is empty (see note on flux_view_t).
            const int vidx =
                pack_in.GetLowerBound(idx_range.block, Ts()) + (v + sparse_offset);
            const auto &fvar = pack_in.flux(idx_range.block, dir, vidx);
            if (pack_in.GetSize(idx_range.block, Ts()) > 0 && fvar.size() > 0) {
              out.data_[vstart + v] = fvar.data() + out.shift_;
            } else {
              out.data_[vstart + v] = nullptr;
            }
          }
        }(),
        ...);
    return out;
  }
}

// Flux view over the dense (non-sparse) flux-carrying variables of `pack_in`, for
// sweep direction `dir` (X1DIR/X2DIR/X3DIR).
template <class IndexSpaceType, class... Ts>
KOKKOS_INLINE_FUNCTION auto
make_flux_view(const InnerIndexRange<IndexSpaceType> &idx_range,
               const parthenon::SparsePack<Ts...> &pack_in, const int dir) {
  using full_tl = parthenon::TypeList<Ts...>;
  using no_sparse_tl = parthenon::filter_type_list_t<full_tl, check_not_sparse_type>;
  using filtered_tl = parthenon::filter_type_list_t<no_sparse_tl, check_fixed_size>;
  return make_flux_view_impl(idx_range, pack_in, dir, 0, filtered_tl{});
}

// Flux view over the sparse (material) flux-carrying variables, sparse index `s`.
template <class IndexSpaceType, class... Ts>
KOKKOS_INLINE_FUNCTION auto
make_sparse_flux_view(const InnerIndexRange<IndexSpaceType> &idx_range,
                      const parthenon::SparsePack<Ts...> &pack_in, const int dir,
                      const int s) {
  using full_tl = parthenon::TypeList<Ts...>;
  using no_sparse_tl = parthenon::filter_type_list_t<full_tl, check_sparse_type>;
  using filtered_tl = parthenon::filter_type_list_t<no_sparse_tl, check_fixed_size>;
  return make_flux_view_impl(idx_range, pack_in, dir, s, filtered_tl{});
}

} // namespace loop_abstraction
