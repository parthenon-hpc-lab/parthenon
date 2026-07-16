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
#ifndef LOOP_ABSTRACTION_LOOP_ABSTRACTION_PACK_VIEW_HPP_
#define LOOP_ABSTRACTION_LOOP_ABSTRACTION_PACK_VIEW_HPP_

// This file was made in part with generative AI.

#include "pack/sparse_pack/sparse_pack.hpp"
#include "utils/type_list.hpp"

#include "loop_abstraction_base.hpp"

namespace parthenon::loop_abstraction {

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

// A view over a *single* (anonymous) variable of a pack -- the single-variable analog
// of pack_view_t. Constructed from either a raw integer index into the pack or a typed
// index (ccmat::rho(m)); both collapse to one absolute variable index at construction.
// Unlike pack_view_t, operator() takes no variable argument: it addresses that one
// variable directly. Index contracts (int / MemoryOffset / Index3 / k,j,i) match
// pack_view_t.
template <class IndexSpaceType, class PackType>
struct var_view_t {
  KOKKOS_DEFAULTED_FUNCTION
  var_view_t() = default;

  KOKKOS_FORCEINLINE_FUNCTION parthenon::Real &operator()(int idx) const {
    return data_[idx];
  }
  KOKKOS_FORCEINLINE_FUNCTION parthenon::Real &operator()(MemoryOffset idx) const {
    return data_[idx.flat];
  }
  KOKKOS_FORCEINLINE_FUNCTION parthenon::Real &operator()(Index3 in) const {
    return data_[pidx_space->GetMemoryIndexer().GetFlatIdx(in.k, in.j, in.i) - shift_];
  }
  KOKKOS_FORCEINLINE_FUNCTION parthenon::Real &operator()(int k, int j, int i) const {
    return (*this)(Index3{k, j, i});
  }

  parthenon::Real *data_ = nullptr;
  int shift_ = 0;
  const IndexSpaceType *pidx_space = nullptr;
};

// logical_coords specialization: forward straight to pack(b, vidx, k,j,i), no cached
// pointer (mirrors pack_view_t's logical_coords specialization).
template <loop_tag LOOP_TAG, loop_backend BACKEND, class PackType>
struct var_view_t<IndexSpace<LOOP_TAG, inner_tag::logical_coords, BACKEND>, PackType> {
  using IndexSpaceType = IndexSpace<LOOP_TAG, inner_tag::logical_coords, BACKEND>;

  const PackType *pack = nullptr;
  int b = 0;
  int vidx = 0;

  KOKKOS_DEFAULTED_FUNCTION
  var_view_t() = default;

  KOKKOS_INLINE_FUNCTION
  var_view_t(const PackType *pack_in, int block, int var_in)
      : pack(pack_in), b(block), vidx(var_in) {}

  KOKKOS_FORCEINLINE_FUNCTION parthenon::Real &operator()(Index3 in) const {
    return (*pack)(b, vidx, in.k, in.j, in.i);
  }
  KOKKOS_FORCEINLINE_FUNCTION parthenon::Real &operator()(int k, int j, int i) const {
    return (*pack)(b, vidx, k, j, i);
  }
};

// View over a single (anonymous) variable of `pack_in`, addressed by raw int or typed
// index. See var_view_t.
template <class IndexSpaceType, class PackType, class IndexType>
KOKKOS_INLINE_FUNCTION auto
make_var_view(const InnerIndexRange<IndexSpaceType> &idx_range, const PackType &pack_in,
              const IndexType &var) {
  const int vidx = pack_in.GetIndex(idx_range.block, var);
  if constexpr (IndexSpaceType::inner_tag_v == inner_tag::logical_coords) {
    return var_view_t<IndexSpaceType, PackType>{&pack_in, idx_range.block, vidx};
  } else {
    var_view_t<IndexSpaceType, PackType> out;
    out.pidx_space = idx_range.pidx_space;
    out.shift_ = idx_range.pidx_space->GetMemoryIndexer().GetFlatIdx(
        idx_range.ks, idx_range.js, idx_range.is);
    out.data_ = pack_in(idx_range.block, vidx).data() + out.shift_;
    return out;
  }
}

// A view over the *flux* arrays of a pack for one sweep direction, mirroring
// pack_view_t (which views state). Constructed with a fixed direction; index
// contracts (int / MemoryOffset / Index3 / k,j,i) match pack_view_t.
//
// NOTE: not every variable in a with_fluxes pack has a flux array -- a variable
// present in the pack but without Metadata::WithFluxes leaves its flux slot
// default-constructed (null data, size 0). make_flux_pack_view detects this and stores
// nullptr for such variables; accessing one is a bug caught (debug builds only) by
// the DEBUG_REQUIRE below.
template <class IndexSpaceType, class PackType, class... Ts>
struct flux_pack_view_t {
  using TL = parthenon::TypeList<Ts...>;
  KOKKOS_DEFAULTED_FUNCTION
  flux_pack_view_t() = default;

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
struct flux_pack_view_t<IndexSpace<LOOP_TAG, inner_tag::logical_coords, BACKEND>, PackType,
                        Ts...> {
  using IndexSpaceType = IndexSpace<LOOP_TAG, inner_tag::logical_coords, BACKEND>;

  const PackType *pack = nullptr;
  int b = 0;
  int s = 0;
  int dir = 0;

  KOKKOS_DEFAULTED_FUNCTION
  flux_pack_view_t() = default;

  KOKKOS_INLINE_FUNCTION
  flux_pack_view_t(const PackType *pack_in, int block, int sparse, int dir_in)
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
make_flux_pack_view_impl(const InnerIndexRange<IndexSpaceType> &idx_range,
                         const sparse_pack_t &pack_in, const int dir, const int s,
                         parthenon::TypeList<Ts...>) {
  using TL = parthenon::TypeList<Ts...>;
  if constexpr (IndexSpaceType::inner_tag_v == inner_tag::logical_coords) {
    return flux_pack_view_t<IndexSpaceType, sparse_pack_t, Ts...>{
        &pack_in, idx_range.block, s, dir};
  } else {
    flux_pack_view_t<IndexSpaceType, sparse_pack_t, Ts...> out;
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
            if (pack_in.GetSize(idx_range.block, Ts()) > 0) {
              const int vidx = pack_in.GetLowerBound(idx_range.block, Ts()) + (v + sparse_offset);
              const auto &fvar = pack_in.flux(idx_range.block, dir, vidx);
              out.data_[vstart + v] = fvar.size() > 0 ? fvar.data() + out.shift_ : nullptr;
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
make_flux_pack_view(const InnerIndexRange<IndexSpaceType> &idx_range,
                    const parthenon::SparsePack<Ts...> &pack_in, const int dir) {
  using full_tl = parthenon::TypeList<Ts...>;
  using no_sparse_tl = parthenon::filter_type_list_t<full_tl, check_not_sparse_type>;
  using filtered_tl = parthenon::filter_type_list_t<no_sparse_tl, check_fixed_size>;
  return make_flux_pack_view_impl(idx_range, pack_in, dir, 0, filtered_tl{});
}

// Flux view over the sparse (material) flux-carrying variables, sparse index `s`.
template <class IndexSpaceType, class... Ts>
KOKKOS_INLINE_FUNCTION auto
make_sparse_flux_pack_view(const InnerIndexRange<IndexSpaceType> &idx_range,
                           const parthenon::SparsePack<Ts...> &pack_in, const int dir,
                           const int s) {
  using full_tl = parthenon::TypeList<Ts...>;
  using no_sparse_tl = parthenon::filter_type_list_t<full_tl, check_sparse_type>;
  using filtered_tl = parthenon::filter_type_list_t<no_sparse_tl, check_fixed_size>;
  return make_flux_pack_view_impl(idx_range, pack_in, dir, s, filtered_tl{});
}

// A view over the *flux* array of a *single* (anonymous) variable of a pack, for one
// sweep direction -- the single-variable analog of flux_pack_view_t, and the flux-side
// counterpart of var_view_t. Constructed from either a raw int or a typed index; both
// collapse to one absolute variable index at construction. operator() takes no variable
// argument. Index contracts match var_view_t.
template <class IndexSpaceType, class PackType>
struct flux_view_t {
  KOKKOS_DEFAULTED_FUNCTION
  flux_view_t() = default;

  KOKKOS_FORCEINLINE_FUNCTION parthenon::Real &operator()(int idx) const {
    return data_[idx];
  }
  KOKKOS_FORCEINLINE_FUNCTION parthenon::Real &operator()(MemoryOffset idx) const {
    return data_[idx.flat];
  }
  KOKKOS_FORCEINLINE_FUNCTION parthenon::Real &operator()(Index3 in) const {
    return data_[pidx_space->GetMemoryIndexer().GetFlatIdx(in.k, in.j, in.i) - shift_];
  }
  KOKKOS_FORCEINLINE_FUNCTION parthenon::Real &operator()(int k, int j, int i) const {
    return (*this)(Index3{k, j, i});
  }

  parthenon::Real *data_ = nullptr;
  int shift_ = 0;
  const IndexSpaceType *pidx_space = nullptr;
};

// logical_coords specialization: forward straight to pack.flux(b, dir, vidx, k,j,i),
// no cached pointer (mirrors flux_pack_view_t's logical_coords specialization).
template <loop_tag LOOP_TAG, loop_backend BACKEND, class PackType>
struct flux_view_t<IndexSpace<LOOP_TAG, inner_tag::logical_coords, BACKEND>, PackType> {
  using IndexSpaceType = IndexSpace<LOOP_TAG, inner_tag::logical_coords, BACKEND>;

  const PackType *pack = nullptr;
  int b = 0;
  int vidx = 0;
  int dir = 0;

  KOKKOS_DEFAULTED_FUNCTION
  flux_view_t() = default;

  KOKKOS_INLINE_FUNCTION
  flux_view_t(const PackType *pack_in, int block, int var_in, int dir_in)
      : pack(pack_in), b(block), vidx(var_in), dir(dir_in) {}

  KOKKOS_FORCEINLINE_FUNCTION parthenon::Real &operator()(Index3 in) const {
    return pack->flux(b, dir, vidx, in.k, in.j, in.i);
  }
  KOKKOS_FORCEINLINE_FUNCTION parthenon::Real &operator()(int k, int j, int i) const {
    return pack->flux(b, dir, vidx, k, j, i);
  }
};

// Flux view over a single (anonymous) variable of `pack_in` for sweep direction `dir`
// (X1DIR/X2DIR/X3DIR), addressed by raw int or typed index. See flux_view_t.
template <class IndexSpaceType, class PackType, class IndexType>
KOKKOS_INLINE_FUNCTION auto
make_flux_view(const InnerIndexRange<IndexSpaceType> &idx_range, const PackType &pack_in,
               const int dir, const IndexType &var) {
  const int vidx = pack_in.GetIndex(idx_range.block, var);
  if constexpr (IndexSpaceType::inner_tag_v == inner_tag::logical_coords) {
    return flux_view_t<IndexSpaceType, PackType>{&pack_in, idx_range.block, vidx, dir};
  } else {
    flux_view_t<IndexSpaceType, PackType> out;
    out.pidx_space = idx_range.pidx_space;
    out.shift_ = idx_range.pidx_space->GetMemoryIndexer().GetFlatIdx(
        idx_range.ks, idx_range.js, idx_range.is);
    out.data_ = pack_in.flux(idx_range.block, dir, vidx).data() + out.shift_;
    return out;
  }
}

} // namespace parthenon::loop_abstraction

#endif // LOOP_ABSTRACTION_LOOP_ABSTRACTION_PACK_VIEW_HPP_
