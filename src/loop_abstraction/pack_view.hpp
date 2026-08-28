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
#ifndef LOOP_ABSTRACTION_PACK_VIEW_HPP_
#define LOOP_ABSTRACTION_PACK_VIEW_HPP_

// This file was made in part with generative AI.

// Views that adapt a SparsePack of state variables to the loop-abstraction index
// contracts. pack_view_t exposes a set of variable types; var_view_t is the
// single-variable analog. Each has a logical_coords specialization that forwards
// straight to the pack and a flat/memory form that caches base pointers. The shared
// check_* predicates defined here also drive the flux-side views in
// flux_view.hpp.

#include "pack/sparse_pack/sparse_pack.hpp"
#include "utils/type_list.hpp"

#include "base.hpp"

namespace parthenon::loop_abstraction {

template <loop_tag LOOP_TAG, inner_tag INNER_TAG, class PackType, class... Ts>
struct pack_view_t {
  using TL = parthenon::TypeList<Ts...>;
  KOKKOS_DEFAULTED_FUNCTION
  pack_view_t() = default;

  template <class var_t>
  KOKKOS_INLINE_FUNCTION parthenon::Real &operator()(var_t v, int idx) const {
    static_assert(TL::template IsIn<var_t>(), "Type must be in pack view type list.");
    Real *base = data_[SumSizesBefore<TL, var_t>() + v.idx];
    PARTHENON_DEBUG_REQUIRE(base != nullptr,
                            "pack view accessed for a variable with no flux array");
    return base[idx];
  }

  template <class var_t>
  KOKKOS_INLINE_FUNCTION parthenon::Real &operator()(var_t v, MemoryOffset idx) const {
    return (*this)(v, idx.flat);
  }

  template <class var_t>
  KOKKOS_INLINE_FUNCTION parthenon::Real &operator()(TopologicalElement te, var_t v,
                                                     int idx) const {
    static_assert(TL::template IsIn<var_t>(), "Type must be in pack view type list.");
    PARTHENON_DEBUG_REQUIRE(GetTopologicalType(te) == var_t::topological_type,
                            "Topological element must match variable topological type.");
    Real *base = data_[SumSizesBefore<TL, var_t>() +
                       (static_cast<int>(te) % 3) * var_t::size() + v.idx];
    PARTHENON_DEBUG_REQUIRE(base != nullptr,
                            "pack view accessed for a variable with no flux array");
    return base[idx];
  }

  template <class var_t>
  KOKKOS_INLINE_FUNCTION parthenon::Real &operator()(TopologicalElement te, var_t v,
                                                     MemoryOffset idx) const {
    return (*this)(te, v, idx.flat);
  }

  template <class var_t>
  KOKKOS_INLINE_FUNCTION parthenon::Real &operator()(var_t v, Index3 in) const {
    static_assert(TL::template IsIn<var_t>(), "Type must be in pack view type list.");
    Real *base = data_[SumSizesBefore<TL, var_t>() + v.idx];
    PARTHENON_DEBUG_REQUIRE(base != nullptr,
                            "pack view accessed for a variable with no flux array");
    return base[memory_indexer->GetFlatIdx(in.k, in.j, in.i) - shift_];
  }

  template <class var_t>
  KOKKOS_INLINE_FUNCTION parthenon::Real &operator()(var_t v, int k, int j, int i) const {
    return (*this)(v, Index3{k, j, i});
  }

  template <class var_t>
  KOKKOS_INLINE_FUNCTION parthenon::Real &operator()(TopologicalElement te, var_t v,
                                                     Index3 in) const {
    static_assert(TL::template IsIn<var_t>(), "Type must be in pack view type list.");
    PARTHENON_DEBUG_REQUIRE(GetTopologicalType(te) == var_t::topological_type,
                            "Topological element must match variable topological type.");
    Real *base = data_[SumSizesBefore<TL, var_t>() +
                       (static_cast<int>(te) % 3) * var_t::size() + v.idx];
    PARTHENON_DEBUG_REQUIRE(base != nullptr,
                            "pack view accessed for a variable with no flux array");
    return base[memory_indexer->GetFlatIdx(in.k, in.j, in.i) - shift_];
  }

  template <class var_t>
  KOKKOS_INLINE_FUNCTION parthenon::Real &operator()(TopologicalElement te, var_t v,
                                                     int k, int j, int i) const {
    return (*this)(te, v, Index3{k, j, i});
  }

  std::array<parthenon::Real *, SumSizesBefore<TL>()> data_{};
  int shift_ = 0;
  const parthenon::Indexer3D *memory_indexer = nullptr;
};

template <loop_tag LOOP_TAG, class PackType, class... Ts>
class pack_view_t<LOOP_TAG, inner_tag::logical_coords, PackType, Ts...> {
  int b = 0;
  int s = 0;

 public:
  const PackType *pack = nullptr;

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

  template <class var_t>
  KOKKOS_FORCEINLINE_FUNCTION parthenon::Real &operator()(TopologicalElement te, var_t v,
                                                          Index3 in) const {
    static_assert(parthenon::TypeList<Ts...>::template IsIn<var_t>(),
                  "Type must be in pack view type list.");
    PARTHENON_DEBUG_REQUIRE(GetTopologicalType(te) == var_t::topological_type,
                            "Topological element must match variable topological type.");
    return (*pack)(b, te, var_t(v.idx + s * var_t::size()), in.k, in.j, in.i);
  }

  template <class var_t>
  KOKKOS_FORCEINLINE_FUNCTION parthenon::Real &operator()(TopologicalElement te, var_t v,
                                                          int k, int j, int i) const {
    static_assert(parthenon::TypeList<Ts...>::template IsIn<var_t>(),
                  "Type must be in pack view type list.");
    PARTHENON_DEBUG_REQUIRE(GetTopologicalType(te) == var_t::topological_type,
                            "Topological element must match variable topological type.");
    return (*pack)(b, te, var_t(v.idx + s * var_t::size()), k, j, i);
  }
};

template <class IndexSpaceType, class sparse_pack_t, class... Ts>
KOKKOS_INLINE_FUNCTION auto
make_pack_view_impl(const InnerIndexRange<IndexSpaceType> &idx_range,
                    const sparse_pack_t &pack_in, const int s,
                    parthenon::TypeList<Ts...>) {
  using TL = parthenon::TypeList<Ts...>;
  constexpr loop_tag LOOP_TAG = IndexSpaceType::loop_tag_v;
  constexpr inner_tag INNER_TAG = IndexSpaceType::inner_tag_v;
  if constexpr (INNER_TAG == inner_tag::logical_coords) {
    return pack_view_t<LOOP_TAG, INNER_TAG, sparse_pack_t, Ts...>{&pack_in,
                                                                  idx_range.block, s};
  } else {
    pack_view_t<LOOP_TAG, INNER_TAG, sparse_pack_t, Ts...> out;
    const auto &memory_indexer = idx_range.pidx_space->GetMemoryIndexer();
    out.memory_indexer = &memory_indexer;
    out.shift_ = memory_indexer.GetFlatIdx(idx_range.ks, idx_range.js, idx_range.is);
    (
        [&] {
          constexpr std::size_t vstart = SumSizesBefore<TL, Ts>();
          constexpr std::size_t ntopo = NumberOfTopologicalElements(Ts::topological_type);
          const std::size_t sparse_offset = s * Ts::size();
          for (std::size_t t = 0; t < ntopo; ++t) {
            for (std::size_t v = 0; v < Ts::size(); ++v) {
              if (pack_in.GetSize(idx_range.block, Ts()) > 0) {
                const auto te = GetTopologicalElementInDir(Ts::topological_type, t);
                const auto &var = pack_in(idx_range.block, te, Ts(v + sparse_offset));
                out.data_[vstart + t * Ts::size() + v] = var.data() + out.shift_;
              } else {
                out.data_[vstart + t * Ts::size() + v] = nullptr;
              }
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
  using sparse_tl = parthenon::filter_type_list_t<full_tl, check_sparse_type>;
  using filtered_tl = parthenon::filter_type_list_t<sparse_tl, check_fixed_size>;
  return make_pack_view_impl(idx_range, pack_in, s, filtered_tl{});
}

// A view over a *single* (anonymous) variable of a pack -- the single-variable analog
// of pack_view_t. Constructed from either a raw integer index into the pack or a typed
// index (ccmat::rho(m)); both collapse to one absolute variable index at construction.
// Unlike pack_view_t, operator() takes no variable argument: it addresses that one
// variable directly. Index contracts (int / MemoryOffset / Index3 / k,j,i) match
// pack_view_t.
template <loop_tag LOOP_TAG, inner_tag INNER_TAG, class PackType>
struct var_view_t {
  KOKKOS_DEFAULTED_FUNCTION
  var_view_t() = default;

  KOKKOS_FORCEINLINE_FUNCTION parthenon::Real &operator()(int idx) const {
    return data_[idx];
  }

  KOKKOS_FORCEINLINE_FUNCTION parthenon::Real &operator()(int var_offset, int idx) const {
    return data_[var_offset * stride_ + idx];
  }

  KOKKOS_FORCEINLINE_FUNCTION parthenon::Real &operator()(MemoryOffset idx) const {
    return data_[idx.flat];
  }
  KOKKOS_FORCEINLINE_FUNCTION parthenon::Real &operator()(Index3 in) const {
    return data_[memory_indexer->GetFlatIdx(in.k, in.j, in.i) - shift_];
  }
  KOKKOS_FORCEINLINE_FUNCTION parthenon::Real &operator()(int k, int j, int i) const {
    return (*this)(Index3{k, j, i});
  }

  // TODO(JMM/LFR): If we are really worried about the number of
  // members in var_views impacting register pressure or having other
  // performance impacts, we could specialize var_views more to only
  // include the Real* member and nothing else for most inner loop
  // tags. That would require not allowing var_views to be used in
  // loops with the functor signature (int k, int j, int i). We could
  // also template on the variable type itself, and only store the
  // offset when the variable is not a scalar. This may be overkill
  // though.
  parthenon::Real *data_ = nullptr;
  int shift_ = 0;
  int stride_ = 0;
  const parthenon::Indexer3D *memory_indexer = nullptr;
};

// logical_coords specialization: forward straight to pack(b, vidx, k,j,i), no cached
// pointer (mirrors pack_view_t's logical_coords specialization).
template <loop_tag LOOP_TAG, class PackType>
struct var_view_t<LOOP_TAG, inner_tag::logical_coords, PackType> {
  const PackType *pack = nullptr;
  int b = 0;
  int vidx = 0;
  TopologicalElement te;

  KOKKOS_DEFAULTED_FUNCTION
  var_view_t() = default;

  KOKKOS_INLINE_FUNCTION
  var_view_t(const PackType *pack_in, int block, int var_in, TopologicalElement te)
      : pack(pack_in), b(block), vidx(var_in), te(te) {}

  KOKKOS_INLINE_FUNCTION
  var_view_t(const PackType *pack_in, int block, int var_in)
      : pack(pack_in), b(block), vidx(var_in), te(TopologicalElement::CC) {}

  KOKKOS_FORCEINLINE_FUNCTION parthenon::Real &operator()(Index3 in) const {
    return (*pack)(b, te, vidx, in.k, in.j, in.i);
  }
  KOKKOS_FORCEINLINE_FUNCTION parthenon::Real &operator()(int k, int j, int i) const {
    return (*pack)(b, te, vidx, k, j, i);
  }
  KOKKOS_FORCEINLINE_FUNCTION parthenon::Real &operator()(int offset, Index3 in) const {
    return (*pack)(b, te, vidx + offset, in.k, in.j, in.i);
  }
  KOKKOS_FORCEINLINE_FUNCTION parthenon::Real &operator()(int offset, int k, int j,
                                                          int i) const {
    return (*pack)(b, te, vidx + offset, k, j, i);
  }
};

// View over a single (anonymous) variable of `pack_in`, addressed by raw int or typed
// index. See var_view_t.
template <class IndexSpaceType, class PackType, class IndexType>
KOKKOS_INLINE_FUNCTION auto
make_var_view(const InnerIndexRange<IndexSpaceType> &idx_range, const PackType &pack_in,
              TopologicalElement te, const IndexType &var) {
  constexpr loop_tag LOOP_TAG = IndexSpaceType::loop_tag_v;
  constexpr inner_tag INNER_TAG = IndexSpaceType::inner_tag_v;
  const int vidx = pack_in.GetIndex(idx_range.block, var);
  if constexpr (INNER_TAG == inner_tag::logical_coords) {
    return var_view_t<LOOP_TAG, INNER_TAG, PackType>{&pack_in, idx_range.block, vidx, te};
  } else {
    var_view_t<LOOP_TAG, INNER_TAG, PackType> out;
    const auto &memory_indexer = idx_range.pidx_space->GetMemoryIndexer();
    out.memory_indexer = &memory_indexer;
    out.shift_ = memory_indexer.GetFlatIdx(idx_range.ks, idx_range.js, idx_range.is);
    out.data_ = pack_in(idx_range.block, te, vidx).data() + out.shift_;
    const int vidx_next = ((pack_in.GetSize() > vidx + 1) &&
                           (pack_in(idx_range.block, te, vidx).tensor_components > 1))
                              ? vidx + 1
                              : vidx;
    out.stride_ = pack_in(idx_range.block, te, vidx_next).data() + out.shift_ - out.data_;
    return out;
  }
}

template <class IndexSpaceType, class PackType, class IndexType>
KOKKOS_INLINE_FUNCTION auto
make_var_view(const InnerIndexRange<IndexSpaceType> &idx_range, const PackType &pack_in,
              const IndexType &var) {
  return make_var_view(idx_range, pack_in, TopologicalElement::CC, var);
}

} // namespace parthenon::loop_abstraction

#endif // LOOP_ABSTRACTION_PACK_VIEW_HPP_
