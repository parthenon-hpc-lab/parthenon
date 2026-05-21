//========================================================================================
// (C) (or copyright) 2020-2025. Triad National Security, LLC. All rights reserved.
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
#ifndef PACK_SPARSE_PACK_PACK_VIEW_HPP_
#define PACK_SPARSE_PACK_PACK_VIEW_HPP_

#include <algorithm>
#include <array>
#include <utility>
#include <type_traits>
#include <vector>

#include "coordinates/coordinates.hpp"
#include "pack/block_selector.hpp"
#include "pack/pack_utils.hpp"
#include "pack/sparse_pack/sparse_pack.hpp"
#include "utils/concepts_lite.hpp"
#include "utils/type_list.hpp"

namespace parthenon {

namespace impl {
  struct SumAllTypes {};
  
  template <class TL, std::size_t... Is>
  constexpr std::size_t SumSizesImpl(std::index_sequence<Is...>) {
    return (std::size_t{0} + ... + TL::template type<Is>::size());
  }
}

// Exclusive sum up to StopT; default = sum all
template <class TL, class StopT = impl::SumAllTypes>
constexpr std::size_t SumSizesBefore() {
  if constexpr (std::is_same_v<StopT, impl::SumAllTypes>) {
    return impl::SumSizesImpl<TL>(std::make_index_sequence<TL::n_types>{});
  } else {
    constexpr std::size_t stop_idx = TL::template GetIdx<StopT>();
    return impl::SumSizesImpl<TL>(std::make_index_sequence<stop_idx>{});
  }
}

// This needs to be updated some and probably tied into the index space
struct var_view_t {
 public:
  parthenon::Real* data = nullptr;
  int shift = 0;

  // Temporary for testing, probably don't want to carry around the view
  ParArray3D<Real, VariableState> var;

  KOKKOS_FUNCTION
  parthenon::Real &operator()(int idx) const {
    return data[idx + shift];
  }
  KOKKOS_FUNCTION
  parthenon::Real &operator()(int k, int j, int i) const {
    return var(k, j, i);
  }
};

template <class... Ts>
struct pack_view_t {
  using TL = TypeList<Ts...>;
  KOKKOS_DEFAULTED_FUNCTION
  pack_view_t() = default;

  template <class var_t, class... Idxs>
  KOKKOS_INLINE_FUNCTION
  parthenon::Real &operator()(var_t v, Idxs&&... idxs) {
    static_assert(TL::template IsIn<var_t>(), "Type must be in pack view type list.");
    return data_[SumSizesBefore<TL, var_t>() + v.idx](std::forward<Idxs>(idxs)...);
  }

  std::array<var_view_t, SumSizesBefore<TL>()> data_;
};

template <class sparse_pack_t, class... Ts>
KOKKOS_INLINE_FUNCTION
auto make_pack_view_impl(const sparse_pack_t& pack_in, const int b, const int s,
                         TypeList<Ts...>) {
  using TL = TypeList<Ts...>;
  pack_view_t<Ts...> out;
  ([&]{
    constexpr std::size_t vstart = SumSizesBefore<TL, Ts>();
    const std::size_t sparse_offset = s * Ts::size();
    for (std::size_t v = 0; v < Ts::size(); ++v) { 
      out.data_[vstart + v].data = pack_in(b, Ts(v + sparse_offset)).data();
      out.data_[vstart + v].shift = 0;
      // Temporary for testing, probably don't want to carry around the view
      out.data_[vstart + v].var = pack_in(b, Ts(v + sparse_offset));
    }
  }(), ...);
  return out;
}

// Some template filters for filtering out non-desired types
template <class T>
using check_not_sparse_type = std::bool_constant<!T::is_sparse()>;

template <class T>
using check_sparse_type = std::bool_constant<T::is_sparse()>;

template <class T>
using check_fixed_size = std::bool_constant<(T::size() > 0)>;

template <class... Ts>
KOKKOS_INLINE_FUNCTION
auto make_pack_view(const SparsePack<Ts...>& pack_in, const int b) {
  using full_tl = TypeList<Ts...>;
  // Filter out any types from the list that correspond to sparse variables
  using no_sparse_tl = filter_type_list_t<full_tl, check_not_sparse_type>;

  // Filter out any types from the list that are not fixed size
  using filtered_tl = filter_type_list_t<no_sparse_tl, check_fixed_size>;

  return make_pack_view_impl(pack_in, b, 0, filtered_tl{});
}

template <class... Ts>
KOKKOS_INLINE_FUNCTION
auto make_sparse_pack_view(const SparsePack<Ts...>& pack_in, const int b, const int s) {
  using full_tl = TypeList<Ts...>;
  // Filter out any types from the list that correspond to sparse variables
  using no_sparse_tl = filter_type_list_t<full_tl, check_sparse_type>;

  // Filter out any types from the list that are not fixed size
  using filtered_tl = filter_type_list_t<no_sparse_tl, check_fixed_size>;

  return make_pack_view_impl(pack_in, b, s, filtered_tl{});
}

} // namespace parthenon

#endif // PACK_SPARSE_PACK_PACK_VIEW_HPP_
