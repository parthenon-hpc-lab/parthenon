#ifndef UTILS_TYPE_ARRAY_HPP_
#define UTILS_TYPE_ARRAY_HPP_

#include "Kokkos_Macros.hpp"
#include "pack/sparse_pack.hpp"
#include "utils/type_list.hpp"
#include <utility>

namespace parthenon {
namespace impl {
template <typename>
struct TypeListArray {};

template <template <typename...> typename PackType, typename... Ts>
struct TypeListArray<PackType<Ts...>> {
  using type = PackType<Ts...>;
  using Arr_t = Kokkos::Array<Real, type::n_types>;

  KOKKOS_INLINE_FUNCTION TypeListArray(const type &pack_in) : pack(pack_in) {}
  KOKKOS_INLINE_FUNCTION TypeListArray(const type &pack_in, const Real &value)
      : TypeListArray(pack_in) {
    for (int idx = 0; idx < type::n_types; idx++) {
      data[idx] = value;
    }
  }
  KOKKOS_INLINE_FUNCTION TypeListArray(const type &pack_in, Arr_t data_in)
      : TypeListArray(pack_in), data(data_in) {}

  template <typename Var>
  KOKKOS_INLINE_FUNCTION Real &operator()(const Var &var) {
    return data[pack.GetIndex(var)];
  }

  KOKKOS_INLINE_FUNCTION Real &operator[](const std::size_t &idx) { return data[idx]; }

 private:
  Arr_t data;
  const type &pack;
};

template <typename PackType, typename ScratchPad, typename... Ts>
struct ScratchPackArray_impl {
  KOKKOS_INLINE_FUNCTION
  ScratchPackArray_impl(ScratchPad scratch_, const int &b_, const int &i_)
      : scratch(scratch_), b(b_), i(i_) {}

  template <typename V>
  KOKKOS_INLINE_FUNCTION Real &operator()(const V &var) const {
    return scratch(GetIndex(var), i);
  }

 private:
  ScratchPad scratch;
  const int i, b;
};

template <typename ScratchPad, typename... Ts>
struct ScratchPack_impl {
  KOKKOS_INLINE_FUNCTION
  ScratchPack_impl(const SparsePack<Ts...> &pack_, ScratchPad scratch_, const int &b_,
                   const int &i_)
      : pack(pack_), scratch(scratch_), b(b_), i(i_) {}

  template <typename V>
  KOKKOS_INLINE_FUNCTION Real &operator()(const V &var) const {
    return scratch(pack.GetIndex(b, var), i);
  }

 private:
  const SparsePack<Ts...> &pack;
  ScratchPad scratch;
  const int i, b;
};

template <typename ScratchPad, typename... Ts>
KOKKOS_INLINE_FUNCTION auto ScratchPack(const SparsePack<Ts...> &pack,
                                        ScratchPad &scratch, const int &b, const int &i) {
  return ScratchPack_impl<ScratchPad, Ts...>(pack, scratch, b, i);
}
} // namespace impl

template <template <typename...> typename PackType, typename... Ts, typename... Args>
KOKKOS_INLINE_FUNCTION auto TypeListArray(const PackType<Ts...> &pack, Args &&...args) {
  return impl::TypeListArray<PackType<Ts...>>(pack, std::forward<Args>(args)...);
}
} // namespace parthenon
#endif // UTILS_TYPE_ARRAY_HPP_
