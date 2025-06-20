#ifndef UTILS_TYPE_ARRAY_HPP_
#define UTILS_TYPE_ARRAY_HPP_

#include "Kokkos_Macros.hpp"
#include "pack/sparse_pack.hpp"
#include "utils/concepts_lite.hpp"
#include "utils/type_list.hpp"
#include <utility>

namespace parthenon {
template <typename>
struct SparsePackList {};

template <typename... Ts>
struct SparsePackList<SparsePack<Ts...>> {
  using type = SparsePack<Ts...>;
  static constexpr std::size_t ncomp = sizeof...(Ts);

  KOKKOS_INLINE_FUNCTION SparsePackList(const type &pack_in, const int &b_in)
      : pack(pack_in), b(b_in) {}

  template <typename T>
  KOKKOS_INLINE_FUNCTION std::size_t GetIndex(const T &t) const {
    return pack.GetIndex(b, t);
  }

 private:
  const type &pack;
  const int b;
};

template <typename... Vars>
struct VarList {
  template <typename... Ts>
  using TypeList = parthenon::TypeList<Ts...>;

  static constexpr std::size_t GetSize() {
    std::size_t size = 0;
    ([&] { size += Vars::ncomp; }(), ...);
    return size;
  }

  static constexpr std::size_t ncomp = GetSize();
  template <typename V>
  KOKKOS_INLINE_FUNCTION std::size_t GetIndex(const V &var) const {
    return GetIndex_(TypeList<Vars...>(), var);
  }

 private:
  template <typename V, typename... Vs>
  KOKKOS_INLINE_FUNCTION std::size_t GetIndex_(TypeList<V, Vs...>, const V &var) const {
    return var.idx;
  }

  template <typename V, typename U, typename... Us>
  KOKKOS_INLINE_FUNCTION std::size_t GetIndex_(TypeList<U, Us...>, const V &var) const {
    return U::ncomp + GetIndex_(TypeList<Us...>(), var);
  }
};

namespace impl {
template <typename>
struct TypeListArray {};

template <template <typename...> typename PackType, typename... Ts>
struct TypeListArray<PackType<Ts...>> {
  using type = PackType<Ts...>;
  using Arr_t = Kokkos::Array<Real, type::ncomp>;

  KOKKOS_INLINE_FUNCTION TypeListArray(const type &pack_in) : pack(pack_in) {}
  KOKKOS_INLINE_FUNCTION TypeListArray(const type &pack_in, const Real &value)
      : TypeListArray(pack_in) {
    for (int idx = 0; idx < type::ncomp; idx++) {
      data[idx] = value;
    }
  }
  KOKKOS_INLINE_FUNCTION TypeListArray(const type &pack_in, Arr_t data_in)
      : TypeListArray(pack_in), data(data_in) {}

  template <typename V, REQUIRES(IncludesType<V, Ts...>::value)>
  KOKKOS_INLINE_FUNCTION Real &operator()(const V &var) {
    return data[pack.GetIndex(var)];
  }

  KOKKOS_INLINE_FUNCTION Real &operator[](const std::size_t &idx) { return data[idx]; }

 private:
  Arr_t data;
  const type &pack;
};

template <typename, typename>
struct ScratchPack_impl {};

template <typename ScratchPad, template <typename...> typename PackType, typename... Ts>
struct ScratchPack_impl<ScratchPad, PackType<Ts...>> {
  using type = PackType<Ts...>;
  KOKKOS_INLINE_FUNCTION
  ScratchPack_impl(const type pack_, ScratchPad scratch_, const int &i_)
      : pack(pack_), scratch(scratch_), i(i_) {}

  template <typename V, REQUIRES(IncludesType<V, Ts...>::value)>
  KOKKOS_INLINE_FUNCTION Real &operator()(const V &var) const {
    return scratch(pack.GetIndex(var), i);
  }

  template <typename V, REQUIRES(IncludesType<V, Ts...>::value)>
  KOKKOS_INLINE_FUNCTION Real &operator()(const V &var, const int &idx) const {
    return scratch(pack.GetIndex(var), i + idx);
  }

  KOKKOS_INLINE_FUNCTION Real &operator()(const int &var, const int &idx) const {
    return scratch(var, i + idx);
  }

  KOKKOS_INLINE_FUNCTION Real &operator()(const int &var) const {
    return scratch(var, i);
  }

 private:
  const type pack;
  ScratchPad scratch;
  const int i;
};

// TypeList containers that can be used to index into an integer array need
// to provide
//    * a static constexpr std::size_t ncomp
//       that declares the size of the array to index into
//       note that this is not used by the ScratchPack, as it assumes
//       that the scratch memory is already allocated
//    * and an int GetIndex() method templated on the types in the list
template <typename... Ts>
struct PackLike {
  template <typename T, REQUIRES(implements<integral(decltype(T::ncomp))>::value)>
  auto requires_(T) -> void_t<decltype(T::ncomp), decltype(T().GetIndex(Ts()))...>;
};

} // namespace impl

template <template <typename...> typename PackType, typename... Ts, typename... Args,
          REQUIRES(implements<PackLike<Ts...>(PackType<Ts...>)>::value &&
                   !is_specialization_of<PackType<Ts...>, SparsePackList>::value)>
KOKKOS_INLINE_FUNCTION auto TypeListArray(const PackType<Ts...> &pack, Args &&...args) {
  return impl::TypeListArray<PackType<Ts...>>(pack, std::forward<Args>(args)...);
}

template <typename ScratchPad, template <typename...> typename PackType, typename... Ts,
          REQUIRES(implements<PackLike<Ts...>(PackType<Ts...>)>::value)>
KOKKOS_INLINE_FUNCTION auto ScratchPack(const PackType<Ts...> &pack, ScratchPad &scratch,
                                        const int &i) {
  return ScratchPack_impl<ScratchPad, PackType<Ts...>>(pack, scratch, i);
}

template <typename ScratchPad, typename... Ts>
KOKKOS_INLINE_FUNCTION auto ScratchPack(const SparsePack<Ts...> &pack,
                                        ScratchPad &scratch, const int &b, const int &i) {
  auto spl = SparsePackList(pack, b);
  return ScratchPack(spl, scratch, i);
}

} // namespace parthenon
#endif // UTILS_TYPE_ARRAY_HPP_
