#ifndef PACK_SUBPACK_HPP_
#define PACK_SUBPACK_HPP_

#include "basic_types.hpp"
#include "utils/concepts_lite.hpp"

namespace parthenon {

namespace impl {

constexpr int axis_to_kji(Axis axis) {
  switch (axis) {
  case (Axis::I):
    return 2;
  case (Axis::J):
    return 1;
  case (Axis::K):
    return 0;
  }
}

template <typename PackType, Axis... axes>
struct SubPack_impl {
  KOKKOS_INLINE_FUNCTION SubPack_impl(PackType &pack, const int &b, const int &k,
                                      const int &j, const int &i)
      : pack_(pack), b_(b), k_(k), j_(j), i_(i) {}

  template <typename Var_t>
  KOKKOS_INLINE_FUNCTION Real &operator()(const Var_t &var) const {
    return pack_(b_, var, k_, j_, i_);
  }

  template <typename Var_t>
  KOKKOS_INLINE_FUNCTION Real &operator()(TopologicalElement te, const Var_t &var) const {
    return pack_(b_, te, var, k_, j_, i_);
  }

  template <typename Var_t>
  KOKKOS_INLINE_FUNCTION Real &flux(TopologicalElement te, const Var_t &var) const {
    return pack_.flux(b_, te, var, k_, j_, i_);
  }

  template <typename V>
  KOKKOS_INLINE_FUNCTION std::size_t GetSize(const V &var) const {
    return pack_.GetSize(b_, var);
  }

  KOKKOS_INLINE_FUNCTION
  const auto &GetCoordinates() const { return pack_.GetCoordinates(b_); }

 private:
  PackType &pack_;
  const int b_, k_, j_, i_;
};

template <typename PackType, Axis... axes>
struct StencilSubPack_impl {
  KOKKOS_INLINE_FUNCTION StencilSubPack_impl(PackType &pack, const int &b, const int &k,
                                             const int &j, const int &i)
      : pack_(pack), b_(b), kji_({k, j, i}) {}

  template <typename Var_t, typename... Is>
  KOKKOS_INLINE_FUNCTION Real &operator()(const Var_t &var, Is &&...idxs) const {
    static_assert(sizeof...(Is) == sizeof...(axes),
                  "number of indices passed to sub pack must match number of axes.");
    Kokkos::Array<int, 3> kji = kji_;
    ([&]() { kji[axis_to_kji(axes)] += idxs; }(), ...);
    return pack_(b_, var, kji[0], kji[1], kji[2]);
  }

  template <typename Var_t, typename... Is>
  KOKKOS_INLINE_FUNCTION Real &operator()(TopologicalElement te, const Var_t &var,
                                          Is &&...idxs) const {
    static_assert(sizeof...(Is) == sizeof...(axes),
                  "number of indices passed to sub pack must match number of axes.");
    Kokkos::Array<int, 3> kji = kji_;
    ([&]() { kji[axis_to_kji(axes)] += idxs; }(), ...);
    return pack_(b_, te, var, kji[0], kji[1], kji[2]);
  }

  template <typename Var_t, typename... Is>
  KOKKOS_INLINE_FUNCTION Real &flux(TopologicalElement te, const Var_t &var,
                                    Is &&...idxs) const {
    static_assert(sizeof...(Is) == sizeof...(axes),
                  "number of indices passed to sub pack must match number of axes.");
    Kokkos::Array<int, 3> kji = kji_;
    ([&]() { kji[axis_to_kji(axes)] += idxs; }(), ...);
    return pack_.flux(b_, te, var, kji[0], kji[1], kji[2]);
  }

  template <typename V>
  KOKKOS_INLINE_FUNCTION std::size_t GetSize(const V &var) const {
    return pack_.GetSize(b_, var);
  }

  KOKKOS_INLINE_FUNCTION
  const auto &GetCoordinates() const { return pack_.GetCoordinates(b_); }

 private:
  const PackType &pack_;
  const Kokkos::Array<int, 3> kji_;
  const int b_;
};

template <typename Var_t, typename PackType, Axis... axes>
struct VarStencilSubPack_impl {
  KOKKOS_INLINE_FUNCTION VarStencilSubPack_impl(PackType &pack, const int &b,
                                                const Var_t &var, const int &k,
                                                const int &j, const int &i)
      : pack_(pack), b_(b), var_(var), kji_({k, j, i}) {}

  template <typename... Is>
  KOKKOS_INLINE_FUNCTION Real &operator()(Is &&...idxs) const {
    static_assert(sizeof...(Is) == sizeof...(axes),
                  "number of indices passed to sub pack must match number of axes.");
    Kokkos::Array<int, 3> kji = kji_;
    ([&]() { kji[axis_to_kji(axes)] += idxs; }(), ...);
    return pack_(b_, var_, kji[0], kji[1], kji[2]);
  }

  template <typename... Is>
  KOKKOS_INLINE_FUNCTION Real &operator()(TopologicalElement te, Is &&...idxs) const {
    static_assert(sizeof...(Is) == sizeof...(axes),
                  "number of indices passed to sub pack must match number of axes.");
    Kokkos::Array<int, 3> kji = kji_;
    ([&]() { kji[axis_to_kji(axes)] += idxs; }(), ...);
    return pack_(b_, te, var_, kji[0], kji[1], kji[2]);
  }

  template <typename... Is>
  KOKKOS_INLINE_FUNCTION Real &flux(TopologicalElement te, Is &&...idxs) const {
    static_assert(sizeof...(Is) == sizeof...(axes),
                  "number of indices passed to sub pack must match number of axes.");
    Kokkos::Array<int, 3> kji = kji_;
    ([&]() { kji[axis_to_kji(axes)] += idxs; }(), ...);
    return pack_.flux(b_, te, var_, kji[0], kji[1], kji[2]);
  }

  KOKKOS_INLINE_FUNCTION
  const auto &GetCoordinates() const { return pack_.GetCoordinates(b_); }

 private:
  const PackType &pack_;
  const Kokkos::Array<int, 3> kji_;
  const Var_t var_;
  const int b_;
};
} // namespace impl

template <typename... Ts>
class SparsePack;

template <typename PackType>
constexpr bool is_sparse_pack =
    is_specialization_of<base_type<PackType>, SparsePack>::value;

template <Axis axis, Axis... axes, typename Var_t, typename PackType,
          REQUIRES(is_sparse_pack<PackType>)>
KOKKOS_INLINE_FUNCTION auto SubPack(PackType &pack, const int &b, const Var_t &var,
                                    const int &k, const int &j, const int &i) {
  return VarStencilSubPack_impl<Var_t, PackType, axis, axes...>(pack, b, var, k, j, i);
}

template <Axis axis, Axis... axes, typename PackType, REQUIRES(is_sparse_pack<PackType>)>
KOKKOS_INLINE_FUNCTION auto SubPack(PackType &pack, const int &b, const int &k,
                                    const int &j, const int &i) {
  return StencilSubPack_impl<PackType, axis, axes...>(pack, b, k, j, i);
}

template <typename PackType, REQUIRES(is_sparse_pack<PackType>)>
KOKKOS_INLINE_FUNCTION auto SubPack(PackType &pack, const int &b, const int &k,
                                    const int &j, const int &i) {
  return SubPack_impl<PackType>(pack, b, k, j, i);
}

// Some convenience types to differentiate between the types of subpacks
struct SubPack0D {
  static constexpr int Naxes = 0;
};

template <Axis axis1>
struct SubPack1D {
  static constexpr int Naxes = 1;
  static constexpr Axis axis = axis1;
};

template <Axis a1, Axis a2>
struct SubPack2D {
  static constexpr int Naxes = 2;
  static constexpr Axis axis1 = a1;
  static constexpr Axis axis2 = a2;
};

template <Axis a1, Axis a2, Axis a3>
struct SubPack3D {
  static constexpr int Naxes = 3;
  static constexpr Axis axis1 = a1;
  static constexpr Axis axis2 = a2;
  static constexpr Axis axis3 = a3;
};

} // namespace parthenon
#endif // PACK_SUBPACK_HPP_
