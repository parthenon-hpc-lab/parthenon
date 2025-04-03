//========================================================================================
// (C) (or copyright) 2020-2024. Triad National Security, LLC. All rights reserved.
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
#ifndef PACK_PACK_UTILS_HPP_
#define PACK_PACK_UTILS_HPP_

#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

// SFINAE for block iter so that Sparse/SwarmPacks can work for MeshBlockData and MeshData
namespace {
template <class T, class F>
inline auto ForEachBlock(T *pmd, const std::vector<bool> &include_block, F func)
    -> decltype(T().GetBlockData(0), void()) {
  for (int b = 0; b < pmd->NumBlocks(); ++b) {
    if (include_block.size() == 0 || include_block[b]) {
      auto &pmbd = pmd->GetBlockData(b);
      func(b, pmbd.get());
    }
  }
}
template <class T, class F>
inline auto ForEachBlock(T *pmbd, const std::vector<bool> &include_block, F func)
    -> decltype(T().GetBlockPointer(), void()) {
  if (include_block.size() == 0 || include_block[0]) func(0, pmbd);
}
} // namespace

// Check data types of requested variables
namespace {
template <typename Head, typename... Tail>
struct GetDataType {
  using value = typename Head::data_type;
  static_assert(std::is_same<value, typename GetDataType<Tail...>::value>::value,
                "Types must all be the same");
};
template <typename T>
struct GetDataType<T> {
  using value = typename T::data_type;
};
} // namespace

namespace parthenon {

// Namespace in which to put variable name types that are used for indexing into
// SparsePack<[type list of variable name types]> on device
namespace variable_names {
// Struct that all variable_name types should inherit from
constexpr int ANYDIM = -1234; // ANYDIM must be a slowest-moving index
template <bool REGEX, typename T, int... NCOMP>
struct var_base_t {
  using data_type = T;

  KOKKOS_INLINE_FUNCTION
  var_base_t() : idx(0) {}

  KOKKOS_INLINE_FUNCTION
  explicit var_base_t(int idx1) : idx(idx1) {}

  /*
    for 2D:, (M, N),
    idx(m, n) = N*m + n
    for 3D: (L, M, N)
    idx(l, m, n) = (M*l + m)*N + n
                 = l*M*N + m*N + n
   */
  template <typename... Args, REQUIRES(all_implement<integral(Args...)>::value),
            REQUIRES(sizeof...(Args) == sizeof...(NCOMP))>
  KOKKOS_INLINE_FUNCTION explicit var_base_t(Args... args)
      : idx(GetIndex_(std::forward<Args>(args)...)) {
    static_assert(CheckArgs_(NCOMP...),
                  "All dimensions must be strictly positive, "
                  "except the first (slowest), which may be ANYDIM.");
  }
  virtual ~var_base_t() = default;

  // All of these are just static methods so that there is no
  // extra storage in the struct
  static std::string name() {
    PARTHENON_FAIL("Need to implement your own name method.");
    return "error";
  }
  template <int idx>
  static constexpr auto GetDim() {
    return std::get<sizeof...(NCOMP) - idx>(std::make_tuple(NCOMP...));
  }
  static std::vector<int> GetShape() { return std::vector<int>{NCOMP...}; }
  KOKKOS_INLINE_FUNCTION
  static bool regex() { return REGEX; }
  KOKKOS_INLINE_FUNCTION
  static int ndim() { return sizeof...(NCOMP); }
  KOKKOS_INLINE_FUNCTION
  static int size() { return multiply<NCOMP...>::value; }

  const int idx;

 private:
  template <typename... Tail, REQUIRES(all_implement<integral(Tail...)>::value)>
  static constexpr bool CheckArgs_(int head, Tail... tail) {
    return (... && (tail > 0));
  }
  template <class... Args>
  KOKKOS_INLINE_FUNCTION static auto GetIndex_(Args... args) {
    int idx = 0;
    (
        [&] {
          idx *= NCOMP;
          idx += args;
        }(),
        ...);
    return idx;
  }
};
template <bool REGEX, int... NCOMP>
struct base_t : public var_base_t<REGEX, Real, NCOMP...> {
  using var_base_t<REGEX, Real, NCOMP...>::var_base_t;
};
// An example variable name type that selects all variables available
// on Mesh*Data
struct any_withautoflux : public base_t<true> {
  template <class... Ts>
  KOKKOS_INLINE_FUNCTION any_withautoflux(Ts &&...args)
      : base_t<true>(std::forward<Ts>(args)...) {}
  static std::string name() { return ".*"; }
};
struct any_nonautoflux : public base_t<true> {
  template <class... Ts>
  KOKKOS_INLINE_FUNCTION any_nonautoflux(Ts &&...args)
      : base_t<true>(std::forward<Ts>(args)...) {}
  static std::string name() {
    return "^(?!" + internal_fluxname + internal_varname_seperator + ").+";
  }
};
using any = any_nonautoflux;
} // namespace variable_names

// Namespace in which to put swarm variable name types that are used for indexing into
// SwarmPack<[type list of variable name types]> on device
namespace swarm_variable_names {
template <typename T, int... NCOMP>
struct base_t : public variable_names::var_base_t<false, T, NCOMP...> {
  using variable_names::var_base_t<false, T, NCOMP...>::var_base_t;
};
} // namespace swarm_variable_names

// Sparse/Swarm pack index types which allow for relatively simple indexing into
// non-variable name type based SparsePacks/SwarmPacks (i.e. objects of type
// SparsePack<>/SwarmPack<> which are created with a vector of variable names
// and/or regexes)
class PackIdx {
 public:
  KOKKOS_INLINE_FUNCTION
  explicit PackIdx(std::size_t var_idx) : vidx(var_idx), offset(0) {}
  KOKKOS_INLINE_FUNCTION
  PackIdx(std::size_t var_idx, int off) : vidx(var_idx), offset(off) {}

  KOKKOS_INLINE_FUNCTION
  PackIdx &operator=(std::size_t var_idx) {
    vidx = var_idx;
    offset = 0;
    return *this;
  }

  KOKKOS_INLINE_FUNCTION
  std::size_t VariableIdx() { return vidx; }
  KOKKOS_INLINE_FUNCTION
  int Offset() { return offset; }

 private:
  std::size_t vidx;
  int offset;
};
// Operator overloads to make calls like `my_pack(b, my_pack_idx + 3, k, j, i)` work
template <class T, REQUIRES(std::is_integral<T>::value)>
KOKKOS_INLINE_FUNCTION PackIdx operator+(PackIdx idx, T offset) {
  return PackIdx(idx.VariableIdx(), idx.Offset() + offset);
}
template <class T, REQUIRES(std::is_integral<T>::value)>
KOKKOS_INLINE_FUNCTION PackIdx operator+(T offset, PackIdx idx) {
  return idx + offset;
}

} // namespace parthenon

#endif // PACK_PACK_UTILS_HPP_
