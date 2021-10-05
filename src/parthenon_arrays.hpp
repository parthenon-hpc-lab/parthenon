//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2021. Triad National Security, LLC. All rights reserved.
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
#ifndef PARTHENON_ARRAYS_HPP_
#define PARTHENON_ARRAYS_HPP_

#include <cassert>
#include <cstddef>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "basic_types.hpp"
#include "kokkos_abstraction.hpp"

// Macro for automatically creating a useful name
#define PARARRAY_TEMP                                                                    \
  "ParArrayNDGeneric:" + std::string(__FILE__) + ":" + std::to_string(__LINE__)

namespace parthenon {

// API designed with Data = Kokkos::View<T******> in mind
template <typename Data>
class ParArrayNDGeneric {
 public:
  using index_pair_t = std::pair<size_t, size_t>;

  ParArrayNDGeneric() = default;
  ParArrayNDGeneric(const std::string &label, int nx6, int nx5, int nx4, int nx3, int nx2,
                    int nx1) {
    assert(nx6 > 0 && nx5 > 0 && nx4 > 0 && nx3 > 0 && nx2 > 0 && nx1 > 0);
    d6d_ = Data(label, nx6, nx5, nx4, nx3, nx2, nx1);
  }
  ParArrayNDGeneric(const std::string &label, int nx5, int nx4, int nx3, int nx2,
                    int nx1) {
    assert(nx5 > 0 && nx4 > 0 && nx3 > 0 && nx2 > 0 && nx1 > 0);
    d6d_ = Data(label, 1, nx5, nx4, nx3, nx2, nx1);
  }
  ParArrayNDGeneric(const std::string &label, int nx4, int nx3, int nx2, int nx1) {
    assert(nx4 > 0 && nx3 > 0 && nx2 > 0 && nx1 > 0);
    d6d_ = Data(label, 1, 1, nx4, nx3, nx2, nx1);
  }
  ParArrayNDGeneric(const std::string &label, int nx3, int nx2, int nx1) {
    assert(nx3 > 0 && nx2 > 0 && nx1 > 0);
    d6d_ = Data(label, 1, 1, 1, nx3, nx2, nx1);
  }
  ParArrayNDGeneric(const std::string &label, int nx2, int nx1) {
    assert(nx2 > 0 && nx1 > 0);
    d6d_ = Data(label, 1, 1, 1, 1, nx2, nx1);
  }
  ParArrayNDGeneric(const std::string &label, int nx1) {
    assert(nx1 > 0);
    d6d_ = Data(label, 1, 1, 1, 1, 1, nx1);
  }
  KOKKOS_INLINE_FUNCTION
  explicit ParArrayNDGeneric(const Data &v) : d6d_(v) {}

  // legacy functions, as requested for backwards compatibility
  // with athena++ patterns
  void NewParArrayND(int nx1, const std::string &label = "ParArray1D") {
    assert(nx1 > 0);
    d6d_ = Data(label, 1, 1, 1, 1, 1, nx1);
  }
  void NewParArrayND(int nx2, int nx1, const std::string &label = "ParArray2D") {
    assert(nx2 > 0 && nx1 > 0);
    d6d_ = Data(label, 1, 1, 1, 1, nx2, nx1);
  }
  void NewParArrayND(int nx3, int nx2, int nx1, const std::string &label = "ParArray3D") {
    assert(nx3 > 0 && nx2 > 0 && nx1 > 0);
    d6d_ = Data(label, 1, 1, 1, nx3, nx2, nx1);
  }
  void NewParArrayND(int nx4, int nx3, int nx2, int nx1,
                     const std::string &label = "ParArray4D") {
    assert(nx4 > 0 && nx3 > 0 && nx2 > 0 && nx1 > 0);
    d6d_ = Data(label, 1, 1, nx4, nx3, nx2, nx1);
  }
  void NewParArrayND(int nx5, int nx4, int nx3, int nx2, int nx1,
                     const std::string &label = "ParArray5D") {
    assert(nx5 > 0 && nx4 > 0 && nx3 > 0 && nx2 > 0 && nx1 > 0);
    d6d_ = Data(label, 1, nx5, nx4, nx3, nx2, nx1);
  }
  void NewParArrayND(int nx6, int nx5, int nx4, int nx3, int nx2, int nx1,
                     const std::string &label = "ParArray6D") {
    assert(nx6 > 0 && nx5 > 0 && nx4 > 0 && nx3 > 0 && nx2 > 0 && nx1 > 0);
    d6d_ = Data(label, nx6, nx5, nx4, nx3, nx2, nx1);
  }

  __attribute__((nothrow)) ParArrayNDGeneric(const ParArrayNDGeneric<Data> &t) = default;
  __attribute__((nothrow)) ~ParArrayNDGeneric() = default;
  __attribute__((nothrow)) ParArrayNDGeneric<Data> &
  operator=(const ParArrayNDGeneric<Data> &t) = default;
  __attribute__((nothrow)) ParArrayNDGeneric(ParArrayNDGeneric<Data> &&t) = default;
  __attribute__((nothrow)) ParArrayNDGeneric<Data> &
  operator=(ParArrayNDGeneric<Data> &&t) = default;

  // function to get the label
  std::string label() const { return d6d_.label(); }

  // functions to get array dimensions
  KOKKOS_INLINE_FUNCTION int GetDim(const int i) const {
    assert(0 < i && i <= 6 && "ParArrayNDGenerics are max 6D");
    return d6d_.extent_int(6 - i);
  }

  // a function to get the total size of the array
  KOKKOS_INLINE_FUNCTION int GetSize() const {
    return GetDim(1) * GetDim(2) * GetDim(3) * GetDim(4) * GetDim(5) * GetDim(6);
  }

  template <typename MemSpace>
  auto GetMirror(MemSpace const &memspace) {
    auto mirror = Kokkos::create_mirror_view(memspace, d6d_);
    return ParArrayNDGeneric<decltype(mirror)>(mirror);
  }
  auto GetHostMirror() { return GetMirror(Kokkos::HostSpace()); }
  auto GetDeviceMirror() { return GetMirror(Kokkos::DefaultExecutionSpace()); }

  template <typename Other>
  void DeepCopy(const Other &src) {
    Kokkos::deep_copy(d6d_, src.Get());
  }

  template <typename MemSpace>
  auto GetMirrorAndCopy(MemSpace const &memspace) {
    auto mirror = Kokkos::create_mirror_view_and_copy(memspace, d6d_);
    return ParArrayNDGeneric<decltype(mirror)>(mirror);
  }
  auto GetHostMirrorAndCopy() { return GetMirrorAndCopy(Kokkos::HostSpace()); }

  // JMM: DO NOT put noexcept here. It somehow interferes with inlining
  // and the code slows down by a factor of 5.
  KOKKOS_FORCEINLINE_FUNCTION
  auto &operator()(const int n) const { return d6d_(0, 0, 0, 0, 0, n); }
  KOKKOS_FORCEINLINE_FUNCTION
  auto &operator()(const int n, const int i) const { return d6d_(0, 0, 0, 0, n, i); }
  KOKKOS_FORCEINLINE_FUNCTION
  auto &operator()(const int n, const int j, const int i) const {
    return d6d_(0, 0, 0, n, j, i);
  }
  KOKKOS_FORCEINLINE_FUNCTION
  auto &operator()(const int n, const int k, const int j, const int i) const {
    return d6d_(0, 0, n, k, j, i);
  }
  KOKKOS_FORCEINLINE_FUNCTION
  auto &operator()(const int m, const int n, const int k, const int j,
                   const int i) const {
    return d6d_(0, m, n, k, j, i);
  }
  KOKKOS_FORCEINLINE_FUNCTION
  auto &operator()(const int p, const int m, const int n, const int k, const int j,
                   const int i) const {
    return d6d_(p, m, n, k, j, i);
  }
  template <typename... Args>
  KOKKOS_INLINE_FUNCTION auto Slice(Args... args) const {
    auto v = Kokkos::subview(d6d_, std::forward<Args>(args)...);
    return ParArrayNDGeneric<decltype(v)>(v);
  }

  // AthenaArray.InitWithShallowSlice(src,dim,indx,nvar)
  // translates into auto dest = src.SliceD<dim>(std::make_pair(indx,indx+nvar))
  // Call me as SliceD<N>(slc);
  template <std::size_t N = 6>
  auto
  SliceD(index_pair_t slc,
         std::integral_constant<int, N> ic = std::integral_constant<int, N>{}) const {
    return SliceD(slc, ic);
  }

  // AthenaArray.InitWithShallowSlice(src,dim,indx,nvar)
  // translates into auto dest = src.SliceD<dim>(indx,nvar)
  template <std::size_t N = 6>
  auto SliceD(const int indx, const int nvar,
              std::integral_constant<int, N> ic = std::integral_constant<int, N>{}) {
    return SliceD(std::make_pair(indx, indx + nvar), ic);
  }
  KOKKOS_INLINE_FUNCTION
  auto Get(const int i) const {
    assert(i >= 0);
    assert(GetSize() > 0 && "Can't take the subview of an unitialized array");
    return Kokkos::subview(d6d_, i, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(),
                           Kokkos::ALL(), Kokkos::ALL());
  }
  KOKKOS_INLINE_FUNCTION
  auto Get(const int j, const int i) const {
    assert(j >= 0 && i >= 0);
    assert(GetSize() > 0 && "Can't take the subview of an unitialized array");
    return Kokkos::subview(d6d_, j, i, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(),
                           Kokkos::ALL());
  }
  KOKKOS_INLINE_FUNCTION
  auto Get(const int k, const int j, const int i) const {
    assert(k >= 0 && j >= 0 && i >= 0);
    assert(GetSize() > 0 && "Can't take the subview of an unitialized array");
    return Kokkos::subview(d6d_, k, j, i, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
  }
  KOKKOS_INLINE_FUNCTION
  auto Get(const int n, const int k, const int j, const int i) const {
    assert(n >= 0 && k >= 0 && j >= 0 && i >= 0);
    assert(GetSize() > 0 && "Can't take the subview of an unitialized array");
    return Kokkos::subview(d6d_, n, k, j, i, Kokkos::ALL(), Kokkos::ALL());
  }
  KOKKOS_INLINE_FUNCTION
  auto Get(const int m, const int n, const int k, const int j, const int i) const {
    assert(m >= 0 && n >= 0 && k >= 0 && j >= 0 && i >= 0);
    assert(GetSize() > 0 && "Can't take the subview of an unitialized array");
    return Kokkos::subview(d6d_, m, n, k, j, i, Kokkos::ALL());
  }
  KOKKOS_INLINE_FUNCTION
  auto Get(const int l, const int m, const int n, const int k, const int j,
           const int i) const {
    assert(l >= 0 && m >= 0 && n >= 0 && k >= 0 && j >= 0 && i >= 0);
    assert(GetSize() > 0 && "Can't take the subview of an unitialized array");
    return Kokkos::subview(d6d_, l, m, n, k, j, i); // 0d view
  }
  KOKKOS_INLINE_FUNCTION
  auto Get() const { return d6d_; }
  // Resize View while preserving contents for shared extent
  void Resize(const int nx1) { Kokkos::resize(d6d_, 1, 1, 1, 1, 1, nx1); }
  void Resize(const int nx2, const int nx1) {
    Kokkos::resize(d6d_, 1, 1, 1, 1, nx2, nx1);
  }
  void Resize(const int nx3, const int nx2, const int nx1) {
    Kokkos::resize(d6d_, 1, 1, 1, nx3, nx2, nx1);
  }
  void Resize(const int nx4, const int nx3, const int nx2, const int nx1) {
    Kokkos::resize(d6d_, 1, 1, nx4, nx3, nx2, nx1);
  }
  void Resize(const int nx5, const int nx4, const int nx3, const int nx2, const int nx1) {
    Kokkos::resize(d6d_, 1, nx5, nx4, nx3, nx2, nx1);
  }
  void Resize(const int nx6, const int nx5, const int nx4, const int nx3, const int nx2,
              const int nx1) {
    Kokkos::resize(d6d_, nx6, nx5, nx4, nx3, nx2, nx1);
  }

  // Reset size to 0
  void Reset() { Kokkos::resize(d6d_, 0, 0, 0, 0, 0, 0); }

  // call me as Get<D>();
  template <std::size_t N = 6>
  KOKKOS_INLINE_FUNCTION auto
  Get(const std::integral_constant<int, N> &ic = std::integral_constant<int, N>{}) const {
    return Get(ic);
  }

 private:
  // These functions exist to get around the fact that partial template
  // specializations are forbidden for functions within a namespace.
  // The trick then is to use tag dispatch with std::integral_constant
  // and template on std::integral_constant.
  // This trick is thanks to Daniel Holladay.
  KOKKOS_INLINE_FUNCTION
  auto Get(std::integral_constant<int, 6>) const { return Get(); }
  KOKKOS_INLINE_FUNCTION
  auto Get(std::integral_constant<int, 5>) const { return Get(0); }
  KOKKOS_INLINE_FUNCTION
  auto Get(std::integral_constant<int, 4>) const { return Get(0, 0); }
  KOKKOS_INLINE_FUNCTION
  auto Get(std::integral_constant<int, 3>) const { return Get(0, 0, 0); }
  KOKKOS_INLINE_FUNCTION
  auto Get(std::integral_constant<int, 2>) const { return Get(0, 0, 0, 0); }
  KOKKOS_INLINE_FUNCTION
  auto Get(std::integral_constant<int, 1>) const { return Get(0, 0, 0, 0, 0); }
  KOKKOS_INLINE_FUNCTION
  auto Get(std::integral_constant<int, 0>) const { return Get(0, 0, 0, 0, 0, 0); }

#define SLC0 std::make_pair(0, 1)
  KOKKOS_INLINE_FUNCTION
  auto SliceD(index_pair_t slc, std::integral_constant<int, 6>) const {
    assert(GetSize() > 0 && "Can't take the subview of an unitialized array");
    return Slice(slc, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(),
                 Kokkos::ALL());
  }
  KOKKOS_INLINE_FUNCTION
  auto SliceD(index_pair_t slc, std::integral_constant<int, 5>) const {
    assert(GetSize() > 0 && "Can't take the subview of an unitialized array");
    return Slice(SLC0, slc, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
  }
  KOKKOS_INLINE_FUNCTION
  auto SliceD(index_pair_t slc, std::integral_constant<int, 4>) const {
    assert(GetSize() > 0 && "Can't take the subview of an unitialized array");
    return Slice(SLC0, SLC0, slc, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
  }
  KOKKOS_INLINE_FUNCTION
  auto SliceD(index_pair_t slc, std::integral_constant<int, 3>) const {
    assert(GetSize() > 0 && "Can't take the subview of an unitialized array");
    return Slice(SLC0, SLC0, SLC0, slc, Kokkos::ALL(), Kokkos::ALL());
  }
  KOKKOS_INLINE_FUNCTION
  auto SliceD(index_pair_t slc, std::integral_constant<int, 2>) const {
    assert(GetSize() > 0 && "Can't take the subview of an unitialized array");
    return Slice(SLC0, SLC0, SLC0, SLC0, slc, Kokkos::ALL());
  }
  KOKKOS_INLINE_FUNCTION
  auto SliceD(index_pair_t slc, std::integral_constant<int, 1>) const {
    assert(GetSize() > 0 && "Can't take the subview of an unitialized array");
    return Slice(SLC0, SLC0, SLC0, SLC0, SLC0, slc);
  }
#undef SLC0

  Data d6d_;
};

template <typename T, typename Layout = LayoutWrapper>
using device_view_t = Kokkos::View<T ******, Layout, DevMemSpace>;

template <typename T, typename Layout = LayoutWrapper>
using host_view_t = typename device_view_t<T, Layout>::HostMirror;

template <typename T, typename Layout = LayoutWrapper>
using ParArrayND = ParArrayNDGeneric<device_view_t<T, Layout>>;

template <typename T, typename Layout = LayoutWrapper>
using ParArrayHost = ParArrayNDGeneric<host_view_t<T, Layout>>;

template <typename T>
struct FaceArray {
  ParArrayND<T> x1f, x2f, x3f;
  FaceArray() = default;
  FaceArray(const std::string &label, int ncells3, int ncells2, int ncells1)
      : x1f(label + "x1f", ncells3, ncells2, ncells1 + 1),
        x2f(label + "x2f", ncells3, ncells2 + 1, ncells1),
        x3f(label + "x3f", ncells3 + 1, ncells2, ncells1) {}
  FaceArray(const std::string &label, int ncells4, int ncells3, int ncells2, int ncells1)
      : x1f(label + "x1f", ncells4, ncells3, ncells2, ncells1 + 1),
        x2f(label + "x2f", ncells4, ncells3, ncells2 + 1, ncells1),
        x3f(label + "x3f", ncells4, ncells3 + 1, ncells2, ncells1) {}
  FaceArray(const std::string &label, int ncells5, int ncells4, int ncells3, int ncells2,
            int ncells1)
      : x1f(label + "x1f", ncells5, ncells4, ncells3, ncells2, ncells1 + 1),
        x2f(label + "x2f", ncells5, ncells4, ncells3, ncells2 + 1, ncells1),
        x3f(label + "x3f", ncells5, ncells4, ncells3 + 1, ncells2, ncells1) {}
  FaceArray(const std::string &label, int ncells6, int ncells5, int ncells4, int ncells3,
            int ncells2, int ncells1)
      : x1f(label + "x1f", ncells6, ncells5, ncells4, ncells3, ncells2, ncells1 + 1),
        x2f(label + "x2f", ncells6, ncells5, ncells4, ncells3, ncells2 + 1, ncells1),
        x3f(label + "x3f", ncells6, ncells5, ncells4, ncells3 + 1, ncells2, ncells1) {}
  __attribute__((nothrow)) ~FaceArray() = default;

  // TODO(JMM): should this be 0,1,2?
  // Should we return the reference? Or something else?
  KOKKOS_FORCEINLINE_FUNCTION
  ParArrayND<T> &Get(int i) {
    assert(1 <= i && i <= 3);
    if (i == 1) return (x1f);
    if (i == 2)
      return (x2f);
    else
      return (x3f); // i == 3
  }
  template <typename... Args>
  KOKKOS_FORCEINLINE_FUNCTION T &operator()(int dir, Args... args) const {
    assert(1 <= dir && dir <= 3);
    if (dir == 1) return x1f(std::forward<Args>(args)...);
    if (dir == 2)
      return x2f(std::forward<Args>(args)...);
    else
      return x3f(std::forward<Args>(args)...); // i == 3
  }
};

// this is for backward compatibility with Athena++ functionality
using FaceField = FaceArray<Real>;

template <typename T>
struct EdgeArray {
  ParArrayND<Real> x1e, x2e, x3e;
  EdgeArray() = default;
  EdgeArray(const std::string &label, int ncells3, int ncells2, int ncells1)
      : x1e(label + "x1e", ncells3 + 1, ncells2 + 1, ncells1),
        x2e(label + "x2e", ncells3 + 1, ncells2, ncells1 + 1),
        x3e(label + "x3e", ncells3, ncells2 + 1, ncells1 + 1) {}
  EdgeArray(const std::string &label, int ncells4, int ncells3, int ncells2, int ncells1)
      : x1e(label + "x1e", ncells4, ncells3 + 1, ncells2 + 1, ncells1),
        x2e(label + "x2e", ncells4, ncells3 + 1, ncells2, ncells1 + 1),
        x3e(label + "x3e", ncells4, ncells3, ncells2 + 1, ncells1 + 1) {}
  EdgeArray(const std::string &label, int ncells5, int ncells4, int ncells3, int ncells2,
            int ncells1)
      : x1e(label + "x1e", ncells5, ncells4, ncells3 + 1, ncells2 + 1, ncells1),
        x2e(label + "x2e", ncells5, ncells4, ncells3 + 1, ncells2, ncells1 + 1),
        x3e(label + "x3e", ncells5, ncells4, ncells3, ncells2 + 1, ncells1 + 1) {}
  EdgeArray(const std::string &label, int ncells6, int ncells5, int ncells4, int ncells3,
            int ncells2, int ncells1)
      : x1e(label + "x1e", ncells6, ncells5, ncells4, ncells3 + 1, ncells2 + 1, ncells1),
        x2e(label + "x2e", ncells6, ncells5, ncells4, ncells3 + 1, ncells2, ncells1 + 1),
        x3e(label + "x3e", ncells6, ncells5, ncells4, ncells3, ncells2 + 1, ncells1 + 1) {
  }
  __attribute__((nothrow)) ~EdgeArray() = default;
};

// backwards compatibility with Athena++ functionality
using EdgeField = EdgeArray<Real>;

} // namespace parthenon

#endif // PARTHENON_ARRAYS_HPP_
