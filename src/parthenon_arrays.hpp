//========================================================================================
// Parthenon++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
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

// C headers
#include<assert.h>

// C++ headers
#include <cstddef> // size_t
#include <string>
#include <tuple>
#include <type_traits>
#include <utility> // make_pair
#include <vector>

// Kokkos Headers
#include <Kokkos_Core.hpp>

// Parthenon++ headers
#include "kokkos_abstraction.hpp"

namespace parthenon {

using KokkosUnmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
using index_pair_t = std::pair<size_t,size_t>;
constexpr auto SLC0 = std::make_pair(0,1);

template <typename T, typename Layout = LayoutWrapper>
class ParArrayND {
 public:
  ParArrayND() = default;
  explicit ParArrayND(const std::string& label,
                      int nx6, int nx5, int nx4, int nx3, int nx2, int nx1)
    : d6d_(label,nx6,nx5,nx4,nx3,nx2,nx1)
  { }
  ParArrayND(const std::string& label,
             int nx5, int nx4, int nx3, int nx2, int nx1)
    : d6d_(label,1,nx5,nx4,nx3,nx2,nx1)
  { }
  ParArrayND(const std::string& label,
             int nx4, int nx3, int nx2, int nx1)
    : d6d_(label,1,1,nx4,nx3,nx2,nx1)
  { }
  ParArrayND(const std::string& label, int nx3, int nx2, int nx1)
    : d6d_(label,1,1,1,nx3,nx2,nx1)
  { }
  ParArrayND(const std::string& label, int nx2, int nx1)
    : d6d_(label,1,1,1,1,nx2,nx1)
  { }
  ParArrayND(const std::string& label, int nx1)
    : d6d_(label,1,1,1,1,1,nx1)
  { }
  ParArrayND(const Kokkos::View<T******,Layout,DevSpace>& v)
    : d6d_(v)
  {}

  KOKKOS_INLINE_FUNCTION __attribute__((nothrow))
  ParArrayND(const ParArrayND<T,Layout>& t) = default;
  KOKKOS_INLINE_FUNCTION __attribute__((nothrow))
  ~ParArrayND() = default;
  KOKKOS_INLINE_FUNCTION __attribute__((nothrow))
  ParArrayND<T,Layout> &operator= (const ParArrayND<T,Layout> &t) = default;
  KOKKOS_INLINE_FUNCTION __attribute__((nothrow))
  ParArrayND(ParArrayND<T,Layout>&& t) = default;
  KOKKOS_INLINE_FUNCTION __attribute__((nothrow))
  ParArrayND<T,Layout> &operator= (ParArrayND<T,Layout> &&t) = default;
  
  // functions to get array dimensions
  KOKKOS_INLINE_FUNCTION int GetDim1() const { return d6d_.extent_int(5); }
  KOKKOS_INLINE_FUNCTION int GetDim2() const { return d6d_.extent_int(5-1); }
  KOKKOS_INLINE_FUNCTION int GetDim3() const { return d6d_.extent_int(5-2); }
  KOKKOS_INLINE_FUNCTION int GetDim4() const { return d6d_.extent_int(5-3); }
  KOKKOS_INLINE_FUNCTION int GetDim5() const { return d6d_.extent_int(5-4); }
  KOKKOS_INLINE_FUNCTION int GetDim6() const { return d6d_.extent_int(5-5); }
  KOKKOS_INLINE_FUNCTION int GetDim(size_t i) const {
    // TODO(JMM): remove if performance cirtical
    assert( 0 < i && i <= 6 && "ParArrayNDs are max 6D" );
    switch (i) {
    case 1: return GetDim1();
    case 2: return GetDim2();
    case 3: return GetDim3();
    case 4: return GetDim4();
    case 5: return GetDim5();
    case 6: return GetDim6();
    }
    return -1;
  }

  std::vector<int> GetShape() const {
    return std::vector<int>({GetDim(6), GetDim(5), GetDim(4),
          GetDim(3), GetDim(2), GetDim(1)});
  }

  // a function to get the total size of the array
  KOKKOS_INLINE_FUNCTION int GetSize() const {
    return GetDim(1)*GetDim(2)*GetDim(3)*GetDim(4)*GetDim(5)*GetDim(6);
  }
  std::size_t GetSizeInBytes() const {
    return GetDim(1)*GetDim(2)*GetDim(3)*GetDim(4)*GetDim(5)*GetDim(6)*sizeof(T);
  }

  KOKKOS_INLINE_FUNCTION
  T &operator() (const int n) {
    return d6d_(0,0,0,0,0,n);
  }
  KOKKOS_INLINE_FUNCTION
  T &operator() (const int n) const {
    return d6d_(0,0,0,0,0,n);
  }
  KOKKOS_INLINE_FUNCTION
  T &operator() (const int n, const int i) {
    return d6d_(0,0,0,0,n,i);
  }
  KOKKOS_INLINE_FUNCTION
  T &operator() (const int n, const int i) const {
    return d6d_(0,0,0,0,n,i);
  }
  KOKKOS_INLINE_FUNCTION
  T &operator() (const int n, const int j, const int i) {
    return d6d_(0,0,0,n,j,i);
  }
  KOKKOS_INLINE_FUNCTION
  T &operator() (const int n, const int j, const int i) const {
    return d6d_(0,0,0,n,j,i);
  }
  KOKKOS_INLINE_FUNCTION
  T &operator() (const int n, const int k, const int j, const int i) {
    return d6d_(0,0,n,k,j,i);
  }
  KOKKOS_INLINE_FUNCTION
  T &operator() (const int n, const int k, const int j, const int i) const {
    return d6d_(0,0,n,k,j,i);
  }
  KOKKOS_INLINE_FUNCTION
  T &operator() (const int m, const int n, const int k, const int j, const int i) {
    return d6d_(0,m,n,k,j,i);
  }
  KOKKOS_INLINE_FUNCTION
  T &operator() (const int m, const int n, const int k, const int j, const int i) const {
    return d6d_(0,m,n,k,j,i);
  }
  KOKKOS_INLINE_FUNCTION
  T &operator() (const int p, const int m, const int n, const int k, const int j,
                 const int i) {
    return d6d_(p,m,n,k,j,i);
  }
  KOKKOS_INLINE_FUNCTION
  T &operator() (const int p, const int m, const int n, const int k, const int j,
                 const int i) const {
    return d6d_(p,m,n,k,j,i);
  }

  /*
  // This version only works for contiguous/strided layouts
  template<typename...Args>
  auto Slice(Args...args) {
    return ParArrayND<T,Kokkos::LayoutStride>(Kokkos::subview(d6d_,std::forward<Args>(args)...));
  }
  auto SliceD(index_pair_t slc, int dim=6) {
    assert( 1 <= dim && dim <= 6);
    if (dim == 6) return Slice(slc,Kokkos::ALL(),Kokkos::ALL(),
                               Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL());
    if (dim == 5) return Slice(SLC0,slc,Kokkos::ALL(),
                               Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL());
    if (dim == 4) return Slice(SLC0,SLC0,slc,
                               Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL());
    if (dim == 3) return Slice(SLC0,SLC0,SLC0,
                               slc,Kokkos::ALL(),Kokkos::ALL());
    if (dim == 2) return Slice(SLC0,SLC0,SLC0,
                               Kokkos::ALL(),slc,Kokkos::ALL());
    // dim == 1
    return Slice(SLC0,SLC0,SLC0,Kokkos::ALL(),Kokkos::ALL(),slc);
  }
  */
  // this version works for all layouts
  template<typename...Args>
  auto Slice(Args...args) {
    auto v = Kokkos::subview(d6d_,std::forward<Args>(args)...);
    return ParArrayND<T,typename decltype(v)::array_layout>(v);
  }

  auto Slice6D(index_pair_t slc) {
    return Slice(slc,Kokkos::ALL(),Kokkos::ALL(),
                 Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL());
  }
  auto Slice5D(index_pair_t slc) {
    return Slice(SLC0,slc,Kokkos::ALL(),
                 Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL());
  }
  auto Slice4D(index_pair_t slc) {
    return Slice(SLC0,SLC0,slc,
                 Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL());
  }
  auto Slice3D(index_pair_t slc) {
    return Slice(SLC0,SLC0,SLC0,
                 slc,Kokkos::ALL(),Kokkos::ALL());
  }
  auto Slice2D(index_pair_t slc) {
    return Slice(SLC0,SLC0,SLC0,
                 Kokkos::ALL(),slc,Kokkos::ALL());
  }
  auto Slice1D(index_pair_t slc) {
    return Slice(SLC0,SLC0,SLC0,Kokkos::ALL(),Kokkos::ALL(),slc);
  }

  template<std::size_t N = 6>
  auto SliceD(index_pair_t slc,
              std::integral_constant<int,N> = std::integral_constant<int,6>{});

  auto SliceD(index_pair_t slc, std::integral_constant<int,6>) {
    return Slice6D(slc);
  }
  auto SliceD(index_pair_t slc, std::integral_constant<int,5>) {
    return Slice5D(slc);
  }
  auto SliceD(index_pair_t slc, std::integral_constant<int,4>) {
    return Slice4D(slc);
  }
  auto SliceD(index_pair_t slc, std::integral_constant<int,3>) {
    return Slice3D(slc);
  }
  auto SliceD(index_pair_t slc, std::integral_constant<int,2>) {
    return Slice2D(slc);
  }
  auto SliceD(index_pair_t slc, std::integral_constant<int,1>) {
    return Slice1D(slc);
  }
  // TODO(JMM): Can we make a mirror object for ParArrayND? Probably.
  // Might not even be very hard.
  auto GetMirror() {
    return Kokkos::create_mirror(d6d_);
  }
  ParArray6D<T> Get() {
    return d6d_;
  }

 private:
  Kokkos::View<T******,Layout,DevSpace> d6d_;
};

// TODO(JMM): A std::variant would be more efficient
// and require fewer literals
// but won't work on GPU unless we use mpark's backport
// std::tuple works only with CUDA 8+ and C++14+
template<typename T, typename Layout>
using flexview_t =  std::tuple<Kokkos::View<T*,Layout,DevSpace>,
                               Kokkos::View<T**,Layout,DevSpace>,
                               Kokkos::View<T***,Layout,DevSpace>,
                               Kokkos::View<T****,Layout,DevSpace>,
                               Kokkos::View<T*****,Layout,DevSpace>,
                               Kokkos::View<T******,Layout,DevSpace>>;
template<typename T, typename Layout = LayoutWrapper>
class ParArrayFlex {
 public:
  ParArrayFlex() = default;
  template<typename...Args>
  explicit ParArrayFlex(const std::string& label, Args&&...args)
    : rank_(sizeof...(Args)) {
    constexpr int rm1 = sizeof...(Args) - 1;
    using type = typename std::tuple_element<rm1,flexview_t<T,Layout>>::type;
    std::get<rm1>(data_) = type(label,std::forward<Args>(args)...);
  }
  // TODO(JMM): This can probably be made to go away with template magic
  ParArrayFlex(const Kokkos::View<T*,Layout,DevSpace>& v)
    : rank_(1) {
    std::get<0>(data_) = v;
  }
  ParArrayFlex(const Kokkos::View<T**,Layout,DevSpace>& v)
    : rank_(2) {
    std::get<1>(data_) = v;
  }
  ParArrayFlex(const Kokkos::View<T***,Layout,DevSpace>& v)
    : rank_(3) {
    std::get<2>(data_) = v;
  }
  ParArrayFlex(const Kokkos::View<T****,Layout,DevSpace>& v)
    : rank_(4) {
    std::get<3>(data_) = v;
  }
  ParArrayFlex(const Kokkos::View<T*****,Layout,DevSpace>& v)
    : rank_(5) {
    std::get<4>(data_) = v;
  }
  ParArrayFlex(const Kokkos::View<T******,Layout,DevSpace>& v)
    : rank_(6) {
    std::get<5>(data_) = v;
  }

  KOKKOS_INLINE_FUNCTION __attribute__((nothrow))
  ParArrayFlex(const ParArrayFlex<T,Layout>& t) = default;
  KOKKOS_INLINE_FUNCTION __attribute__((nothrow))
  ~ParArrayFlex() = default;
  KOKKOS_INLINE_FUNCTION __attribute__((nothrow))
  ParArrayFlex<T,Layout> &operator= (const ParArrayFlex<T,Layout> &t) = default;
  KOKKOS_INLINE_FUNCTION __attribute__((nothrow))
  ParArrayFlex(ParArrayFlex<T,Layout>&& t) = default;
  KOKKOS_INLINE_FUNCTION __attribute__((nothrow))
  ParArrayFlex<T,Layout> &operator= (ParArrayFlex<T,Layout> &&t) = default;

  constexpr int GetRank() {
    return rank_;
  }
  
  template<std::size_t I>
  auto Get() {
    return std::get<I-1>(data_);
  }

  template<std::size_t I>
  auto GetMirror() {
    return Kokkos::create_mirror(Get<I>());
  }

  KOKKOS_INLINE_FUNCTION int GetDim(const int i) const {
    if (i <= 0 || i > rank_) return 0;
    switch (rank_) {
    case 1: return std::get<0>(data_).extent_int(0);
    case 2: return std::get<1>(data_).extent_int(2-i);
    case 3: return std::get<2>(data_).extent_int(3-i);
    case 4: return std::get<3>(data_).extent_int(4-i);
    case 5: return std::get<4>(data_).extent_int(5-i);
    case 6: return std::get<5>(data_).extent_int(6-i);
    }
  }

  std::vector<int> GetShape() const {
    return std::vector<int>({GetDim(6), GetDim(5), GetDim(4),
          GetDim(3), GetDim(2), GetDim(1)});
  }

  KOKKOS_INLINE_FUNCTION int GetSize() const {
    return GetDim(1)*GetDim(2)*GetDim(3)*GetDim(4)*GetDim(5)*GetDim(6);
  }
  std::size_t GetSizeInBytes() const {
    return GetDim(1)*GetDim(2)*GetDim(3)*GetDim(4)*GetDim(5)*GetDim(6)*sizeof(T);
  }

  // TODO(JMM): There's probably a recursive-template
  // way to write this that's better.
  KOKKOS_INLINE_FUNCTION
  T &operator() (const int i) {
    assert( 1 <= rank_ && rank_ <= 6 );
    if (rank_ == 1) return std::get<0>(data_)(i);
    if (rank_ == 2) return std::get<1>(data_)(0,i);
    if (rank_ == 3) return std::get<2>(data_)(0,0,i);
    if (rank_ == 4) return std::get<3>(data_)(0,0,0,i);
    if (rank_ == 5) return std::get<4>(data_)(0,0,0,0,i);
    if (rank_ == 6) return std::get<5>(data_)(0,0,0,0,0,i);
  }
  KOKKOS_INLINE_FUNCTION
  T &operator() (const int j, const int i) {
    assert( 2 <= rank_ && rank_ <= 6 );
    if (rank_ == 2) return std::get<1>(data_)(j,i);
    if (rank_ == 3) return std::get<2>(data_)(0,j,i);
    if (rank_ == 4) return std::get<3>(data_)(0,0,j,i);
    if (rank_ == 5) return std::get<4>(data_)(0,0,0,j,i);
    if (rank_ == 6) return std::get<5>(data_)(0,0,0,0,j,i);
  }
  KOKKOS_INLINE_FUNCTION
  T &operator() (const int k, const int j, const int i) {
    assert( 3 <= rank_ && rank_ <= 6 );
    if (rank_ == 3) return std::get<2>(data_)(k,j,i);
    if (rank_ == 4) return std::get<3>(data_)(0,k,j,i);
    if (rank_ == 5) return std::get<4>(data_)(0,0,k,j,i);
    if (rank_ == 6) return std::get<5>(data_)(0,0,0,k,j,i);
  }
  KOKKOS_INLINE_FUNCTION
  T &operator() (const int n, const int k, const int j, const int i) {
    assert( 4 <= rank_ && rank_ <= 6 );
    if (rank_ == 4) return std::get<3>(data_)(n,k,j,i);
    if (rank_ == 5) return std::get<4>(data_)(0,n,k,j,i);
    if (rank_ == 6) return std::get<5>(data_)(0,0,n,k,j,i);
  }
  KOKKOS_INLINE_FUNCTION
  T &operator() (const int m, const int n,
                 const int k, const int j, const int i) {
    assert( 5 <= rank_ && rank_ <= 6 );
    if (rank_ == 5) return std::get<4>(data_)(m,n,k,j,i);
    if (rank_ == 6) return std::get<5>(data_)(0,m,n,k,j,i);
  }
  KOKKOS_INLINE_FUNCTION
  T &operator() (const int l, const int m, const int n,
                 const int k, const int j, const int i) {
    assert( rank_ == 6 );
    return std::get<5>(data_)(l,m,n,k,j,i);
  }
  KOKKOS_INLINE_FUNCTION
  T &operator() (const int i) const {
    assert( 1 <= rank_ && rank_ <= 6 );
    if (rank_ == 1) return std::get<0>(data_)(i);
    if (rank_ == 2) return std::get<1>(data_)(0,i);
    if (rank_ == 3) return std::get<2>(data_)(0,0,i);
    if (rank_ == 4) return std::get<3>(data_)(0,0,0,i);
    if (rank_ == 5) return std::get<4>(data_)(0,0,0,0,i);
    if (rank_ == 6) return std::get<5>(data_)(0,0,0,0,0,i);
  }
  KOKKOS_INLINE_FUNCTION
  T &operator() (const int j, const int i) const {
    assert( 2 <= rank_ && rank_ <= 6 );
    if (rank_ == 2) return std::get<1>(data_)(j,i);
    if (rank_ == 3) return std::get<2>(data_)(0,j,i);
    if (rank_ == 4) return std::get<3>(data_)(0,0,j,i);
    if (rank_ == 5) return std::get<4>(data_)(0,0,0,j,i);
    if (rank_ == 6) return std::get<5>(data_)(0,0,0,0,j,i);
  }
  KOKKOS_INLINE_FUNCTION
  T &operator() (const int k, const int j, const int i) const {
    assert( 3 <= rank_ && rank_ <= 6 );
    if (rank_ == 3) return std::get<2>(data_)(k,j,i);
    if (rank_ == 4) return std::get<3>(data_)(0,k,j,i);
    if (rank_ == 5) return std::get<4>(data_)(0,0,k,j,i);
    if (rank_ == 6) return std::get<5>(data_)(0,0,0,k,j,i);
  }
  KOKKOS_INLINE_FUNCTION
  T &operator() (const int n, const int k,
                 const int j, const int i) const {
    assert( 4 <= rank_ && rank_ <= 6 );
    if (rank_ == 4) return std::get<3>(data_)(n,k,j,i);
    if (rank_ == 5) return std::get<4>(data_)(0,n,k,j,i);
    if (rank_ == 6) return std::get<5>(data_)(0,0,n,k,j,i);
  }
  KOKKOS_INLINE_FUNCTION
  T &operator() (const int m, const int n,
                 const int k, const int j, const int i) const {
    assert( 5 <= rank_ && rank_ <= 6 );
    if (rank_ == 5) return std::get<4>(data_)(m,n,k,j,i);
    if (rank_ == 6) return std::get<5>(data_)(0,m,n,k,j,i);
  }
  KOKKOS_INLINE_FUNCTION
  T &operator() (const int l, const int m, const int n,
                 const int k, const int j, const int i) const {
    assert( rank_ == 6 );
    return std::get<5>(data_)(l,m,n,k,j,i);
  }

  template<typename...Args>
  auto Slice(Args...args) {
    constexpr int ndims = sizeof...(Args);
    assert(ndims == rank_);
    return ParArrayFlex<T,Kokkos::LayoutStride>
      (Kokkos::subview(Get<ndims>(),std::forward<Args>(args)...));
  }

 private:
  int rank_;
  flexview_t<T,Layout> data_;
};

} // namespace parthenon
#endif // PARTHENON_ARRAYS_HPP
