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

// C++ headers
#include <cassert>
#include <cstddef> // size_t
#include <string>
#include <type_traits>
#include <utility> // make_pair
#include <vector>

// Kokkos Headers
#include <Kokkos_Core.hpp>

// Parthenon++ headers
#include "kokkos_abstraction.hpp"

// Macro for automatically creating a useful name
#define PARARRAY_TEMP\
  "ParArrayNDGeneric:"+std::string(__FILE__)+":"+std::to_string(__LINE__)

namespace parthenon {

// API designed with Data = Kokkos::View<T******> in mind
template <typename Data>
class ParArrayNDGeneric {
 public:
  using index_pair_t = std::pair<size_t,size_t>;

  KOKKOS_INLINE_FUNCTION
  ParArrayNDGeneric() = default;
  ParArrayNDGeneric(const std::string& label,
                      int nx6, int nx5, int nx4, int nx3, int nx2, int nx1) {
    assert( nx6 > 0 && nx5 > 0 && nx4 > 0 && nx3 > 0 && nx2 > 0 && nx1 > 0 );
    d6d_ = Data(label,nx6,nx5,nx4,nx3,nx2,nx1);
  }
  ParArrayNDGeneric(const std::string& label,
             int nx5, int nx4, int nx3, int nx2, int nx1) {
    assert( nx5 > 0 && nx4 > 0 && nx3 > 0 && nx2 > 0 && nx1 > 0 );
    d6d_ = Data(label, 1, nx5, nx4, nx3, nx2, nx1);
  }
  ParArrayNDGeneric(const std::string& label,
             int nx4, int nx3, int nx2, int nx1) {
    assert( nx4 > 0 && nx3 > 0 && nx2 > 0 && nx1 > 0 );
    d6d_ = Data(label, 1, 1, nx4, nx3, nx2, nx1);
  }
  ParArrayNDGeneric(const std::string& label, int nx3, int nx2, int nx1) {
    assert( nx3 > 0 && nx2 > 0 && nx1 > 0 );
    d6d_ = Data(label, 1, 1, 1, nx3, nx2, nx1);
  }
  ParArrayNDGeneric(const std::string& label, int nx2, int nx1) {
    assert( nx2 > 0 && nx1 > 0 );
    d6d_ = Data(label,1,1,1,1,nx2,nx1);
  }
  ParArrayNDGeneric(const std::string& label, int nx1) {
    assert( nx1 > 0 );
    d6d_ = Data(label, 1, 1, 1, 1, 1, nx1);
  }
  KOKKOS_INLINE_FUNCTION
  explicit ParArrayNDGeneric(const Data& v)
    : d6d_(v)
  {}

  // legacy functions, as requested for backwards compatibility
  // with athena++ patterns
  void NewParArrayND(int nx1,
                     const std::string& label="ParArray1D") {
    assert( nx1 > 0 );
    d6d_ = Data(label,1,1,1,1,1,nx1);
  }
  void NewParArrayND(int nx2, int nx1,
                     const std::string& label="ParArray2D") {
    assert( nx2 > 0 && nx1 > 0 );
    d6d_ = Data(label,1,1,1,1,nx2,nx1);
  }
  void NewParArrayND(int nx3, int nx2, int nx1,
                     const std::string& label="ParArray3D") {
    assert( nx3 > 0 && nx2 > 0 && nx1 > 0 );
    d6d_ = Data(label,1,1,1,nx3,nx2,nx1);
  }
  void NewParArrayND(int nx4, int nx3, int nx2, int nx1,
                     const std::string& label="ParArray4D") {
    assert( nx4 > 0 && nx3 > 0 && nx2 > 0 && nx1 > 0 );
    d6d_ = Data(label,1,1,nx4,nx3,nx2,nx1);
  }
  void NewParArrayND(int nx5, int nx4, int nx3, int nx2, int nx1,
                     const std::string& label="ParArray5D") {
    assert( nx5 > 0 && nx4 > 0 && nx3 > 0 && nx2 > 0 && nx1 > 0 );
    d6d_ = Data(label,1,nx5,nx4,nx3,nx2,nx1);
  }
  void NewParArrayND(int nx6, int nx5, int nx4, int nx3, int nx2, int nx1,
                     const std::string& label="ParArray6D") {
    assert( nx6 > 0 && nx5 > 0 && nx4 > 0 && nx3 > 0 && nx2 > 0 && nx1 > 0 );
    d6d_ = Data(label,nx6,nx5,nx4,nx3,nx2,nx1);
  }

  KOKKOS_INLINE_FUNCTION __attribute__((nothrow))
  ParArrayNDGeneric(const ParArrayNDGeneric<Data>& t) = default;

  KOKKOS_INLINE_FUNCTION __attribute__((nothrow))
  ~ParArrayNDGeneric() = default;

  KOKKOS_INLINE_FUNCTION __attribute__((nothrow))
  ParArrayNDGeneric<Data>
  &operator= (const ParArrayNDGeneric<Data> &t) = default;

  KOKKOS_INLINE_FUNCTION __attribute__((nothrow))
  ParArrayNDGeneric(ParArrayNDGeneric<Data>&& t) = default;

  KOKKOS_INLINE_FUNCTION __attribute__((nothrow))
  ParArrayNDGeneric<Data> &operator= (ParArrayNDGeneric<Data> &&t) = default;

  // function to get the label
  std::string label() const {
    return d6d_.label();
  }

  // functions to get array dimensions
  KOKKOS_INLINE_FUNCTION int GetDim(const int i) const {
    assert( 0 < i && i <= 6 && "ParArrayNDGenerics are max 6D" );
    return d6d_.extent_int(6-i);
  }

  // a function to get the total size of the array
  KOKKOS_INLINE_FUNCTION int GetSize() const {
    return GetDim(1)*GetDim(2)*GetDim(3)*GetDim(4)*GetDim(5)*GetDim(6);
  }

  // TODO(JMM): expose wrapper for create_mirror_view_and_copy?
  template<typename MemSpace>
  auto GetMirror(MemSpace const& memspace) {
    auto mirror = Kokkos::create_mirror_view(memspace,d6d_);
    return ParArrayNDGeneric<decltype(mirror)>(mirror);
  }
  auto GetHostMirror() {
    return GetMirror(Kokkos::HostSpace());
  }
  auto GetDeviceMirror() {
    return GetMirror(Kokkos::DefaultExecutionSpace());
  }

  template<typename Other>
  void DeepCopy(const Other& src) {
    Kokkos::deep_copy(d6d_,src.Get());
  }

  // JMM: DO NOT put noexcept here. It somehow interferes with inlining
  // and the code slows down by a factor of 5.
  KOKKOS_FORCEINLINE_FUNCTION
  auto &operator() (const int n) const {
    return d6d_(0,0,0,0,0,n);
  }
  KOKKOS_FORCEINLINE_FUNCTION
  auto &operator() (const int n, const int i) const {
    return d6d_(0,0,0,0,n,i);
  }
  KOKKOS_FORCEINLINE_FUNCTION
  auto &operator() (const int n, const int j, const int i) const {
    return d6d_(0,0,0,n,j,i);
  }
  KOKKOS_FORCEINLINE_FUNCTION
  auto &operator() (const int n, const int k, const int j, const int i) const {
    return d6d_(0,0,n,k,j,i);
  }
  KOKKOS_FORCEINLINE_FUNCTION
  auto &operator() (const int m, const int n, const int k,
                    const int j, const int i) const {
    return d6d_(0,m,n,k,j,i);
  }
  KOKKOS_FORCEINLINE_FUNCTION
  auto &operator() (const int p, const int m, const int n,
                    const int k, const int j, const int i) const {
    return d6d_(p,m,n,k,j,i);
  }
  template<typename...Args>
  KOKKOS_INLINE_FUNCTION
  auto Slice(Args...args) const {
    auto v = Kokkos::subview(d6d_,std::forward<Args>(args)...);
    return ParArrayNDGeneric<decltype(v)>(v);
  }

  // AthenaArray.InitWithShallowSlice(src,dim,indx,nvar)
  // translates into auto dest = src.SliceD<dim>(std::make_pair(indx,indx+nvar))
  // Call me as SliceD<N>(slc);
  template<std::size_t N = 6>
  auto SliceD(index_pair_t slc,
              std::integral_constant<int,N> ic =
              std::integral_constant<int,N>{}) const {
    return SliceD(slc, ic);
  }

  // AthenaArray.InitWithShallowSlice(src,dim,indx,nvar)
  // translates into auto dest = src.SliceD<dim>(indx,nvar)
  template<std::size_t N = 6>
  auto SliceD(const int indx, const int nvar,
              std::integral_constant<int,N> ic =
              std::integral_constant<int,N>{}) {
    return SliceD(std::make_pair(indx,indx+nvar),ic);
  }
  KOKKOS_INLINE_FUNCTION
  auto Get(const int i) const {
    assert( i >= 0 );
    return Kokkos::subview(d6d_,i,
                           Kokkos::ALL(),Kokkos::ALL(),
                           Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL());
  }
  KOKKOS_INLINE_FUNCTION
  auto Get(const int j, const int i) const {
    assert( j >= 0 && i >= 0 );
    return Kokkos::subview(d6d_,j,i,
                           Kokkos::ALL(),Kokkos::ALL(),
                           Kokkos::ALL(),Kokkos::ALL());
  }
  KOKKOS_INLINE_FUNCTION
  auto Get(const int k, const int j, const int i) const {
    assert( k >= 0 && j >= 0 && i >= 0 );
    return Kokkos::subview(d6d_,k,j,i,
                           Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL());
  }
  KOKKOS_INLINE_FUNCTION
  auto Get(const int n, const int k, const int j, const int i) const {
    assert( n >= 0 && k >= 0 && j >= 0 && i >= 0 );
    return Kokkos::subview(d6d_,n,k,j,i,
                           Kokkos::ALL(),Kokkos::ALL());
  }
  KOKKOS_INLINE_FUNCTION
  auto Get(const int m, const int n, const int k,
           const int j, const int i) const {
    assert( m >= 0 && n >= 0 && k >= 0 && j >= 0 && i >= 0 );
    return Kokkos::subview(d6d_,m,n,k,j,i,
                           Kokkos::ALL());
  }
  KOKKOS_INLINE_FUNCTION
  auto Get(const int l, const int m, const int n,
           const int k, const int j, const int i) const {
    assert( l >= 0 && m >= 0 && n >= 0 && k >= 0 && j >= 0 && i >= 0 );
    return Kokkos::subview(d6d_,l,m,n,k,j,i); // 0d view
  }
  KOKKOS_INLINE_FUNCTION
  auto Get() const {
    return d6d_;
  }
  // call me as Get<D>();
  template<std::size_t N = 6>
  KOKKOS_INLINE_FUNCTION
  auto Get(const std::integral_constant<int,N>& ic =
           std::integral_constant<int,N>{}) const {
    return Get(ic);
  }

 private:

  // These functions exist to get around the fact that partial template
  // specializations are forbidden for functions within a namespace.
  // The trick then is to use tag dispatch with std::integral_constant
  // and template on std::integral_constant.
  // This trick is thanks to Daniel Holladay.
  KOKKOS_INLINE_FUNCTION
  auto Get(std::integral_constant<int,6>) const {
    return Get();
  }
  KOKKOS_INLINE_FUNCTION
  auto Get(std::integral_constant<int,5>) const {
    return Get(0);
  }
  KOKKOS_INLINE_FUNCTION
  auto Get(std::integral_constant<int,4>) const {
    return Get(0,0);
  }
  KOKKOS_INLINE_FUNCTION
  auto Get(std::integral_constant<int,3>) const {
    return Get(0,0,0);
  }
  KOKKOS_INLINE_FUNCTION
  auto Get(std::integral_constant<int,2>) const {
    return Get(0,0,0,0);
  }
  KOKKOS_INLINE_FUNCTION
  auto Get(std::integral_constant<int,1>) const {
    return Get(0,0,0,0,0);
  }
  KOKKOS_INLINE_FUNCTION
  auto Get(std::integral_constant<int,0>) const {
    return Get(0,0,0,0,0,0);
  }
  
  #define SLC0 std::make_pair(0,1)
  KOKKOS_INLINE_FUNCTION
  auto SliceD(index_pair_t slc, std::integral_constant<int,6>) const {
    return Slice(slc,Kokkos::ALL(),Kokkos::ALL(),
                 Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL());
  }
  KOKKOS_INLINE_FUNCTION
  auto SliceD(index_pair_t slc, std::integral_constant<int,5>) const {
    return Slice(SLC0,slc,Kokkos::ALL(),
                 Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL());
  }
  KOKKOS_INLINE_FUNCTION
  auto SliceD(index_pair_t slc, std::integral_constant<int,4>) const {
    return Slice(SLC0,SLC0,slc,
                 Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL());
  }
  KOKKOS_INLINE_FUNCTION
  auto SliceD(index_pair_t slc, std::integral_constant<int,3>) const {
    return Slice(SLC0,SLC0,SLC0,
                 slc,Kokkos::ALL(),Kokkos::ALL());
  }
  KOKKOS_INLINE_FUNCTION
  auto SliceD(index_pair_t slc, std::integral_constant<int,2>) const {
    return Slice(SLC0,SLC0,SLC0,
                 Kokkos::ALL(),slc,Kokkos::ALL());
  }
  KOKKOS_INLINE_FUNCTION
  auto SliceD(index_pair_t slc, std::integral_constant<int,1>) const {
    return Slice(SLC0,SLC0,SLC0,Kokkos::ALL(),Kokkos::ALL(),slc);
  }
  #undef SLC0

  Data d6d_;
};

template<typename T, typename Layout=LayoutWrapper>
using device_view_t = Kokkos::View<T******,Layout,DevSpace>;

template<typename T, typename Layout=LayoutWrapper>
using host_view_t = typename device_view_t<T,Layout>::HostMirror;

template<typename T, typename Layout=LayoutWrapper>
using ParArrayND = ParArrayNDGeneric<device_view_t<T,Layout>>;

template<typename T, typename Layout=LayoutWrapper>
using ParArrayHost = ParArrayNDGeneric<host_view_t<T,Layout>>;

} // namespace parthenon
#endif // PARTHENON_ARRAYS_HPP_
