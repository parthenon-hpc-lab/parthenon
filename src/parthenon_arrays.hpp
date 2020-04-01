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
#include <type_traits>
#include <utility> // make_pair
#include <vector>

// Kokkos Headers
#include <Kokkos_Core.hpp>

// Parthenon++ headers
#include "kokkos_abstraction.hpp"

namespace parthenon {

#define SLC0 std::make_pair(0,1)

template <typename T, typename Layout = LayoutWrapper>
class ParArrayND {
 public:
  using index_pair_t = std::pair<size_t,size_t>;
  using device_view_t = Kokkos::View<T******,Layout,DevSpace>;

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
  { }

  // legacy functions, as requested for backwards compatibility
  // with athena++ patterns
  void NewParArrayND(int nx1) {
    d6d_ = device_view_t("ParArray1D",1,1,1,1,1,nx1);
  }
  void NewParArrayND(int nx2, int nx1) {
    d6d_ = device_view_t("ParArray2D",1,1,1,1,nx2,nx1);
  }
  void NewParArrayND(int nx3, int nx2, int nx1) {
    d6d_ = device_view_t("ParArray3D",1,1,1,nx3,nx2,nx1);
  }
  void NewParArrayND(int nx4, int nx3, int nx2, int nx1) {
    d6d_ = device_view_t("ParArray4D",1,1,nx4,nx3,nx2,nx1);
  }
  void NewParArrayND(int nx5, int nx4, int nx3, int nx2, int nx1) {
    d6d_ = device_view_t("ParArray5D",1,nx5,nx4,nx3,nx2,nx1);
  }
  void NewParArrayND(int nx6, int nx5, int nx4, int nx3, int nx2, int nx1) {
    d6d_ = device_view_t("ParArray6D",nx6,nx5,nx4,nx3,nx2,nx1);
  }

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
  KOKKOS_INLINE_FUNCTION int GetDim(const int i) const {
    assert( 0 < i && i <= 6 && "ParArrayNDs are max 6D" );
    return d6d_.extent_int(6-i);
  }

  std::vector<int> GetShape() const {
    return std::vector<int>({GetDim(6), GetDim(5), GetDim(4),
          GetDim(3), GetDim(2), GetDim(1)});
  }

  // a function to get the total size of the array
  KOKKOS_INLINE_FUNCTION int GetSize() const {
    return GetDim(1)*GetDim(2)*GetDim(3)*GetDim(4)*GetDim(5)*GetDim(6);
  }
  KOKKOS_INLINE_FUNCTION std::size_t GetSizeInBytes() const {
    return GetDim(1)*GetDim(2)*GetDim(3)*GetDim(4)*GetDim(5)*GetDim(6)*sizeof(T);
  }

  KOKKOS_FORCEINLINE_FUNCTION
  T &operator() (const int n) const {
    return d6d_(0,0,0,0,0,n);
  }
  KOKKOS_FORCEINLINE_FUNCTION
  T &operator() (const int n, const int i) const {
    return d6d_(0,0,0,0,n,i);
  }
  KOKKOS_FORCEINLINE_FUNCTION
  T &operator() (const int n, const int j, const int i) const {
    return d6d_(0,0,0,n,j,i);
  }
  KOKKOS_FORCEINLINE_FUNCTION
  T &operator() (const int n, const int k, const int j, const int i) const {
    return d6d_(0,0,n,k,j,i);
  }
  KOKKOS_FORCEINLINE_FUNCTION
  T &operator() (const int m, const int n, const int k,
                 const int j, const int i) const {
    return d6d_(0,m,n,k,j,i);
  }
  KOKKOS_FORCEINLINE_FUNCTION
  T &operator() (const int p, const int m, const int n,
                 const int k, const int j, const int i) const {
    return d6d_(p,m,n,k,j,i);
  }
  template<typename...Args>
  auto Slice(Args...args) {
    auto v = Kokkos::subview(d6d_,std::forward<Args>(args)...);
    return ParArrayND<T,typename decltype(v)::array_layout>(v);
  }

  auto SliceD(index_pair_t slc, std::integral_constant<int,6>) {
    return Slice(slc,Kokkos::ALL(),Kokkos::ALL(),
                 Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL());
  }

  auto SliceD(index_pair_t slc, std::integral_constant<int,5>) {
    return Slice(SLC0,slc,Kokkos::ALL(),
                 Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL());
  }

  auto SliceD(index_pair_t slc, std::integral_constant<int,4>) {
    return Slice(SLC0,SLC0,slc,
                 Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL());
  }

  auto SliceD(index_pair_t slc, std::integral_constant<int,3>) {
    return Slice(SLC0,SLC0,SLC0,
                 slc,Kokkos::ALL(),Kokkos::ALL());
  }

  auto SliceD(index_pair_t slc, std::integral_constant<int,2>) {
    return Slice(SLC0,SLC0,SLC0,
                 Kokkos::ALL(),slc,Kokkos::ALL());
  }

  auto SliceD(index_pair_t slc, std::integral_constant<int,1>) {
    return Slice(SLC0,SLC0,SLC0,Kokkos::ALL(),Kokkos::ALL(),slc);
  }

  // AthenaArray.InitWithShallowSlice(src,dim,indx,nvar)
  // translates into auto dest = src.SliceD<dim>(std::make_pair(indx,indx+nvar))
  template<std::size_t N = 6>
  auto SliceD(index_pair_t slc,
              std::integral_constant<int,N> ic =
              std::integral_constant<int,N>{}) {
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

  auto Get(int i) {
    return Kokkos::subview(d6d_,i,
                           Kokkos::ALL(),Kokkos::ALL(),
                           Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL());
  }
  auto Get(int j, int i) {
    return Kokkos::subview(d6d_,j,i,
                           Kokkos::ALL(),Kokkos::ALL(),
                           Kokkos::ALL(),Kokkos::ALL());
  }
  auto Get(int k, int j, int i) {
    return Kokkos::subview(d6d_,k,j,i,
                           Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL());
  }
  auto Get(int n, int k, int j, int i) {
    return Kokkos::subview(d6d_,n,k,j,i,
                           Kokkos::ALL(),Kokkos::ALL());
  }
  auto Get(int m, int n, int k, int j, int i) {
    return Kokkos::subview(d6d_,m,n,k,j,i,
                           Kokkos::ALL());
  }
  auto Get(int l, int m, int n, int k, int j, int i) {
    return Kokkos::subview(d6d_,l,m,n,k,j,i); // 0d view
  }
  auto Get() {
    return d6d_;
  }
  auto Get(std::integral_constant<int,6>) {
    return Get();
  }
  auto Get(std::integral_constant<int,5>) {
    return Get(0);
  }
  auto Get(std::integral_constant<int,4>) {
    return Get(0,0);
  }
  auto Get(std::integral_constant<int,3>) {
    return Get(0,0,0);
  }
  auto Get(std::integral_constant<int,2>) {
    return Get(0,0,0,0);
  }
  auto Get(std::integral_constant<int,1>) {
    return Get(0,0,0,0,0);
  }
  auto Get(std::integral_constant<int,0>) {
    return Get(0,0,0,0,0,0);
  }
  template<std::size_t N = 6>
  auto Get(std::integral_constant<int,N> ic =
           std::integral_constant<int,N>{}) {
    return Get(ic);
  }

 private:
  device_view_t d6d_;
};

#undef SLC0

} // namespace parthenon
#endif // PARTHENON_ARRAYS_HPP
