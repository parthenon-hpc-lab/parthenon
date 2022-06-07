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
  "ParArrayND:" + std::string(__FILE__) + ":" + std::to_string(__LINE__)

namespace parthenon {

template<bool...> struct bool_pack;
template<bool... bs> 
using all_true = std::is_same<bool_pack<bs..., true>, bool_pack<true, bs...>>;

template<class... Ts>
using are_all_integral = all_true<std::is_integral<Ts>::value...>;

template<class T> bool all_greater_than(T) {return true;}
template<class T, class Arg, class... Args> 
bool all_greater_than(T val, Arg v, Args... args) {
  return (v > val) && all_greater_than(val, args...);
}

// API designed with Data = Kokkos::View<T******> in mind
template <typename Data>
class ParArrayGeneric {
 public:
  using index_pair_t = std::pair<size_t, size_t>;
  
  ParArrayGeneric() = default;
  __attribute__((nothrow)) ParArrayGeneric(const ParArrayGeneric<Data> &t) = default;
  __attribute__((nothrow)) ~ParArrayGeneric() = default;
  __attribute__((nothrow)) ParArrayGeneric<Data> &
  operator=(const ParArrayGeneric<Data> &t) = default;
  __attribute__((nothrow)) ParArrayGeneric(ParArrayGeneric<Data> &&t) = default;
  __attribute__((nothrow)) ParArrayGeneric<Data> &
  operator=(ParArrayGeneric<Data> &&t) = default;

  KOKKOS_INLINE_FUNCTION
  explicit ParArrayGeneric(const Data &v) : data_(v) {}

  // Allow a ParArrayGeneric to be cast to its underlying Kokkos view 
  KOKKOS_FORCEINLINE_FUNCTION
  operator Data() { return data_; }
  
  template<class... Args, 
           class = typename std::enable_if<are_all_integral<Args...>::value>::type>
  ParArrayGeneric(const std::string &label, Args... args) : 
      ParArrayGeneric(label, std::make_index_sequence<Data::rank - sizeof...(Args)>{}, args...)
  {
    assert(all_greater_than(0, args...));
    static_assert(Data::rank - sizeof...(Args) >= 0);
  }

  template<class... Args, 
           class = typename std::enable_if<are_all_integral<Args...>::value>::type>
  void NewParArrayND(Args... args, const std::string &label = "ParArrayND") {
    assert(all_greater_than(0, args...));
    static_assert(Data::rank - sizeof...(Args) >= 0);
    NewParArrayND(std::make_index_sequence<Data::rank - sizeof...(Args)>{}, args..., label);
  }

  template<class... Args, 
           class = typename std::enable_if<are_all_integral<Args...>::value>::type>
  KOKKOS_FORCEINLINE_FUNCTION  
  auto Get(Args... args) const {
    static_assert(Data::rank - sizeof...(Args) >= 0);
    return Get(std::make_index_sequence<Data::rank - sizeof...(Args)>{}, args...);
  }
  
  template<class... Args, 
           class = typename std::enable_if<are_all_integral<Args...>::value>::type> 
  void Resize(Args... args) {
    static_assert(Data::rank - sizeof...(Args) >= 0);
    Resize(std::make_index_sequence<Data::rank - sizeof...(Args)>{}, args...);
  }
  
  template<class... Args, 
           class = typename std::enable_if<are_all_integral<Args...>::value>::type> 
  KOKKOS_FORCEINLINE_FUNCTION
  auto &operator() (Args... args) const {
    static_assert(Data::rank - sizeof...(Args) >= 0);
    return _operator_impl(std::make_index_sequence<Data::rank - sizeof...(Args)>{}, args...);
  }

  // function to get the label
  std::string label() const { return data_.label(); }

  // functions to get array dimensions
  KOKKOS_INLINE_FUNCTION int GetDim(const int i) const {
    assert(0 < i && i <= Data::rank);
    return data_.extent_int(Data::rank - i);
  }

  // a function to get the total size of the array
  KOKKOS_INLINE_FUNCTION int GetSize() const {
    return data_.size(); 
    // TODO(LFR) : Make sure there is no inconsistency here 
    //return GetDim(1) * GetDim(2) * GetDim(3) * GetDim(4) * GetDim(5) * GetDim(6);
  }

  template <typename MemSpace>
  auto GetMirror(MemSpace const &memspace) {
    auto mirror = Kokkos::create_mirror_view(memspace, data_);
    return ParArrayGeneric<decltype(mirror)>(mirror);
  }
  auto GetHostMirror() { return GetMirror(Kokkos::HostSpace()); }
  auto GetDeviceMirror() { return GetMirror(Kokkos::DefaultExecutionSpace()); }

  template <typename Other>
  void DeepCopy(const Other &src) {
    Kokkos::deep_copy(data_, src.Get());
  }

  template <typename MemSpace>
  auto GetMirrorAndCopy(MemSpace const &memspace) {
    auto mirror = Kokkos::create_mirror_view_and_copy(memspace, data_);
    return ParArrayGeneric<decltype(mirror)>(mirror);
  }
  auto GetHostMirrorAndCopy() { return GetMirrorAndCopy(Kokkos::HostSpace()); }
  
  template <typename... Args>
  KOKKOS_INLINE_FUNCTION auto Slice(Args... args) const {
    auto v = Kokkos::subview(data_, std::forward<Args>(args)...);
    return ParArrayGeneric<decltype(v)>(v);
  }

  // AthenaArray.InitWithShallowSlice(src,dim,indx,nvar)
  // translates into auto dest = src.SliceD<dim>(std::make_pair(indx,indx+nvar)) 
  template<std::size_t N = Data::rank> 
  auto SliceD(index_pair_t slc) const {
    static_assert(N <= Data::rank);
    static_assert(N > 0);
    return SliceD(std::make_index_sequence<Data::rank - N>{}, std::make_index_sequence<N - 1>{}, slc);  
  }

  // AthenaArray.InitWithShallowSlice(src,dim,indx,nvar)
  // translates into auto dest = src.SliceD<dim>(indx,nvar)
  template <std::size_t N = Data::rank>
  auto SliceD(const int indx, const int nvar) {
    return SliceD<N>(std::make_pair(indx, indx + nvar));
  }
  
  // Reset size to 0
  // Note: Copies of this array won't be affected
  void Reset() { data_ = Data(); }

  template <std::size_t... I>
  KOKKOS_FORCEINLINE_FUNCTION auto
  GetTemplate_impl(std::index_sequence<I...>) const {
    return Get(((void) I, 0)...);
  } 

  // call me as Get<D>();
  template <std::size_t N>
  KOKKOS_INLINE_FUNCTION auto
  Get() const {
    return GetTemplate_impl(std::make_index_sequence<Data::rank - N>{}); 
  }
  
 private:
  // The stupid void casts below are to suppress compiler warnings about 
  // an unused value. Found this trick buried deep in the gcc documentation 
  template<class... Args, std::size_t... I>
  ParArrayGeneric(const std::string &label, std::index_sequence<I...>, Args... args) 
      : data_(label, ((void) I, 1)..., args...) {}
  
  template<class... Args, std::size_t... I>
  void NewParArrayND(std::index_sequence<I...>, Args... args, const std::string& label) {
    data_ = Data(label, ((void) I, 1)..., args...); 
  }

  template<class... Args, std::size_t... I>
  KOKKOS_FORCEINLINE_FUNCTION  
  auto Get(std::index_sequence<I...>, Args... args) const { 
    return Kokkos::subview(data_, args..., ((void) I, Kokkos::ALL())...);
  }

  template<class... Args, std::size_t... I>
  void Resize(std::index_sequence<I...>, Args... args) { 
    Kokkos::resize(data_, ((void) I, 1)..., args...);
  }
  
  template<class... Args, std::size_t... I>
  KOKKOS_FORCEINLINE_FUNCTION
  auto &_operator_impl(std::index_sequence<I...>, Args... args) const {
    return data_(((void) I, 0)..., args...);
  }
  
  template<std::size_t... I, std::size_t... J>
  auto SliceD(std::index_sequence<I...>, std::index_sequence<J...>, index_pair_t slc) const {
    return Slice(((void) I, std::make_pair(0, 1))..., slc, ((void) J, Kokkos::ALL())...); 
  }

  Data data_;
};

template <typename T, typename Layout = LayoutWrapper>
using device_view_t = Kokkos::View<T ******, Layout, DevMemSpace>;
template <typename T, typename Layout = LayoutWrapper>
using device_view4_t = Kokkos::View<T ****, Layout, DevMemSpace>;

template <typename T, typename Layout = LayoutWrapper>
using host_view_t = typename device_view_t<T, Layout>::HostMirror;

template <typename T, typename Layout = LayoutWrapper>
using ParArrayND = ParArrayGeneric<device_view_t<T, Layout>>;

template <typename T, typename Layout = LayoutWrapper>
using ParArray4ND = ParArrayGeneric<device_view4_t<T, Layout>>;

template <typename T, typename Layout = LayoutWrapper>
using ParArrayHost = ParArrayGeneric<host_view_t<T, Layout>>;

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
