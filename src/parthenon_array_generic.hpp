//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2022 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2022. Triad National Security, LLC. All rights reserved.
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
#ifndef PARTHENON_ARRAY_GENERIC_HPP_
#define PARTHENON_ARRAY_GENERIC_HPP_

#include <string>
#include <type_traits>
#include <utility>

#include "utils/concepts_lite.hpp"

namespace parthenon {

namespace impl {
template <class T>
bool all_greater_than(T) {
  return true;
}
template <class T, class Arg, class... Args>
bool all_greater_than(T val, Arg v, Args... args) {
  return (v > val) && all_greater_than(val, args...);
}
} // namespace impl

using namespace impl;

struct empty_state_t {
  // LFR: We can't easily have virtual destructors on device, see Kokkos issue #1591.
  // It is non-ideal for there not to be a virtual destructor on this and other
  // state structs that ParArrayGeneric ends up inheriting from, but it is unlikely
  // that there will be a need to use these with runtime polymorphism.
  // KOKKOS_INLINE_FUNCTION
  // virtual ~empty_state_t() {};
};

// The State class should be small and contain metadata that might be
// useful to store along with the array that will be (const) available
// on device
template <typename Data, typename State = empty_state_t,
          class = ENABLEIF(implements<kokkos_view(Data)>::value)>
class ParArrayGeneric : public State {
 public:
  using index_pair_t = std::pair<size_t, size_t>;
  using base_t = Data;
  using state_t = State;
  using HostMirror = ParArrayGeneric<typename Data::HostMirror, State>;
  using host_mirror_type = HostMirror;
  using value_type = typename Data::value_type; // To conform to vector and View types

  ParArrayGeneric() = default;
  __attribute__((nothrow)) ~ParArrayGeneric() = default;
  __attribute__((nothrow)) ParArrayGeneric(const ParArrayGeneric &t) = default;
  __attribute__((nothrow)) ParArrayGeneric &operator=(const ParArrayGeneric &t) = default;
  __attribute__((nothrow)) ParArrayGeneric(ParArrayGeneric &&t) = default;
  __attribute__((nothrow)) ParArrayGeneric &operator=(ParArrayGeneric &&t) = default;

  KOKKOS_INLINE_FUNCTION
  explicit ParArrayGeneric(const Data &v, const State &state = State())
      : State(state), data_(v) {}

  template <class Data2, class State2>
  KOKKOS_INLINE_FUNCTION explicit ParArrayGeneric(
      const ParArrayGeneric<Data2, State2> &arr_in)
      : State(static_cast<State2>(arr_in)), data_(arr_in.data_) {}

  template <class State2, class... Ts>
  operator ParArrayGeneric<Kokkos::View<Ts...>, State2>() const {
    return ParArrayGeneric<Kokkos::View<Ts...>, State2>(
        data_, State2(static_cast<State>(*this)));
  }

  // Allow a ParArrayGeneric to be cast to any compatible Kokkos view
  template <class... Ts>
  operator Kokkos::View<Ts...>() const {
    return data_;
  }

  // Throughout this chunk of code we use std::index_sequence to fill in default
  // arguments where none were given up to the rank of the input View. The
  // basic trick look is to use the comma operator and parameter pack unpacking
  // to create a list of the required size but just containing a single value repeated.
  // For example, if we had a function that takes six arguments but we wanted to allow
  // for calling it with six or less arguments and fill in the missing arguments at the
  // beginning with a default value of VAL, we would write
  //
  // public:
  //  template <class... Ts>
  //  std::vector<int> OurFunc(Ts... args) {
  //    return OurFuncImpl(args..., std::make_index_sequence<6 - sizeof...(Ts)>{});
  //  }
  // private:
  //  template <class... Ts, std::size_t... I>
  //  std::vector<int> OurFuncImpl(Ts... args, std::index_sequence<I...>) {
  //    return {(void) I, VAL)..., args...};
  //  }
  //
  // The (void) cast is just to suppress compiler warnings about unused variables.

  // Construct with an unallocated view when no shape arguments are given
  // Rank zero arrays are a special case, so should not be set up with default
  // constructor. The first template parameter here is to get Data into the
  // immediate context of the function template so that it can be used in the
  // enable_if sfinae
  template <class D = Data, REQUIRES(D::rank > 0)>
  explicit ParArrayGeneric(const std::string & /*label*/, const State &state = State())
      : State(state), data_() {}

  template <class D = Data, REQUIRES(D::rank > 0)>
  KOKKOS_INLINE_FUNCTION explicit ParArrayGeneric(const State &state)
      : State(state), data_() {}

  // Otherwise, assume leading dimensions are not given and set sizes of them to one
  template <class... Args, REQUIRES((sizeof...(Args) > 0) || (Data::rank == 0)),
            REQUIRES(implements<all_integral(Args...)>::value)>
  ParArrayGeneric(const std::string &label, Args... args)
      : ParArrayGeneric(label, State(),
                        std::make_index_sequence<Data::rank - sizeof...(Args)>{},
                        args...) {
    static_assert(Data::rank - sizeof...(Args) >= 0);
  }

  template <class... Args, REQUIRES((sizeof...(Args) > 0) || (Data::rank == 0)),
            REQUIRES(implements<all_integral(Args...)>::value)>
  ParArrayGeneric(const std::string &label, const State &state, Args... args)
      : ParArrayGeneric(label, state,
                        std::make_index_sequence<Data::rank - sizeof...(Args)>{},
                        args...) {
    assert(all_greater_than(0, args...));
    static_assert(Data::rank - sizeof...(Args) >= 0);
  }

  template <class... Args, REQUIRES(implements<all_integral(Args...)>::value)>
  void NewParArrayND(Args... args, const std::string &label = "ParArrayND") {
    assert(all_greater_than(0, args...));
    static_assert(Data::rank - sizeof...(Args) >= 0);
    NewParArrayND(std::make_index_sequence<Data::rank - sizeof...(Args)>{}, args...,
                  label);
  }

  template <class... Args, REQUIRES(implements<all_integral(Args...)>::value)>
  KOKKOS_FORCEINLINE_FUNCTION auto Get(Args... args) const {
    static_assert(Data::rank - sizeof...(Args) >= 0);
    return Get(std::make_index_sequence<Data::rank - sizeof...(Args)>{}, args...);
  }

  // call me as Get<D>();
  template <std::size_t N>
  KOKKOS_INLINE_FUNCTION auto Get() const {
    return Get_TemplateVersion_impl(std::make_index_sequence<Data::rank - N>{});
  }

  template <class... Args, REQUIRES(implements<all_integral(Args...)>::value)>
  void Resize(Args... args) {
    static_assert(Data::rank - sizeof...(Args) >= 0);
    Resize(std::make_index_sequence<Data::rank - sizeof...(Args)>{}, args...);
  }

  template <class... Args, REQUIRES(implements<all_integral(Args...)>::value)>
  KOKKOS_FORCEINLINE_FUNCTION auto &operator()(Args... args) const {
    static_assert(Data::rank - sizeof...(Args) >= 0);
    return _operator_impl(std::make_index_sequence<Data::rank - sizeof...(Args)>{},
                          args...);
  }

  // This operator is only defined for one dimensional Kokkos arrays
  template <typename I0>
  KOKKOS_INLINE_FUNCTION auto &operator[](const I0 &i0) const {
    return data_[i0];
  }

  // function to get the label
  std::string label() const { return data_.label(); }

  // functions to get array dimensions
  KOKKOS_INLINE_FUNCTION int GetDim(const int i) const {
    assert(0 < i && i <= Data::rank);
    return data_.extent_int(Data::rank - i);
  }
  template <class INTEGRAL_T>
  KOKKOS_INLINE_FUNCTION auto extent(const INTEGRAL_T i) const {
    return data_.extent(i);
  }

  template <class INTEGRAL_T>
  KOKKOS_INLINE_FUNCTION auto extent_int(const INTEGRAL_T i) const {
    return data_.extent_int(i);
  }

  // Return pointer to the underlying allocated memory
  KOKKOS_INLINE_FUNCTION auto data() const { return data_.data(); }

  KOKKOS_INLINE_FUNCTION auto &KokkosView() { return data_; }

  KOKKOS_INLINE_FUNCTION auto size() const { return data_.size(); }

  // a function to get the total size of the array
  KOKKOS_INLINE_FUNCTION int GetSize() const {
    return data_.size();
    // TODO(LFR) : Make sure there is no inconsistency here
    // return GetDim(1) * GetDim(2) * GetDim(3) * GetDim(4) * GetDim(5) * GetDim(6);
  }

  template <typename MemSpace>
  auto GetMirror(MemSpace const &memspace) {
    auto mirror = Kokkos::create_mirror_view(memspace, data_);
    return ParArrayGeneric<decltype(mirror), State>(mirror, *this);
  }
  auto GetHostMirror() { return GetMirror(Kokkos::HostSpace()); }
  auto GetDeviceMirror() { return GetMirror(Kokkos::DefaultExecutionSpace()); }

  template <typename Other>
  void DeepCopy(const Other &src) {
    Kokkos::deep_copy(data_, src.data_);
  }

  template <typename MemSpace>
  auto GetMirrorAndCopy(MemSpace const &memspace) {
    auto mirror = Kokkos::create_mirror_view_and_copy(memspace, data_);
    return ParArrayGeneric<decltype(mirror), State>(mirror, *this);
  }
  auto GetHostMirrorAndCopy() { return GetMirrorAndCopy(Kokkos::HostSpace()); }

  template <typename... Args>
  KOKKOS_INLINE_FUNCTION auto Slice(Args... args) const {
    auto v = Kokkos::subview(data_, std::forward<Args>(args)...);
    return ParArrayGeneric<decltype(v), State>(v, *this);
  }

  // translates into auto dest = src.SliceD<dim>(std::make_pair(indx,indx+nvar))
  template <std::size_t N = Data::rank>
  auto SliceD(index_pair_t slc) const {
    static_assert(N <= Data::rank);
    static_assert(N > 0);
    return SliceD(std::make_index_sequence<Data::rank - N>{},
                  std::make_index_sequence<N - 1>{}, slc);
  }

  // translates into auto dest = src.SliceD<dim>(indx,nvar)
  template <std::size_t N = Data::rank>
  auto SliceD(const int indx, const int nvar) {
    return SliceD<N>(std::make_pair(indx, indx + nvar));
  }

  // Reset size to 0
  // Note: Copies of this array won't be affected
  void Reset() { data_ = Data(); }

  KOKKOS_INLINE_FUNCTION
  bool IsAllocated() const { return data_.is_allocated(); }

  KOKKOS_INLINE_FUNCTION
  bool is_allocated() const { return data_.is_allocated(); }

  // Want to be friends with all other specializations of ParArrayGeneric
  template <class Data2, class State2, class enable_if_type>
  friend class ParArrayGeneric;

 private:
  // The stupid void casts below are to suppress compiler warnings about
  // an unused value. Found this trick buried deep in the gcc documentation
  template <class... Args, std::size_t... I>
  ParArrayGeneric(const std::string &label, const State &state, std::index_sequence<I...>,
                  Args... args)
      : State(state), data_(label, ((void)I, 1)..., args...) {}

  template <class... Args, std::size_t... I>
  void NewParArrayND(std::index_sequence<I...>, Args... args, const std::string &label) {
    data_ = Data(label, ((void)I, 1)..., args...);
  }

  template <class... Args, std::size_t... I>
  KOKKOS_FORCEINLINE_FUNCTION auto Get(std::index_sequence<I...>, Args... args) const {
    using view_t = decltype(Kokkos::subview(data_, args..., ((void)I, Kokkos::ALL())...));
    if (IsAllocated()) {
      return ParArrayGeneric<view_t, State>(
          Kokkos::subview(data_, args..., ((void)I, Kokkos::ALL())...), *this);
    } else {
      return ParArrayGeneric<view_t, State>(static_cast<State>(*this));
    }
  }

  template <std::size_t... I>
  KOKKOS_FORCEINLINE_FUNCTION auto
  Get_TemplateVersion_impl(std::index_sequence<I...>) const {
    return Get(((void)I, 0)...);
  }

  template <class... Args, std::size_t... I>
  void Resize(std::index_sequence<I...>, Args... args) {
    Kokkos::resize(data_, ((void)I, 1)..., args...);
  }

  template <class... Args, std::size_t... I>
  KOKKOS_FORCEINLINE_FUNCTION auto &_operator_impl(std::index_sequence<I...>,
                                                   Args... args) const {
    return data_(((void)I, 0)..., args...);
  }

  template <std::size_t... I, std::size_t... J>
  auto SliceD(std::index_sequence<I...>, std::index_sequence<J...>,
              index_pair_t slc) const {
    return Slice(((void)I, std::make_pair(0, 1))..., slc, ((void)J, Kokkos::ALL())...);
  }

  Data data_;

  template <class PA>
  friend inline bool UseSameResource(const PA &pa1, const PA &pa2);
};

template <class PA>
inline bool UseSameResource(const PA &pa1, const PA &pa2) {
  // This should just check to see if the underlying data pointers are the same
  // need to test how this work on device
  return pa1.data_.data() == pa2.data_.data();
}

} // namespace parthenon

// Overload utility functions in the Kokkos namespace on ParArrays so old code that
// assumed ParArrayGeneric = Kokkos::View does not need to be changed
namespace Kokkos {

template <class Space, class U, class SU>
inline auto create_mirror_view_and_copy(Space const &space,
                                        const parthenon::ParArrayGeneric<U, SU> &arr) {
  return parthenon::ParArrayGeneric<
      decltype(Kokkos::create_mirror_view_and_copy(space, static_cast<U>(arr))), SU>(
      Kokkos::create_mirror_view_and_copy(space, static_cast<U>(arr)), arr);
}

template <class U, class SU>
inline auto create_mirror_view_and_copy(const parthenon::ParArrayGeneric<U, SU> &arr) {
  return parthenon::ParArrayGeneric<
      decltype(Kokkos::create_mirror_view_and_copy(static_cast<U>(arr))), SU>(
      Kokkos::create_mirror_view_and_copy(static_cast<U>(arr)), arr);
}

template <class Space, class U, class SU>
inline auto create_mirror_view(Space const &space,
                               const parthenon::ParArrayGeneric<U, SU> &arr) {
  return parthenon::ParArrayGeneric<
      decltype(Kokkos::create_mirror_view(space, static_cast<U>(arr))), SU>(
      Kokkos::create_mirror_view(space, static_cast<U>(arr)), arr);
}

template <class U, class SU>
inline auto create_mirror_view(const parthenon::ParArrayGeneric<U, SU> &arr) {
  return parthenon::ParArrayGeneric<
      decltype(Kokkos::create_mirror_view(static_cast<U>(arr))), SU>(
      Kokkos::create_mirror_view(static_cast<U>(arr)), arr);
}

template <class Space, class U, class SU>
inline auto create_mirror(Space const &space,
                          const parthenon::ParArrayGeneric<U, SU> &arr) {
  return parthenon::ParArrayGeneric<
      decltype(Kokkos::create_mirror(space, static_cast<U>(arr))), SU>(
      Kokkos::create_mirror(space, static_cast<U>(arr)), arr);
}

template <class U, class SU>
inline auto create_mirror(const parthenon::ParArrayGeneric<U, SU> &arr) {
  return parthenon::ParArrayGeneric<decltype(Kokkos::create_mirror(static_cast<U>(arr))),
                                    SU>(Kokkos::create_mirror(static_cast<U>(arr)), arr);
}

template <class T, class U, class SU>
inline void deep_copy(const T &dest, const parthenon::ParArrayGeneric<U, SU> &src) {
  Kokkos::deep_copy(dest, static_cast<U>(src));
}

template <class T, class ST, class U>
inline void deep_copy(const parthenon::ParArrayGeneric<T, ST> &dest, const U &src) {
  Kokkos::deep_copy(static_cast<T>(dest), src);
}

template <class T, class ST, class U, class SU>
inline void deep_copy(const parthenon::ParArrayGeneric<T, ST> &dest,
                      const parthenon::ParArrayGeneric<U, SU> &src) {
  Kokkos::deep_copy(static_cast<T>(dest), static_cast<U>(src));
}

template <class Space, class T, class U, class SU>
inline void deep_copy(Space const &space, const T &dest,
                      const parthenon::ParArrayGeneric<U, SU> &src) {
  Kokkos::deep_copy(space, dest, static_cast<U>(src));
}

template <class Space, class T, class ST, class U>
inline void deep_copy(Space const &space, const parthenon::ParArrayGeneric<T, ST> &dest,
                      const U &src) {
  Kokkos::deep_copy(space, static_cast<T>(dest), src);
}

template <class Space, class T, class ST, class U, class SU>
inline void deep_copy(Space const &space, const parthenon::ParArrayGeneric<T, ST> &dest,
                      const parthenon::ParArrayGeneric<U, SU> &src) {
  Kokkos::deep_copy(space, static_cast<T>(dest), static_cast<U>(src));
}

template <class T, class ST, class... Args>
inline void resize(parthenon::ParArrayGeneric<T, ST> &arr, Args &&...args) {
  Kokkos::resize(arr.KokkosView(), std::forward<Args>(args)...);
}

} // namespace Kokkos

#endif // PARTHENON_ARRAY_GENERIC_HPP_
