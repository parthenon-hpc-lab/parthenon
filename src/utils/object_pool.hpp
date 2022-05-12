//========================================================================================
// (C) (or copyright) 2022. Triad National Security, LLC. All rights reserved.
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

#ifndef UTILS_MEMORY_POOL_HPP_
#define UTILS_MEMORY_POOL_HPP_

#include <Kokkos_Core.hpp>
#include <iostream>
#include <math.h>
#include <memory>
#include <stack>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace parthenon {

// Object for managing a pool of Kokkos::Views that
// have the same instantiation call signature
template <class T>
class ObjectPool {
 public:
  using base_t = T;
  // Forward declarations of pool types
  class weak_t;
  class owner_t;

 private:
  using KEY_T = unsigned long int;
  std::function<T(ObjectPool *)> get_resource_;
  std::stack<weak_t> available_;
  std::unordered_map<KEY_T, std::pair<weak_t, int>> inuse_;
  static const KEY_T default_key_ = KEY_T();
  KEY_T keyc_;

 public:
  template <class... Ts>
  ObjectPool(std::function<T(ObjectPool *)> get_resource)
      : get_resource_(get_resource), available_(), inuse_(), keyc_(default_key_) {}

  weak_t Get();

  void PrintStatistics() {
    std::cout << available_.size() << " unused objects." << std::endl;
    std::cout << inuse_.size() << " used objects." << std::endl;
  }

  // This should be used with care since it can't generically be
  // checked that the input object has the same size as other objects
  // in the pool
  void AddFreeObjectToPool(const T &in) { available_.push(in); }
  void AddFreeObjectToPool(T &&in) { available_.emplace(in); }

 private:
  bool IsValid(const weak_t &in) const { return inuse_.count(in.key_); }

  void ReferenceCountedFree(const weak_t &in) {
    if (!IsValid(in)) return;
    auto &pair = inuse_[in.key_];
    --pair.second;
    if (pair.second <= 0) {
      available_.push(pair.first);
      inuse_.erase(in.key_);
    }
  }

  void Free(const weak_t &in) {
    if (!IsValid(in)) return;
    available_.push(inuse_[in.key_].first);
    inuse_.erase(in.key_);
  }

  void AddCount(const weak_t &in) {
    if (!IsValid(in)) throw 1;
    ++inuse_[in.key_].second;
  }
};

// Lightly wraps a view of type T and holds a key that
// that can be used to query if its storage
// is in use or if it has been freed and allows
// freeing of the storage.
template <class T>
struct ObjectPool<T>::weak_t : public T {
  friend class ObjectPool;

 protected:
  template <class... ARGs>
  KOKKOS_IMPL_HOST_FUNCTION static weak_t make(int key, ARGs &&...args) {
    weak_t out(std::forward<ARGs>(args)...);
    out.key_ = key;
    return out;
  }

 public:
  template <class... Ts>
  KOKKOS_IMPL_HOST_FUNCTION weak_t(Ts &&...args)
      : T(std::forward<Ts>(args)...), key_(default_key_){};

  KOKKOS_IMPL_HOST_FUNCTION
  inline void Free() { (*pool_).Free(*this); }

  KOKKOS_IMPL_HOST_FUNCTION
  inline bool IsValid() {
    if (key_ == default_key_ || pool_ == nullptr) return false;
    return (*pool_).IsValid(*this);
  }

  KOKKOS_IMPL_HOST_FUNCTION
  inline KEY_T GetKey() const { return key_; }

  KOKKOS_DEFAULTED_FUNCTION
  ~weak_t() = default;

  KOKKOS_DEFAULTED_FUNCTION
  weak_t() = default;

  KOKKOS_DEFAULTED_FUNCTION
  weak_t(const weak_t &) = default;

  KOKKOS_DEFAULTED_FUNCTION
  weak_t(weak_t &&) = default;

  KOKKOS_DEFAULTED_FUNCTION
  weak_t &operator=(const weak_t &) = default;

  KOKKOS_DEFAULTED_FUNCTION
  weak_t &operator=(weak_t &&) = default;

  // Allow this to point at an unmanaged object of type T
  KOKKOS_FUNCTION
  weak_t &operator=(const T &in) {
    T::operator=(in);
    return *this;
  }

 protected:
  KEY_T key_ = 0;
  ObjectPool *pool_ = nullptr;
};

// Reference counted version of pool member that has ownership over a resource
// and sends it back to the pool when its destructor is called and it is the
// last owner that holds that resource. Cannot be on device
template <class T>
class ObjectPool<T>::owner_t : public ObjectPool<T>::weak_t {
 public:
  KOKKOS_DEFAULTED_FUNCTION
  owner_t() = default;

  KOKKOS_FUNCTION
  ~owner_t() noexcept {
#ifndef __CUDA_ARCH__ // host code
    if (weak_t::pool_ != nullptr) (*weak_t::pool_).ReferenceCountedFree(*this);
#endif
  }

  // Warning, the move constructors are messed up and don't copy over the weak_t
  // fields for some incomprehensible reason
  KOKKOS_IMPL_HOST_FUNCTION
  owner_t(const owner_t &in) : weak_t(in) {
    // For some reason I don't understand these don't get initialized by the call to
    // the weak_t copy ctor above, even though the T gets moved
    weak_t::key_ = in.key_;
    weak_t::pool_ = in.pool_;
    if (weak_t::pool_ != nullptr) (*weak_t::pool_).AddCount(*this);
  }

  KOKKOS_IMPL_HOST_FUNCTION
  owner_t(const weak_t &in) : weak_t(in) {
    if (weak_t::pool_ != nullptr) (*weak_t::pool_).AddCount(*this);
  }

  KOKKOS_IMPL_HOST_FUNCTION
  owner_t &operator=(const owner_t &in) { return assign(in); }

  KOKKOS_IMPL_HOST_FUNCTION
  owner_t &operator=(const weak_t &in) { return assign(in); }

  KOKKOS_IMPL_HOST_FUNCTION
  owner_t &operator=(owner_t &&in) { return assign(std::move(in)); }

  KOKKOS_IMPL_HOST_FUNCTION
  owner_t &operator=(weak_t &&in) { return assign(std::move(in)); }

 private:
  template <class TIN>
  KOKKOS_IMPL_HOST_FUNCTION owner_t &assign(TIN &&in) {
    const bool same_resource = (weak_t::key_ == in.key_) && (weak_t::pool_ == in.pool_);
    if ((weak_t::pool_ != nullptr) && !same_resource)
      (*weak_t::pool_).ReferenceCountedFree(*this);
    weak_t::key_ = in.key_;
    weak_t::pool_ = in.pool_;
    if (weak_t::pool_ != nullptr && !same_resource) (*weak_t::pool_).AddCount(*this);

    weak_t::operator=(std::forward<TIN>(in));

    return *this;
  }
};

template <class T>
typename ObjectPool<T>::weak_t ObjectPool<T>::Get() {
  weak_t out;
  if (available_.size() > 0) {
    out = available_.top();
    available_.pop();
  } else {
    out = weak_t(get_resource_(this));
  }
  // Find an unused key that is not the default key
  while (inuse_.count(++keyc_) != 0 || keyc_ == default_key_) {
  }
  // Reference count should start from zero since copy constructor
  // or assignment operator of owner_t will increment the count
  // Warning: if a weak_t object is the only one that takes a piece
  //  of memory from the pool, that memory will never be returned to
  //  the pool unless it is explicitly freed.
  inuse_[keyc_] = {out, 0};
  out.key_ = keyc_;
  out.pool_ = this;
  return out;
}

template <class T, class U>
bool UsingSameResource(const T &lhs, const U &rhs) {
  return lhs.GetKey() == rhs.GetKey();
}

} // namespace parthenon

#endif // UTILS_MEMORY_POOL_HPP_
