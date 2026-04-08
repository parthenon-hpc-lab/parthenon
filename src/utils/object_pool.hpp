//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2022-2026 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2022-2026. Triad National Security, LLC. All rights reserved.
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

#ifndef UTILS_OBJECT_POOL_HPP_
#define UTILS_OBJECT_POOL_HPP_

#include "utils/concepts_lite.hpp"
#include <math.h>

#include <cstdint>
#include <iostream>
#include <memory>
#include <sstream>
#include <stack>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <utils/error_checking.hpp>

#include <Kokkos_Core.hpp>

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
  using KEY_T = uint64_t;
  std::function<T(ObjectPool *)> get_resource_;
  std::stack<weak_t> available_;
  std::unordered_map<KEY_T, std::pair<weak_t, int>> inuse_;
  static const KEY_T default_key_ = KEY_T();
  KEY_T keyc_;

 public:
  template <class... Ts>
  explicit ObjectPool(std::function<T(ObjectPool *)> get_resource)
      : get_resource_(get_resource), available_(), inuse_(), keyc_(default_key_) {}

  weak_t Get();

  void PrintStatistics() const {
    std::cout << available_.size() << " unused objects." << std::endl;
    std::cout << inuse_.size() << " used objects." << std::endl;
  }

  auto NumBuffersInPool() const { return inuse_.size() + available_.size(); }
  auto NumBuffersInUse() const { return inuse_.size(); }

  std::uint64_t SizeInBytes() const {
    constexpr std::uint64_t datum_size = sizeof(typename base_t::value_type);
    std::uint64_t object_size = 0;
    if (inuse_.size() > 0)
      object_size = inuse_.begin()->second.first.size();
    else if (available_.size() > 0)
      object_size = available_.top().size();
    return datum_size * object_size * (inuse_.size() + available_.size());
  }

  // This should be used with care since it can't generically be
  // checked that the input object has the same size as other objects
  // in the pool
  void AddFreeObjectToPool(const T &in) { available_.push(in); }
  void AddFreeObjectToPool(T &&in) { available_.emplace(in); }

  void Clear() {
    // normalize by moving everything into the stack
    for (auto &[k, v] : inuse_) {
      v.first.pool_ = nullptr;
      Free(v.first);
    }
    // walk through the stack, disable reference counting, kill the
    // buffer
    while (!available_.empty()) {
      available_.top().pool_ = nullptr;
      available_.pop();
    }
  }

  ~ObjectPool() { Clear(); }

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
  KOKKOS_IMPL_HOST_FUNCTION explicit weak_t(Ts &&...args)
      : T(std::forward<Ts>(args)...), key_(default_key_) {}

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
    KOKKOS_IF_ON_HOST(if (weak_t::pool_ != nullptr) {
      (*weak_t::pool_).ReferenceCountedFree(*this);
    }) // NOLINT
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
  explicit owner_t(const weak_t &in) : weak_t(in) {
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

/*
  TODO(JMM): Currently the key type here is always size_t. It can
  easily be extended to vector types such as std::tuple if needed,
  but I didn't bother since we don't need that right now.

  This means that right now, T really needs to be a 1D Kokkos view
  under the hood.

  Also note this is not thread safe, so will need to be updated if we
  ever worry about that.
 */
template <typename T>
  requires(KokkosView<T>)
class ObjectPoolMap {
 public:
  using pool_t = ObjectPool<T>;
  using map_t = std::unordered_map<std::size_t, pool_t>;
  using owner_t = typename pool_t::owner_t;
  using weak_t = typename pool_t::weak_t;
  using data_t = typename T::data_type;

  auto &GetPool(const std::size_t shape) {
    if (!Contains(shape)) {
      std::stringstream msg;
      msg << "ObjectPoolMap must contain an ObjectPool " << "for objects of shape "
          << shape << "!" << std::endl;
      PARTHENON_THROW(msg);
    }
    return map_.at(shape);
  }
  auto GetBuffer(const std::size_t shape) { return GetPool(shape).Get(); }
  // Sometimes the compiler complains about requiring a static cast
  // when going from weak_t to owner_t. This just is syntatic sugar
  // for that cast.
  auto GetOwningBuffer(const std::size_t shape) {
    return static_cast<owner_t>(GetBuffer(shape));
  }
  // TODO(JMM): This assumes the pool is of a 1D Kokkos view-like object
  void AddPool(const std::size_t shape, const std::size_t chunk_size) {
    static_assert(
        std::is_pointer_v<std::remove_reference_t<data_t>> &&
            !std::is_pointer_v<std::remove_pointer_t<std::remove_reference_t<data_t>>>,
        "Underlying view must be 1D");
    if (map_.count(shape) > 0) return;
    // This lambda is called whenever a buffer is requested but no
    // buffers remain in the pool
    auto allocation_strategy = [shape, chunk_size](ObjectPool<T> *pool) {
      static std::size_t counter = 0; // per shape/chunk size. NOT thread safe.
      const auto pool_size = shape * chunk_size;
      auto label = "pool buffer " + std::to_string(counter++) + " of size " +
                   std::to_string(shape) + " x " + std::to_string(chunk_size);
      T chunk(label, pool_size);
      for (std::size_t i = 1; i < chunk_size; ++i) {
        pool->AddFreeObjectToPool(T(chunk, std::make_pair(i * shape, (i + 1) * shape)));
      }
      return T(chunk, std::make_pair(static_cast<std::size_t>(0), shape));
    };
    map_.emplace(shape, ObjectPool<T>(allocation_strategy));
  }
  // TODO(JMM): This assumes the pool is of a 1D Kokkos view-like object
  void AddFreeObjectsToPool(const std::size_t shape, const std::size_t nobjs) {
    static_assert(
        std::is_pointer_v<std::remove_reference_t<data_t>> &&
            !std::is_pointer_v<std::remove_pointer_t<std::remove_reference_t<data_t>>>,
        "Underlying view must be 1D");
    auto &pool = GetPool(shape);
    const auto pool_size = shape * nobjs;
    auto label =
        "pool buffer of size " + std::to_string(shape) + " x " + std::to_string(nobjs);
    T chunk(label, pool_size);
    for (int i = 0; i < nobjs; ++i) {
      pool.AddFreeObjectToPool(T(chunk, std::make_pair(i * shape, (i + 1) * shape)));
    }
  }
  void Clear() {
    for (auto &[k, p] : map_) {
      p.Clear();
    }
  }
  auto &GetMap() const { return map_; }
  bool Contains(const std::size_t shape) const { return map_.count(shape) > 0; }

 private:
  map_t map_;
};

} // namespace parthenon

#endif // UTILS_OBJECT_POOL_HPP_
