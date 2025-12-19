//========================================================================================
// (C) (or copyright) 2025. Triad National Security, LLC. All rights reserved.
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

#ifndef TENSORS_TENSORS_HPP_
#define TENSORS_TENSORS_HPP_

#include <string>
#include <tuple>

#include "basic_types.hpp"
#include "defs.hpp"
#include "kokkos_abstraction.hpp"
#include "parthenon_arrays.hpp"
#include "utils/object_pool.hpp"

namespace parthenon {

namespace tensors {

// in this model, we carry around a host copy and a device copy of the
// tensor core
using pool_t = ObjectPool<ParArray1DRaw<Real>>;
using pool_map_t = ObjectPoolMap<ParArray1DRaw<Real>>;
using core_data_host_t = Kokkos::View<pool_t::owner_t **, LayoutWrapper, HostMemSpace>;
using core_data_device_t = Kokkos::View<pool_t::weak_t **, LayoutWrapper, DevMemSpace>;

/* TODO(JMM): If we wanted to generalize this machinery to an
   arbitrary number of tensor indices, we would need an array to
   track index ordering, but we could use variadic templates to
   express constructors and operator().
*/
class TensorCore {
 public:
  TensorCore() = default;
  TensorCore(pool_map_t &pool, const int rL, const int c, const int rR)
      : rL_(rL), c_(c), rR_(rR) {

    // data_ is a host-only object because of how the object pools manage
    // memory. Reference counting is done on host, allowing for freeing of
    // pool memory when references are no longer being used
    // Kokkos view of views, the destructor for the view of views must
    // happen on host, not device. This enforces that.
    data_host_ =
        core_data_host_t(ViewOfViewAlloc<HostMemSpace>("tensor core host"), rL, rR);
    data_device_ = core_data_device_t(ViewOfViewAlloc("tensor core device"), rL, rR);

    // construct data object 1d arrays on host, assigning memory from the pool
    for (size_t iL = 0; iL < rL; iL++) {
      for (size_t iR = 0; iR < rR; iR++) {
        data_host_(iL, iR) = pool.GetPool(c).Get();
      }
    }

    Kokkos::deep_copy(data_device_, data_host_);
  }

  KOKKOS_INLINE_FUNCTION
  Real &operator()(int iL, int ic, int iR) const { return data_device_(iL, iR)[ic]; }

  KOKKOS_INLINE_FUNCTION
  auto GetShape() const { return std::make_tuple(rL_, c_, rR_); }

  KOKKOS_INLINE_FUNCTION
  auto GetRanks() const { return std::make_pair(rL_, rR_); }

  KOKKOS_INLINE_FUNCTION
  std::size_t GetLeftRank() const { return rL_; }

  KOKKOS_INLINE_FUNCTION
  std::size_t GetRightRank() const { return rR_; }

  KOKKOS_INLINE_FUNCTION
  std::size_t GetPhysicalIndexSize() const { return c_; }

 private:
  std::size_t rL_, c_, rR_;
  core_data_host_t data_host_;
  core_data_device_t data_device_;
}; // Class TensorCore

class TensorTrain {
 public:
  using cores_device_t = ParArray1DRaw<TensorCore>;
  using cores_host_t = typename ParArray1DRaw<TensorCore>::HostMirror;

  TensorTrain(const std::string &name, const std::vector<TensorCore> &cores) {
    // Kokkos view of views, the destructor for the view of views must
    // happen on host, not device. This enforces that.
    cores_host_ = cores_host_t(ViewOfViewAlloc<HostMemSpace>(name), cores.size());
    cores_device_ = cores_device_t(ViewOfViewAlloc(name), cores.size());
    for (std::size_t i = 0; i < cores.size(); ++i) {
      cores_host_(i) = cores[i];
    }
    Kokkos::deep_copy(cores_device_, cores_host_);
  }

 private:
  cores_host_t cores_host_;
  cores_device_t cores_device_;
}; // class TensorTrain

} // namespace tensors
} // namespace parthenon

#endif // TENSORS_TENSORS_HPP_
