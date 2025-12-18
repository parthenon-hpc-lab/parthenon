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
using core_data_host_t =
    Kokkos::View<pool_t::owner_t **, LayoutWrapper, HostMemSpace>;
using core_data_device_t =
    Kokkos::View<pool_t::weak_t **, LayoutWrapper, DevMemSpace>;
using pool_map_t = std::map<std::size_t, pool_t>;

class TensorCore {

  TensorCore(pool_map_t &pool, const int rL, const int rR, const int c)
      : rL_(rL), rR_(rR), c_(c) {

    // data_ is a host-only object because of how the object pools manage
    // memory. Reference counting is done on host, allowing for freeing of
    // pool memory when references are no longer being used
    data_host_ = core_data_host_t("tensor core host", rL, rR);
    data_device_ = core_data_device_t("tensor core device", rL, rR);

    // construct data object 1d arrays on host, assigning memory from the pool
    for (size_t iL = 0; iL < rL; iL++) {
      for (size_t iR = 0; iR < rR; iR++) {
        data_host_(iL, iR) = pool.at(c).Get();
      }
    }

    Kokkos::deep_copy(data_device_, data_host_);
  }

  KOKKOS_INLINE_FUNCTION
  Real &operator()(int iL, int ic, int iR) { return data_device_(iL, iR)[ic]; }

 private:
  std::size_t rL_, rR_, c_;
  core_data_host_t data_host_;
  core_data_device_t data_device_;

}; // Class TensorCore

} // namespace tensors

} // namespace parthenon

#endif // TENSORS_TENSORS_HPP_
