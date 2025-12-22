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
#include <vector>

#include "basic_types.hpp"
#include "defs.hpp"
#include "kokkos_abstraction.hpp"
#include "parthenon_arrays.hpp"
#include "utils/object_pool.hpp"

namespace parthenon {

namespace tensors {

// JMM: We need a TensorCoreHost type and a TensorCoreDevice type
// because the reference counting for the cores is done on host, and
// so a host array needs to stick around. But if you copy that array
// to device, bad things happen with Kokkos. The performant solution
// is to hang all the views in TensorCoreHost and then make
// TensorCoreDevice essentially an unmanaged view that is a "device
// context" for the host object, similar to how swarms
// work. Book-keeping happens on host. So TensorCoreHost is reference
// counted and works with the memory pool. TensorCoreDevice has device
// access.
using pool_t = ObjectPool<ParArray1DRaw<Real>>;
using pool_map_t = ObjectPoolMap<ParArray1DRaw<Real>>;
using core_data_host_t = Kokkos::View<pool_t::owner_t **, LayoutWrapper, HostMemSpace>;
using core_data_device_t = Kokkos::View<pool_t::weak_t **, LayoutWrapper, DevMemSpace>;
// I'm actually not sure why this needs to be unmanaged memory on
// device. Kokkos view of views works in other contexts, but somehow
// here, making this kokkos managed memory causes Kokkos to lock
// up. Something to do ensuring TensorCoreDevice is trivially
// copyable.
using core_data_device_unmanaged_t =
    Kokkos::View<pool_t::weak_t **, LayoutWrapper, DevMemSpace, MemUnmanaged>;
/* TODO(JMM): If we wanted to generalize this machinery to an
   arbitrary number of tensor indices, we would need an array to
   track index ordering, but we could use variadic templates to
   express constructors and operator().
*/
class TensorCoreHost;
class TensorCoreDevice {
  friend class TensorCoreHost;

 public:
  TensorCoreDevice() = default;

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
  TensorCoreDevice(const core_data_device_unmanaged_t &device_data)
      : rL_(device_data.extent(0)), c_(device_data(0, 0).extent(0)),
        rR_(device_data.extent(1)), data_device_(device_data) {}
  std::size_t rL_, c_, rR_;
  core_data_device_unmanaged_t data_device_;
}; // Class TensorCoreDevice

class TensorCoreHost {
 public:
  TensorCoreHost() = default;
  TensorCoreHost(pool_map_t &pool, const std::size_t rL, const std::size_t c,
                 const std::size_t rR)
      : rL_(rL), c_(c), rR_(rR) {
    // Kokkos view of views, the destructor for the view of views must
    // happen on host, not device. This enforces that.
    data_host_ =
        core_data_host_t(ViewOfViewAlloc<HostMemSpace>("tensor core host"), rL, rR);
    data_device_ = core_data_device_t(ViewOfViewAlloc("tensor core device"), rL, rR);

    // construct data object 1d arrays on host, assigning memory from the pool
    for (std::size_t iL = 0; iL < rL; iL++) {
      for (std::size_t iR = 0; iR < rR; iR++) {
        data_host_(iL, iR) = pool.GetPool(c).Get();
      }
    }
    Kokkos::deep_copy(data_device_, data_host_);
  }

  TensorCoreDevice GetOnDevice() const {
    return TensorCoreDevice(core_data_device_unmanaged_t(data_device_.data(), rL_, rR_));
  }

  auto GetShape() const { return std::make_tuple(rL_, c_, rR_); }
  auto GetRanks() const { return std::make_pair(rL_, rR_); }
  std::size_t GetLeftRank() const { return rL_; }
  std::size_t GetRightRank() const { return rR_; }
  std::size_t GetPhysicalIndexSize() const { return c_; }

 private:
  std::size_t rL_, c_, rR_;
  core_data_host_t data_host_;
  core_data_device_t data_device_;
}; // Class TensorCoreHost
using cores_device_t = Kokkos::View<TensorCoreDevice *, LayoutWrapper, DevMemSpace>;
// TODO(JMM): Should this be a view? Or a std::vector? Any reason to use Kokkos here?
using cores_host_t = Kokkos::View<TensorCoreHost *, LayoutWrapper, HostMemSpace>;

class TensorTrain {
 public:
  // most likely for testing purposes
  TensorTrain(const std::string &name, const std::vector<TensorCoreHost> &cores)
      : label_(name) {
    for (std::size_t i = 0; i < cores.size() - 1; ++i) {
      PARTHENON_REQUIRE_THROWS(cores[i].GetRightRank() == cores[i + 1].GetLeftRank(),
                               "Ranks agree");
    }
    // Host only view-of-views, owns the host only tensor cores
    cores_host_ =
        cores_host_t(ViewOfViewAlloc<HostMemSpace>(name + " host"), cores.size());
    // Device only view-of-views. The device context for cores is
    // non-owning, so no need for SequentialHostInit
    cores_device_ = cores_device_t(ViewOfViewAlloc(name + " device"), cores.size());
    auto cores_device_hm = Kokkos::create_mirror_view(cores_device_);
    for (std::size_t i = 0; i < cores.size(); ++i) {
      cores_host_(i) = cores[i];
      cores_device_hm(i) = cores_host_(i).GetOnDevice();
    }
    Kokkos::deep_copy(cores_device_, cores_device_hm);
  }

  // Evaluates the tensor train and returns the dense array it
  // represents as a Kokkos view. This is mostly for debugging!
  // TODO(JMM): I am giving up on doing this generically. It's not
  // worth it.
  auto ToDenseArray3D() const {
    // Jump through a lot of stupid hoops to generate a ParArrayND
    // TODO(JMM): Only works for howevery many dims ParArrayND supports
    PARTHENON_REQUIRE_THROWS(
        cores_host_.size() <= 3,
        "The dense object must have less dimensions than the output array");
    ParArrayND<Real> out(
        "dense version of " + label_, cores_host_[0].GetPhysicalIndexSize(),
        cores_host_[1].GetPhysicalIndexSize(), cores_host_[2].GetPhysicalIndexSize());
    par_for(
        PARTHENON_AUTO_LABEL, 0, cores_host_[0].GetPhysicalIndexSize() - 1, 0,
        cores_host_[1].GetPhysicalIndexSize() - 1, 0,
        cores_host_[2].GetPhysicalIndexSize() - 1,
        KOKKOS_CLASS_LAMBDA(const int c0, const int c1, const int c2) {
          out(c0, c1, c2) = 0;
          for (std::size_t r01 = 0; r01 < cores_device_(0).GetRightRank(); ++r01) {
            for (std::size_t r12 = 0; r12 < cores_device_(1).GetRightRank(); ++r12) {
              out(c0, c1, c2) += cores_device_(0)(0, c0, r01) *
                                 cores_device_(1)(r01, c1, r12) *
                                 cores_device_(2)(r12, c2, 0);
            }
          }
        });
    return out;
  }

  auto label() const { return label_; };

 private:
  std::string label_;
  cores_host_t cores_host_;
  cores_device_t cores_device_;
}; // class TensorTrain

} // namespace tensors
} // namespace parthenon

#endif // TENSORS_TENSORS_HPP_
