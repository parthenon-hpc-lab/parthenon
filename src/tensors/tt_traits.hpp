//========================================================================================
// (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
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

#ifndef TENSORS_TT_TRAITS_HPP
#define TENSORS_TT_TRAITS_HPP

#include "basic_types.hpp"
#include "kokkos_abstraction.hpp"
#include "tt_unfoldings.hpp"

namespace parthenon {
struct ManagedTag {};
struct UnmanagedTag {};

// TensorTraits collects the basic Kokkos type information used throughout the
// tensor-train code. It centralizes the device type, scalar type, layout, and
// managed/unmanaged view aliases so that the tensor data structures themselves
// can stay relatively clean. In particular, view_t<DataType, OwnershipTag>
// gives a device-space Kokkos::View, while host_view_t<DataType, OwnershipTag>
// gives the corresponding host-mirror-space view with the same ownership mode.
template <class Device,
          class RealT = Real,
          class Layout = Kokkos::LayoutRight,
          bool DFastestMoving = true>
struct TensorTraits {
  using device_type = Device;
  using execution_space = typename device_type::execution_space;
  using memory_space = typename device_type::memory_space;
  using layout = Layout;
  using real_t = RealT;
  using scratch_memory_space = Kokkos::ScratchMemorySpace<execution_space>;

  static constexpr bool d_fastest_moving = DFastestMoving;

  using host_mirror_space =
      typename Kokkos::View<real_t*, layout, memory_space>::host_mirror_space;

  template <class OwnershipTag>
  using memory_traits =
      std::conditional_t<std::is_same_v<OwnershipTag, ManagedTag>,
                         Kokkos::MemoryTraits<0>,
                         Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  template <class DataType, class OwnershipTag>
  using view_t =
      Kokkos::View<DataType, layout, memory_space,
                   memory_traits<OwnershipTag>>;

  template <class DataType, class OwnershipTag>
  using host_view_t =
      Kokkos::View<DataType, layout, host_mirror_space,
                   memory_traits<OwnershipTag>>;

  // Unfolding factory methods
  template <class CoreLike>
  static KOKKOS_FORCEINLINE_FUNCTION auto GetHorizontalUnfolding(const CoreLike &core) {
    return tensor2::horizontal_unfolding<CoreLike, false, DFastestMoving>(core);
  }

  template <class CoreLike>
  static KOKKOS_FORCEINLINE_FUNCTION auto GetHorizontalUnfolding(const CoreLike &core, int nl, int nd, int nr) {
    return tensor2::horizontal_unfolding<CoreLike, false, DFastestMoving>(core, nl, nd, nr);
  }

  template <class CoreLike>
  static KOKKOS_FORCEINLINE_FUNCTION auto GetHorizontalUnfoldingTranspose(const CoreLike &core) {
    return tensor2::horizontal_unfolding<CoreLike, true, DFastestMoving>(core);
  }

  template <class CoreLike>
  static KOKKOS_FORCEINLINE_FUNCTION auto GetHorizontalUnfoldingTranspose(const CoreLike &core, int nl, int nd, int nr) {
    return tensor2::horizontal_unfolding<CoreLike, true, DFastestMoving>(core, nl, nd, nr);
  }

  template <class CoreLike>
  static KOKKOS_FORCEINLINE_FUNCTION auto GetVerticalUnfolding(const CoreLike &core) {
    return tensor2::vertical_unfolding<CoreLike, false>(core);
  }

  template <class CoreLike>
  static KOKKOS_FORCEINLINE_FUNCTION auto GetVerticalUnfolding(const CoreLike &core, int nl, int nd, int nr) {
    return tensor2::vertical_unfolding<CoreLike, false>(core, nl, nd, nr);
  }

  template <class CoreLike>
  static KOKKOS_FORCEINLINE_FUNCTION auto GetVerticalUnfoldingTranspose(const CoreLike &core) {
    return tensor2::vertical_unfolding<CoreLike, true>(core);
  }

  template <class CoreLike>
  static KOKKOS_FORCEINLINE_FUNCTION auto GetVerticalUnfoldingTranspose(const CoreLike &core, int nl, int nd, int nr) {
    return tensor2::vertical_unfolding<CoreLike, true>(core, nl, nd, nr);
  }
};

using DefaultTTraits = TensorTraits<Kokkos::Device<DevExecSpace, DevMemSpace>, Real, Kokkos::LayoutRight, true>;
using ContiguousTTraits = TensorTraits<Kokkos::Device<DevExecSpace, DevMemSpace>, Real, Kokkos::LayoutRight, false>;

// ==============================================================================
// STORAGE POLICIES FOR TENSOR CORES
// ==============================================================================

namespace tensor2 {

// FiberView is the fundamental 1D storage unit for tensor-core data.
template <class TTraits, class OwnershipTag>
using FiberView =
    typename TTraits::template view_t<typename TTraits::real_t*, OwnershipTag>;

// FiberStorageDevice: Device-side storage policy using fiber-based (view-of-views) layout
// Extracted from TensorCoreDeviceT. This is a lightweight unmanaged descriptor
// that wraps a 2D view of fiber handles with dimensions [lr][rr], where each
// fiber has length dd.
template <class TTraits>
class FiberStorageDevice {
 public:
  using traits = TTraits;
  using real_t = typename TTraits::real_t;
  using fiber_unmanaged_t = FiberView<TTraits, UnmanagedTag>;
  using fibers_view_t = typename TTraits::template view_t<fiber_unmanaged_t**, UnmanagedTag>;

 private:
  fibers_view_t fibers;
  int lr{0}, dd{0}, rr{0};

 public:
  KOKKOS_FUNCTION
  FiberStorageDevice() = default;

  KOKKOS_FUNCTION
  FiberStorageDevice(int lr_in, int dd_in, int rr_in, const fibers_view_t &fibers_in)
      : fibers(fibers_in), lr(lr_in), dd(dd_in), rr(rr_in) {}

  KOKKOS_INLINE_FUNCTION
  real_t &operator()(int alpha, int j, int beta) const {
    return fibers(alpha, beta)(j);
  }

  KOKKOS_INLINE_FUNCTION int LR() const { return lr; }
  KOKKOS_INLINE_FUNCTION int DD() const { return dd; }
  KOKKOS_INLINE_FUNCTION int RR() const { return rr; }
};

// FiberStorageHost: Host-side storage policy using fiber-based (view-of-views) layout
// Extracted from TensorCoreHostT. This manages both host and device fiber storage
// and maintains the complex two-level view hierarchy with RebuildOuterViews logic.
template <class TTraits>
class FiberStorageHost {
 public:
  using traits = TTraits;
  using real_t = typename TTraits::real_t;
  using fiber_managed_t = FiberView<TTraits, ManagedTag>;
  using fiber_unmanaged_t = FiberView<TTraits, UnmanagedTag>;
  using host_fibers_view_t =
      typename TTraits::template host_view_t<fiber_managed_t**, ManagedTag>;
  using device_managed_fibers_view_t =
      typename TTraits::template view_t<fiber_unmanaged_t**, ManagedTag>;
  using device_unmanaged_fibers_view_t =
      typename TTraits::template view_t<fiber_unmanaged_t**, UnmanagedTag>;

 private:
  int lr{0}, dd{0}, rr{0};
  host_fibers_view_t host_fibers;
  device_managed_fibers_view_t device_managed_fibers;

 public:
  FiberStorageHost() = default;

  void Allocate(int lr_in, int dd_in, int rr_in) {
    dd = dd_in;
    RebuildOuterViews(lr_in, rr_in, [dd_in](int, int) {
      return fiber_managed_t("fiber_m", dd_in);
    });
  }

  void CopyFrom(const FiberStorageHost &other) {
    dd = other.dd;
    if (other.lr == 0 || other.rr == 0) {
      lr = other.lr;
      rr = other.rr;
      host_fibers = host_fibers_view_t();
      device_managed_fibers = device_managed_fibers_view_t();
      return;
    }
    RebuildOuterViews(other.lr, other.rr, [&](int l, int r) {
      return other.host_fibers(l, r);
    });
  }

  void ReduceSize(int lr_new, int rr_new) {
    PARTHENON_REQUIRE(lr_new <= lr && rr_new <= rr,
                      "Target sizes must be smaller than original sizes.");
    auto old_host_fibers = host_fibers;
    RebuildOuterViews(lr_new, rr_new, [&](int l, int r) {
      return old_host_fibers(l, r);
    });
  }

  FiberStorageHost DeepCopy() const {
    FiberStorageHost out;
    out.Allocate(lr, dd, rr);
    for (int l = 0; l < lr; ++l) {
      for (int r = 0; r < rr; ++r) {
        Kokkos::deep_copy(out.host_fibers(l, r), host_fibers(l, r));
      }
    }
    return out;
  }

  device_unmanaged_fibers_view_t GetDeviceData() const {
    return device_unmanaged_fibers_view_t(device_managed_fibers.data(), lr, rr);
  }

  int LR() const { return lr; }
  int DD() const { return dd; }
  int RR() const { return rr; }

 private:
  template <class ManagedFiberGetter>
  void RebuildOuterViews(int lr_new, int rr_new, ManagedFiberGetter &&get_fiber) {
    lr = lr_new;
    rr = rr_new;

    host_fibers = host_fibers_view_t(ViewOfViewAlloc<HostMemSpace>("fibers_m"), lr, rr);
    device_managed_fibers = device_managed_fibers_view_t(ViewOfViewAlloc("fibers_u"), lr, rr);

    auto device_managed_fibers_h = Kokkos::create_mirror_view(device_managed_fibers);

    for (int l = 0; l < lr; ++l) {
      for (int r = 0; r < rr; ++r) {
        host_fibers(l, r) = get_fiber(l, r);
        auto &f = host_fibers(l, r);
        device_managed_fibers_h(l, r) = fiber_unmanaged_t(f.data(), f.extent(0));
      }
    }

    Kokkos::deep_copy(device_managed_fibers, device_managed_fibers_h);
  }
};

// UnmanagedStorageDevice: Wraps raw pointer for scratch arrays
// This replaces wrap_3D with a proper storage policy
template <class TTraits>
class UnmanagedStorageDevice {
 public:
  using traits = TTraits;
  using real_t = typename TTraits::real_t;

 private:
  real_t *scratch;
  int lr{0}, dd{0}, rr{0};

 public:
  KOKKOS_FUNCTION
  UnmanagedStorageDevice() = default;

  KOKKOS_FUNCTION
  UnmanagedStorageDevice(int lr_in, int dd_in, int rr_in, real_t *scratch_in)
      : scratch(scratch_in), lr(lr_in), dd(dd_in), rr(rr_in) {}

  KOKKOS_INLINE_FUNCTION
  real_t &operator()(int alpha, int j, int beta) const {
    if constexpr (TTraits::d_fastest_moving) {
      return scratch[rr * dd * alpha + dd * beta + j];  // [lr][rr][dd]
    } else {
      return scratch[rr * dd * alpha + rr * j + beta];  // [lr][dd][rr]
    }
  }

  KOKKOS_INLINE_FUNCTION int LR() const { return lr; }
  KOKKOS_INLINE_FUNCTION int DD() const { return dd; }
  KOKKOS_INLINE_FUNCTION int RR() const { return rr; }
};

// ContiguousStorageDevice: 3D array with [lr][dd][rr] layout (rr stride-1)
template <class TTraits>
class ContiguousStorageDevice {
 public:
  using traits = TTraits;
  using real_t = typename TTraits::real_t;
  using data_view_t = typename TTraits::template view_t<real_t***, UnmanagedTag>;

 private:
  data_view_t data;
  int lr{0}, dd{0}, rr{0};

 public:
  KOKKOS_FUNCTION
  ContiguousStorageDevice() = default;

  KOKKOS_FUNCTION
  ContiguousStorageDevice(int lr_in, int dd_in, int rr_in, const data_view_t &data_in)
      : data(data_in), lr(lr_in), dd(dd_in), rr(rr_in) {}

  KOKKOS_INLINE_FUNCTION
  real_t &operator()(int alpha, int j, int beta) const {
    return data(alpha, j, beta);  // [lr][dd][rr] with rr stride-1
  }

  KOKKOS_INLINE_FUNCTION int LR() const { return lr; }
  KOKKOS_INLINE_FUNCTION int DD() const { return dd; }
  KOKKOS_INLINE_FUNCTION int RR() const { return rr; }
};

// ContiguousStorageHost: Host-side 3D array management
template <class TTraits>
class ContiguousStorageHost {
 public:
  using traits = TTraits;
  using real_t = typename TTraits::real_t;
  using data_managed_t = typename TTraits::template view_t<real_t***, ManagedTag>;
  using data_unmanaged_t = typename TTraits::template view_t<real_t***, UnmanagedTag>;

 private:
  data_managed_t data;
  int lr{0}, dd{0}, rr{0};

 public:
  ContiguousStorageHost() = default;

  void Allocate(int lr_in, int dd_in, int rr_in) {
    lr = lr_in;
    dd = dd_in;
    rr = rr_in;
    // Kokkos LayoutRight: rightmost index is stride-1
    data = data_managed_t("tensor_core_contiguous", lr, dd, rr);
  }

  void CopyFrom(const ContiguousStorageHost &other) {
    // Kokkos view has reference semantics
    lr = other.lr;
    dd = other.dd;
    rr = other.rr;
    data = other.data;
  }

  void ReduceSize(int lr_new, int rr_new) {
    PARTHENON_REQUIRE(lr_new <= lr && rr_new <= rr,
                      "Target sizes must be smaller than original sizes.");
    auto new_data = data_managed_t("tensor_core_contiguous", lr_new, dd, rr_new);
    auto old_sub = Kokkos::subview(data,
                                    std::make_pair(0, lr_new),
                                    Kokkos::ALL,
                                    std::make_pair(0, rr_new));
    Kokkos::deep_copy(new_data, old_sub);
    data = new_data;
    lr = lr_new;
    rr = rr_new;
  }

  ContiguousStorageHost DeepCopy() const {
    ContiguousStorageHost out;
    out.Allocate(lr, dd, rr);
    Kokkos::deep_copy(out.data, data);
    return out;
  }

  data_unmanaged_t GetDeviceData() const {
    return data_unmanaged_t(data.data(), lr, dd, rr);
  }

  int LR() const { return lr; }
  int DD() const { return dd; }
  int RR() const { return rr; }
};

} // namespace tensor2

} // namespace parthenon

#endif // TENSOR_TT_TRAITS_HPP