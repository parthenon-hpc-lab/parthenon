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

#ifndef TENSORS_TT_TYPES_HPP
#define TENSORS_TT_TYPES_HPP

#include "basic_types.hpp"
#include "kokkos_abstraction.hpp"
#include "tt_traits.hpp"

namespace parthenon {
namespace tensor2 {
// A FiberView is the fundamental 1D storage unit for tensor-core data.
// Managed fibers own device memory, while unmanaged fibers are shallow
// handles used inside device-facing descriptors.
template <class TTraits, class OwnershipTag>
using FiberView =
    typename TTraits::template view_t<typename TTraits::real_t*, OwnershipTag>;

// Lightweight device-side descriptor for one tensor core. This owns no memory;
// it only wraps an unmanaged device view of fibers together with the logical
// core dimensions (left rank, physical dimension, right rank). This is the
// object kernels should use.
template <class TTraits>
class TensorCoreDeviceT {
 public:
  using real_t = typename TTraits::real_t;
  using fiber_unmanaged_t = FiberView<TTraits, UnmanagedTag>;
  using device_fibers_view_t =
      typename TTraits::template view_t<fiber_unmanaged_t**, UnmanagedTag>;

  KOKKOS_FUNCTION
  TensorCoreDeviceT() = default;

  KOKKOS_FUNCTION
  TensorCoreDeviceT(int lr, int dd, int rr, const device_fibers_view_t &fibers)
      : lr(lr), dd(dd), rr(rr), fibers(fibers) {}

  KOKKOS_INLINE_FUNCTION int RR() const { return rr; }
  KOKKOS_INLINE_FUNCTION int DD() const { return dd; }
  KOKKOS_INLINE_FUNCTION int LR() const { return lr; }

  KOKKOS_INLINE_FUNCTION
  real_t &operator()(int alpha, int j, int beta) const {
    return fibers(alpha, beta)(j);
  }

  KOKKOS_INLINE_FUNCTION
  fiber_unmanaged_t fiber(int alpha, int beta) const {
    return fibers(alpha, beta);
  }

 private:
  int lr{0}, dd{0}, rr{0};
  device_fibers_view_t fibers;
};

// Host-side owning representation of one tensor core. This is the persistent
// object that keeps fiber storage alive. It owns:
//   1. a host-side managed outer view of managed fibers, and
//   2. a device-side managed outer view of unmanaged fiber handles.
// The latter is wrapped in an unmanaged view when constructing
// TensorCoreDeviceT.
template <class TTraits>
class TensorCoreHostT {
 public:
  using real_t = typename TTraits::real_t;
  using fiber_unmanaged_t = FiberView<TTraits, UnmanagedTag>;
  using fiber_managed_t = FiberView<TTraits, ManagedTag>;
  using host_fibers_view_t =
      typename TTraits::template host_view_t<fiber_managed_t**, ManagedTag>;
  using device_managed_fibers_view_t =
      typename TTraits::template view_t<fiber_unmanaged_t**, ManagedTag>;
  using device_unmanaged_fibers_view_t =
      typename TTraits::template view_t<fiber_unmanaged_t**, UnmanagedTag>;

  TensorCoreHostT() = default;

  TensorCoreHostT(int lr_in, int dd_in, int rr_in) : dd(dd_in) {
    printf("Building core of size (%i, %i, %i)\n", lr_in, dd, rr_in);
    RebuildOuterViews(lr_in, rr_in, [dd_in](int, int) {
      // TODO(LFR): Switch to memory pools
      return fiber_managed_t("fiber_m", dd_in);
    });
  }

  // Reduce the active rank-space extent of the core while assuming the fibers
  // in the retained range already contain the correct data.
  void ReduceSize(int lr_in, int rr_in) {
    PARTHENON_REQUIRE(lr_in <= lr,
                      "Target sizes must be smaller than original sizes.");
    PARTHENON_REQUIRE(rr_in <= rr,
                      "Target sizes must be smaller than original sizes.");
    auto old_host_fibers = host_fibers;
    RebuildOuterViews(lr_in, rr_in, [&](int l, int r) {
      return old_host_fibers(l, r);
    });
  }

  int RR() const { return rr; }
  int DD() const { return dd; }
  int LR() const { return lr; }

  // Construct a shallow device descriptor that is safe to place into a device
  // pack. The returned object is valid as long as this TensorCoreHostT remains
  // alive and structurally unchanged.
  TensorCoreDeviceT<TTraits> GetTensorCoreDeviceT() const {
    return TensorCoreDeviceT<TTraits>(lr, dd, rr, device_unmanaged_fibers);
  }
  
  // TensorCoreHost owns the persistent outer hierarchy for a tensor core.
  // In addition to the owning host/device views, it stores
  // device_unmanaged_fibers, which is a non-owning alias derived from
  // device_managed_fibers.data() and the current (lr, rr) extents.
  //
  // Because this unmanaged view is derived state, the default copy/move
  // operations are not sufficient: after any copy or move, the unmanaged alias
  // must be rebound so that it matches the copied/moved managed device view.
  // For that reason TensorCoreHost defines the full rule of five explicitly.
  TensorCoreHostT(const TensorCoreHostT &other)
      : lr(other.lr),
        dd(other.dd),
        rr(other.rr),
        host_fibers(other.host_fibers),
        device_managed_fibers(other.device_managed_fibers) {
    RebindUnmanagedView();
  }
  
  TensorCoreHostT &operator=(const TensorCoreHostT &other) {
    if (this != &other) {
      lr = other.lr;
      dd = other.dd;
      rr = other.rr;
      host_fibers = other.host_fibers;
      device_managed_fibers = other.device_managed_fibers;
      RebindUnmanagedView();
    }
    return *this;
  }
  
  TensorCoreHostT(TensorCoreHostT &&other) noexcept
      : lr(other.lr),
        dd(other.dd),
        rr(other.rr),
        host_fibers(std::move(other.host_fibers)),
        device_managed_fibers(std::move(other.device_managed_fibers)) {
    RebindUnmanagedView();
  }
  
  TensorCoreHostT &operator=(TensorCoreHostT &&other) noexcept {
    if (this != &other) {
      lr = other.lr;
      dd = other.dd;
      rr = other.rr;
      host_fibers = std::move(other.host_fibers);
      device_managed_fibers = std::move(other.device_managed_fibers);
      RebindUnmanagedView();
    }
    return *this;
  }
  
  ~TensorCoreHostT() = default;

 private:
  int lr{0}, dd{0}, rr{0};

  // Stores managed fibers on host to maintain ownership.
  host_fibers_view_t host_fibers;
  // Managed device view of unmanaged fibers, used to keep the device-side outer
  // descriptor alive.
  device_managed_fibers_view_t device_managed_fibers;
  // Unmanaged wrapper over the device-side outer descriptor, used when building
  // TensorCoreDeviceT objects.
  device_unmanaged_fibers_view_t device_unmanaged_fibers;
  
  void RebindUnmanagedView() {
    device_unmanaged_fibers =
        device_unmanaged_fibers_view_t(device_managed_fibers.data(), lr, rr);
  }

  // The only routine that should change the structural state of a TensorCoreHostT.
  // It preserves the invariant that host_fibers, device_managed_fibers, and
  // device_unmanaged_fibers all describe the same (lr, rr) outer hierarchy.
  template <class ManagedFiberGetter>
  void RebuildOuterViews(int lr_new, int rr_new, ManagedFiberGetter &&get_fiber) {
    lr = lr_new;
    rr = rr_new;

    host_fibers = host_fibers_view_t(ViewOfViewAlloc<HostMemSpace>("fibers_m"), lr, rr);
    device_managed_fibers = device_managed_fibers_view_t(ViewOfViewAlloc("fibers_u"), lr, rr);
    RebindUnmanagedView();

    auto device_managed_fibers_h = Kokkos::create_mirror_view(device_managed_fibers);

    for (int l = 0; l < lr; ++l) {
      for (int r = 0; r < rr; ++r) {
        host_fibers(l, r) = get_fiber(l, r);
        auto &f = host_fibers(l, r);
        device_managed_fibers_h(l, r) = fiber_unmanaged_t(f.data(), f.extent(0));
      }
    }

    // Only the outer descriptor is copied here. The fiber data already live on
    // device and are not copied by this deep_copy.
    Kokkos::deep_copy(device_managed_fibers, device_managed_fibers_h);

  }
};

// Host-side owning tensor train. This is primarily a lightweight container for
// a sequence of TensorCoreHostT objects with consistent adjacent ranks. It owns
// no device pack state directly; device access happens through TensorPackT.
template <class TTraits>
class TensorTrainT {
 public:
  TensorTrainT(const std::vector<TensorCoreHostT<TTraits>> &cores_in) : cores(cores_in) {
    PARTHENON_REQUIRE(cores.front().LR() == 1,
                      "First core must have left side size one.");
    PARTHENON_REQUIRE(cores.back().RR() == 1,
                      "Last core must have right side size one.");
    for (int c = 1; c < NCores(); ++c) {
      PARTHENON_REQUIRE(cores[c - 1].RR() == cores[c].LR(),
                        "Cores must have consistent ranks.");
    }
  }

  // Construct a train from physical dimensions and internal bond ranks.
  // The boundary ranks are fixed to one.
  TensorTrainT(const std::vector<int> &phys_dims, const std::vector<int> &ranks) {
    PARTHENON_REQUIRE(phys_dims.size() - 1 == ranks.size(),
                      "Incompatible number of ranks and dimensions.");
    // cores.reserve(phys_dims.size());
    if (ranks.size() == 0) {
      cores.emplace_back(1, phys_dims[0], 1);
    } else {
      cores.emplace_back(1, phys_dims[0], ranks[0]);
      for (int c = 1; c < phys_dims.size() - 1; ++c) {
        cores.emplace_back(ranks[c - 1], phys_dims[c], ranks[c]);
      }
      cores.emplace_back(ranks.back(), phys_dims.back(), 1);
    }
  }

  auto NCores() const { return cores.size(); }
  auto &GetCoreHost(int c) { return cores[c]; }
  const auto &GetCoreHost(int c) const { return cores[c]; }

  auto &operator()(int c) { return cores[c]; }
  const auto &operator()(int c) const { return cores[c]; }

 private:
  std::vector<TensorCoreHostT<TTraits>> cores;
};

// Packed device-facing view of tensor cores. For now this stores cores directly
// with indices (block, variable, core) rather than introducing a separate
// device-side TensorTrainT descriptor. This is the main aggregate object used in
// kernels.
template <class TTraits>
struct TensorPackT {
  using view_t =
      typename TTraits::template view_t<TensorCoreDeviceT<TTraits>***, ManagedTag>;

  // View of size (nblocks, nvars, ncores). At the moment nvars is a placeholder
  // and the pack stores tensor cores directly rather than explicit trains.
  view_t cores;
  int ncores_per_train;

  KOKKOS_INLINE_FUNCTION
  int GetNBlocks() const { return cores.extent_int(0); }

  KOKKOS_INLINE_FUNCTION
  int GetNCores() const { return cores.extent_int(2); }

  TensorPackT(const std::vector<TensorTrainT<TTraits>> &trains) {
    int nvars{1}; // Placeholder
    ncores_per_train = trains[0].NCores();
    cores = view_t("TensorPackT", trains.size(), nvars, ncores_per_train);
    auto cores_h = Kokkos::create_mirror_view(cores);

    for (int v = 0; v < nvars; ++v) {
      for (int t = 0; t < trains.size(); ++t) {
        PARTHENON_REQUIRE(trains[t].NCores() == ncores_per_train,
                          "All trains must be the same dimension.");
        for (int c = 0; c < ncores_per_train; ++c) {
          cores_h(t, v, c) = trains[t].GetCoreHost(c).GetTensorCoreDeviceT();
        }
      }
    }
    Kokkos::deep_copy(cores, cores_h);
  }

  KOKKOS_INLINE_FUNCTION
  auto &operator()(int b, int v, int c) const { return cores(b, v, c); }
};

using TensorCoreDevice = TensorCoreDeviceT<DefaultTTraits>;
using TensorCoreHost   = TensorCoreHostT<DefaultTTraits>;
using TensorTrain      = TensorTrainT<DefaultTTraits>;
using TensorPack       = TensorPackT<DefaultTTraits>;

} // namespace tensor2
} // namespace parthenon

#endif // TENSORS_TT_TYPES_HPP