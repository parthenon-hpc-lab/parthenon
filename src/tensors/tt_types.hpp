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

#include <vector>

#include "basic_types.hpp"
#include "kokkos_abstraction.hpp"
#include "tt_traits.hpp"
#include "utils/indexer.hpp"

namespace parthenon {
namespace tensor2 {

// Lightweight device-side descriptor for one tensor core. This owns no memory;
// it only wraps an unmanaged device view of fibers together with the logical
// core dimensions (left rank, physical dimension, right rank). This is the
// object kernels should use.
//
// Now refactored to use a storage policy pattern for flexibility.
template <class TTraits, class Storage>
class TensorCoreDeviceT {
 public:
  using traits = TTraits;
  using real_t = typename TTraits::real_t;

  // Verify TTraits match between core and storage
  static_assert(std::is_same_v<TTraits, typename Storage::traits>,
                "TTraits mismatch between TensorCoreDeviceT and Storage");

 private:
  Storage storage_;

 public:
  KOKKOS_FUNCTION
  TensorCoreDeviceT() = default;

  // Constructor takes whatever data the policy needs
  template <typename ViewType>
  KOKKOS_FUNCTION
  TensorCoreDeviceT(int lr, int dd, int rr, const ViewType &view)
      : storage_(lr, dd, rr, view) {}

  KOKKOS_FORCEINLINE_FUNCTION int RR() const { return storage_.RR(); }
  KOKKOS_FORCEINLINE_FUNCTION int DD() const { return storage_.DD(); }
  KOKKOS_FORCEINLINE_FUNCTION int LR() const { return storage_.LR(); }

  KOKKOS_FORCEINLINE_FUNCTION
  real_t &operator()(int alpha, int j, int beta) const {
    return storage_(alpha, j, beta);
  }
};

// Host-side owning representation of one tensor core. This is the persistent
// object that keeps fiber storage alive.
//
// Now refactored to use a storage policy pattern for flexibility.
template <class TTraits, class Storage>
class TensorCoreHostT {
 public:
  using traits = TTraits;
  using real_t = typename TTraits::real_t;

  // Verify TTraits match between core and storage
  static_assert(std::is_same_v<TTraits, typename Storage::traits>,
                "TTraits mismatch between TensorCoreHostT and Storage");

 private:
  Storage storage_;
  Indexer6D phys_indexer_{};

 public:
  TensorCoreHostT() = default;

  TensorCoreHostT(int lr, int dd, int rr) 
      : phys_indexer_({0,0}, {0,0}, {0,0}, {0, 0}, {0, 0}, {0, dd - 1}) {
    storage_.Allocate(lr, dd, rr);
  }
 
  TensorCoreHostT(int lr, const Indexer6D& idxer, int rr) 
      : phys_indexer_(idxer) {
    storage_.Allocate(lr, phys_indexer_.size(), rr);
  }

  // Copy constructor delegates to storage policy
  TensorCoreHostT(const TensorCoreHostT &other) {
    phys_indexer_ = other.phys_indexer_;
    storage_.CopyFrom(other.storage_);
  }

  TensorCoreHostT &operator=(const TensorCoreHostT &other) {
    phys_indexer_ = other.phys_indexer_;
    storage_.CopyFrom(other.storage_);
    return *this;
  }

  TensorCoreHostT(TensorCoreHostT &&) = default;
  TensorCoreHostT &operator=(TensorCoreHostT &&) = default;

  ~TensorCoreHostT() = default;

  TensorCoreHostT DeepCopy() const {
    TensorCoreHostT out;
    out.phys_indexer_ = phys_indexer_;
    out.storage_ = storage_.DeepCopy();
    return out;
  }

  // Reduce the active rank-space extent of the core while assuming the fibers
  // in the retained range already contain the correct data.
  void ReduceSize(int lr_new, int rr_new) {
    storage_.ReduceSize(lr_new, rr_new);
  }

  // Release this core's bond storage (fibers/data), leaving a rank-(0 x 0) core.
  void Release() { storage_.Release(); }

  // Generic building blocks for block-structured fiber operations (fiber storage
  // only; the methods below forward to fiber-specific storage members and are
  // therefore only instantiated when called -- e.g. inside DestructiveSum's
  // `if constexpr (d_fastest_moving)` branch). They carry no operation-specific
  // layout knowledge: the operation supplies both the target extents and the
  // per-slot fiber source.

  // Read the managed fiber handle at (l, r), for use as a RebuildFibers source.
  auto GetFiber(int l, int r) const { return storage_.GetFiber(l, r); }

  // Allocate a fresh, zero-initialized fiber sized to this core's physical
  // dimension, for use as a RebuildFibers source in zero blocks.
  auto MakeZeroFiber() const { return storage_.MakeZeroFiber(DD()); }

  // Rebuild this core's fiber storage to (lr x rr) with physical layout `idxer`,
  // sourcing each (l, r) fiber handle from `get_fiber`. Returning an existing
  // handle shares storage (no numeric copy); returning a fresh fiber zero-fills.
  // The physical dimension is inferred from the placed fibers by the storage.
  template <class Getter>
  void RebuildFibers(int lr, int rr, const Indexer6D &idxer, Getter &&get_fiber) {
    phys_indexer_ = idxer;
    storage_.RebuildOuterViews(lr, rr, std::forward<Getter>(get_fiber));
  }

  const auto &Indexer() const {return phys_indexer_;}
  int RR() const { return storage_.RR(); }
  int DD() const { return storage_.DD(); }
  int LR() const { return storage_.LR(); }

  // Construct a shallow device descriptor that is safe to place into a device
  // pack. The returned object is valid as long as this TensorCoreHostT remains
  // alive and structurally unchanged.
  auto GetTensorCoreDevice() const {
    auto device_data = storage_.GetDeviceData();

    using DeviceStorage = std::conditional_t<
      TTraits::d_fastest_moving,
      FiberStorageDevice<TTraits>,
      ContiguousStorageDevice<TTraits>>;

    return TensorCoreDeviceT<TTraits, DeviceStorage>(LR(), DD(), RR(), device_data);
  }
};

// Host-side owning tensor train. This is primarily a lightweight container for
// a sequence of TensorCoreHostT objects with consistent adjacent ranks. It owns
// no device pack state directly; device access happens through TensorPackT.
template <class TTraits>
class TensorTrainT {
 public:
  using traits = TTraits;
  using core_type = std::conditional_t<
    TTraits::d_fastest_moving,
    TensorCoreHostT<TTraits, FiberStorageHost<TTraits>>,
    TensorCoreHostT<TTraits, ContiguousStorageHost<TTraits>>>;

  // Default-constructs an empty train (no cores). Only valid as a placeholder
  // that is subsequently assigned or reshaped (e.g. an output slot in a host
  // pack before a rank-changing op sizes it).
  TensorTrainT() = default;

  // A train's cores must have consistent adjacent bond ranks. The boundary
  // (first-core left, last-core right) ranks are NOT required to be one: a
  // "closed" train has them equal to one and represents a scalar-valued tensor,
  // while an "open" train carries dangling boundary bonds (e.g. a single-core
  // buffer that is a factor of a larger train). Operations that require a closed
  // train guard on IsClosed(); see the op definitions in tt_operations.hpp.
  TensorTrainT(const std::vector<core_type> &cores_in) : cores(cores_in) {
    for (int c = 1; c < NCores(); ++c) {
      PARTHENON_REQUIRE(cores[c - 1].RR() == cores[c].LR(),
                        "Cores must have consistent ranks.");
    }
  }

  // Move-construct a train from an already-assembled sequence of cores. Used by
  // DestructiveSum to install cores whose fiber storage was moved (not copied)
  // from the summands without re-touching the fiber handles.
  TensorTrainT(std::vector<core_type> &&cores_in) : cores(std::move(cores_in)) {
    for (int c = 1; c < NCores(); ++c) {
      PARTHENON_REQUIRE(cores[c - 1].RR() == cores[c].LR(),
                        "Cores must have consistent ranks.");
    }
  }

  // Construct a train from physical dimensions and internal bond ranks. The
  // boundary (left of the first core, right of the last core) ranks default to
  // one -- a closed train -- but can be set to build an open train with dangling
  // boundary bonds (e.g. a single-core buffer factored out of a larger train).
  template <class idx_type>
  TensorTrainT(const std::vector<idx_type> &phys_dims, const std::vector<int> &ranks,
               int left_bond = 1, int right_bond = 1) {
    PARTHENON_REQUIRE(phys_dims.size() - 1 == ranks.size(),
                      "Incompatible number of ranks and dimensions.");
    cores.reserve(phys_dims.size());
    if (ranks.size() == 0) {
      cores.emplace_back(left_bond, phys_dims[0], right_bond);
    } else {
      cores.emplace_back(left_bond, phys_dims[0], ranks[0]);
      for (int c = 1; c < phys_dims.size() - 1; ++c) {
        cores.emplace_back(ranks[c - 1], phys_dims[c], ranks[c]);
      }
      cores.emplace_back(ranks.back(), phys_dims.back(), right_bond);
    }
  }
  
  // Build a train with the same physical structure (per-core physical indexers)
  // as `other` but fresh, zeroed bond space of the given internal ranks.
  TensorTrainT(const TensorTrainT &other, const std::vector<int> &ranks)
      : TensorTrainT(other.GetCoreIndexers(), ranks) {}

  std::vector<Indexer6D> GetCoreIndexers() const {
    std::vector<Indexer6D> out;
    for (int c = 0; c < NCores(); ++c) {
      out.push_back(cores[c].Indexer());
    }
    return out;
  }

  auto NCores() const { return cores.size(); }
  auto &GetCoreHost(int c) { return cores[c]; }
  const auto &GetCoreHost(int c) const { return cores[c]; }

  // A train is "closed" when its boundary bonds are one -- i.e. it contracts to a
  // scalar-valued tensor rather than carrying dangling factor bonds. Operations
  // that assume proper TT structure (sums, rounding) require this; buffers that
  // are a single factored-out core (e.g. AMR coarse buffers) are open.
  bool IsClosed() const {
    return !cores.empty() && cores.front().LR() == 1 && cores.back().RR() == 1;
  }

  // A train is "empty" when it carries no bond data.
  bool IsEmpty() const { return cores.empty() || cores.front().RR() == 0; }

  // Empty the train's bond storage while preserving its structural identity: the
  // core objects, and thus NCores() and each core's physical indexer, are kept.
  void Clear() {
    for (auto &c : cores) c.Release();
  }

  // Rebuild this train in place as a fresh, zeroed train at the given internal
  // bond ranks, reusing the existing cores' physical indexers.
  void BuildFresh(const std::vector<int> &ranks) {
    PARTHENON_REQUIRE(!cores.empty(),
                      "BuildFresh: train has no cores to describe its shape.");
    *this = TensorTrainT(GetCoreIndexers(), ranks);
  }

  auto &operator()(int c) { return cores[c]; }
  const auto &operator()(int c) const { return cores[c]; }

  int GetPhysicalDimension(int dim) const { return cores[dim].DD(); }

  TensorTrainT DeepCopy() const {
    std::vector<core_type> new_cores;
    new_cores.reserve(cores.size());
    for (const auto &core : cores) {
      new_cores.push_back(core.DeepCopy());
    }
    return TensorTrainT(new_cores);
  }

 private:
  std::vector<core_type> cores;
};

template <class TTraits>
std::vector<TensorTrainT<TTraits>>
DeepCopyTrains(const std::vector<TensorTrainT<TTraits>> &trains) {
  std::vector<TensorTrainT<TTraits>> out;
  out.reserve(trains.size());
  for (const auto &train : trains) {
    out.push_back(train.DeepCopy());
  }
  return out;
}

// Type alias to replace wrap_3D - scratch arrays using unmanaged storage
template <class TTraits>
using ScratchCore = TensorCoreDeviceT<TTraits, UnmanagedStorageDevice<TTraits>>;

// Default type aliases (uses DefaultTTraits - can be swapped by Parthenon)
using TensorCoreDevice = std::conditional_t<
  DefaultTTraits::d_fastest_moving,
  TensorCoreDeviceT<DefaultTTraits, FiberStorageDevice<DefaultTTraits>>,
  TensorCoreDeviceT<DefaultTTraits, ContiguousStorageDevice<DefaultTTraits>>>;

using TensorCoreHost = std::conditional_t<
  DefaultTTraits::d_fastest_moving,
  TensorCoreHostT<DefaultTTraits, FiberStorageHost<DefaultTTraits>>,
  TensorCoreHostT<DefaultTTraits, ContiguousStorageHost<DefaultTTraits>>>;

using TensorTrain = TensorTrainT<DefaultTTraits>;

// Contiguous storage variants (explicit TTraits for testing)
using TensorCoreDeviceContiguous = TensorCoreDeviceT<ContiguousTTraits, ContiguousStorageDevice<ContiguousTTraits>>;
using TensorCoreHostContiguous = TensorCoreHostT<ContiguousTTraits, ContiguousStorageHost<ContiguousTTraits>>;
using TensorTrainContiguous = TensorTrainT<ContiguousTTraits>;

} // namespace tensor2
} // namespace parthenon

#endif // TENSORS_TT_TYPES_HPP
