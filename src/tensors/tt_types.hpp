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

#include <string>
#include <vector>

#include "basic_types.hpp"
#include "interface/metadata.hpp"
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

  TensorTrainT(const std::vector<core_type> &cores_in) : cores(cores_in) {
    PARTHENON_REQUIRE(cores.front().LR() == 1,
                      "First core must have left side size one.");
    PARTHENON_REQUIRE(cores.back().RR() == 1,
                      "Last core must have right side size one.");
    for (int c = 1; c < NCores(); ++c) {
      PARTHENON_REQUIRE(cores[c - 1].RR() == cores[c].LR(),
                        "Cores must have consistent ranks.");
    }
  }

  TensorTrainT(const std::vector<core_type> &cores_in, std::string label,
               Metadata metadata)
      : TensorTrainT(cores_in) {
    label_ = std::move(label);
    metadata_ = std::move(metadata);
  }

  // Construct a train from physical dimensions and internal bond ranks.
  // The boundary ranks are fixed to one.
  template <class idx_type> 
  TensorTrainT(const std::vector<idx_type> &phys_dims, const std::vector<int> &ranks) {
    PARTHENON_REQUIRE(phys_dims.size() - 1 == ranks.size(),
                      "Incompatible number of ranks and dimensions.");
    cores.reserve(phys_dims.size());
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
  
  template <class idx_type>
  TensorTrainT(const std::vector<idx_type> &phys_dims, const std::vector<int> &ranks,
               std::string label, Metadata metadata)
      : TensorTrainT(phys_dims, ranks) {
    label_ = std::move(label);
    metadata_ = std::move(metadata);
  }

  TensorTrainT(const TensorTrainT &other, const std::vector<int> &ranks)
      : TensorTrainT(other.GetCoreIndexers(), ranks, other.label_, other.metadata_) {}
  
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

  // Variable-concept surface -------------------------------------------------
  // TensorTrainT is the tensor-train analogue of Variable<T>: the container
  // stamps a label and Metadata onto each train so that the same boundary-comm
  // templates (CalcIndices, ForEachBoundary, SendKey/ReceiveKey) that operate on
  // Variable<T> also accept a TensorTrainT.
  const std::string &label() const { return label_; }
  const Metadata &metadata() const { return metadata_; }
  bool IsSet(const MetadataFlag bit) const { return metadata_.IsSet(bit); }
  // Stamp the identity onto a freshly-produced train (e.g. the result of a sum/round op,
  // which starts metadata-less) so it still satisfies the Variable concept.
  void SetConcept(std::string label, Metadata metadata) {
    label_ = std::move(label);
    metadata_ = std::move(metadata);
  }

  // Physical dimension along Variable tensor axis i (1-indexed), as consumed by
  // CalcIndices for the {GetDim(6), GetDim(5), GetDim(4)} component ranges. These
  // are always 1 for a tensor train: a TT field's fixed extra indices would be
  // flattened into the spatial (first) core, so they are not separate index
  // dimensions the way tensor components are for a regular field. The separate
  // trailing cores (e.g. angular NTHETA/NPHI) are a distinct concept from the
  // unflattened dimensions of the first core and are not exposed here.
  int GetDim(const int i) const {
    PARTHENON_REQUIRE(0 < i && i <= 6, "Index out of bounds");
    return 1;
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
    return TensorTrainT(new_cores, label_, metadata_);
  }

 private:
  std::vector<core_type> cores;
  std::string label_;
  Metadata metadata_;
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
