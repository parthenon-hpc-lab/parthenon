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

// Lightweight device-side descriptor for one tensor core. This owns no memory;
// it only wraps an unmanaged device view of fibers together with the logical
// core dimensions (left rank, physical dimension, right rank). This is the
// object kernels should use.
//
// Now refactored to use a storage policy pattern for flexibility.
template <class TTraits, template<class> class StoragePolicy = FiberStorageDevice>
class TensorCoreDeviceT {
 public:
  using traits = TTraits;
  using real_t = typename TTraits::real_t;

 private:
  StoragePolicy<TTraits> storage_;

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

template <class CoreLike, bool transpose = false>
struct vertical_unfolding {
  const CoreLike &core;
  int nl, nd, nr;
  vertical_unfolding(const CoreLike &core_in) : core(core_in),
                                                nl(core_in.LR()),
                                                nd(core_in.DD()),
                                                nr(core_in.RR()) {}
  vertical_unfolding(const CoreLike &core_in, int nl, int nd, int nr)
      : core(core_in),
        nl(nl),
        nd(nd),
        nr(nr) {}
  KOKKOS_FORCEINLINE_FUNCTION
  decltype(auto) operator()(int j, int i) const {
    if constexpr (transpose) {
      const int rl = i / nd;
      const int d = i % nd;
      return core(rl, d, j);
    } else {
      const int rl = j / nd;
      const int d = j % nd;
      return core(rl, d, i);
    }
  }
};

template<class CoreLike>
KOKKOS_FORCEINLINE_FUNCTION
auto GetVerticalUnfolding(const CoreLike &core_in) {
  return vertical_unfolding<CoreLike, false>(core_in);
}

template<class CoreLike>
KOKKOS_FORCEINLINE_FUNCTION
auto GetVerticalUnfolding(const CoreLike &core_in, int nl, int nd, int nr) {
  return vertical_unfolding<CoreLike, false>(core_in, nl, nd, nr);
}

template<class CoreLike>
KOKKOS_FORCEINLINE_FUNCTION
auto GetVerticalUnfoldingTranspose(const CoreLike &core_in) {
  return vertical_unfolding<CoreLike, true>(core_in);
}

template<class CoreLike>
KOKKOS_FORCEINLINE_FUNCTION
auto GetVerticalUnfoldingTranspose(const CoreLike &core_in, int nl, int nd, int nr) {
  return vertical_unfolding<CoreLike, true>(core_in, nl, nd, nr);
}

template <class T, bool transpose>
KOKKOS_FORCEINLINE_FUNCTION
int GetNrows(const vertical_unfolding<T, transpose> &m) { return transpose ? m.nr : m.nd * m.nl; }
template <class T, bool transpose>
KOKKOS_FORCEINLINE_FUNCTION
int GetNcols(const vertical_unfolding<T, transpose> &m) { return transpose ? m.nd * m.nl : m.nr; }

template <class CoreLike, bool transpose>
struct horizontal_unfolding {
  const CoreLike &core;
  int nl, nd, nr;
  horizontal_unfolding(const CoreLike &core_in) : core(core_in),
                                                   nl(core_in.LR()),
                                                   nd(core_in.DD()),
                                                   nr(core_in.RR()) {}
  horizontal_unfolding(const CoreLike &core_in, int nl, int nd, int nr)
      : core(core_in),
        nl(nl),
        nd(nd),
        nr(nr) {}
  KOKKOS_FORCEINLINE_FUNCTION
  decltype(auto) operator()(int row, int col) const {
    if constexpr (transpose) {
      const int rr = row / nd;
      const int d = row % nd;
      return core(col, d, rr);
    } else {
      const int rr = col / nd;
      const int d = col % nd;
      return core(row, d, rr);
    }
  }
};

template<class CoreLike>
KOKKOS_FORCEINLINE_FUNCTION
auto GetHorizontalUnfolding(const CoreLike &core_in) {
  return horizontal_unfolding<CoreLike, false>(core_in);
}

template<class CoreLike>
KOKKOS_FORCEINLINE_FUNCTION
auto GetHorizontalUnfolding(const CoreLike &core_in, int nl, int nd, int nr) {
  return horizontal_unfolding<CoreLike, false>(core_in, nl, nd, nr);
}

template<class CoreLike>
KOKKOS_FORCEINLINE_FUNCTION
auto GetHorizontalUnfoldingTranspose(const CoreLike &core_in) {
  return horizontal_unfolding<CoreLike, true>(core_in);
}

template<class CoreLike>
KOKKOS_FORCEINLINE_FUNCTION
auto GetHorizontalUnfoldingTranspose(const CoreLike &core_in, int nl, int nd, int nr) {
  return horizontal_unfolding<CoreLike, true>(core_in, nl, nd, nr);
}

template <class T, bool transpose>
KOKKOS_FORCEINLINE_FUNCTION
int GetNrows(const horizontal_unfolding<T, transpose> &m) { return transpose ? m.nd * m.nr : m.nl; }
template <class T, bool transpose>
KOKKOS_FORCEINLINE_FUNCTION
int GetNcols(const horizontal_unfolding<T, transpose> &m) { return transpose ? m.nl : m.nd * m.nr; }

// Host-side owning representation of one tensor core. This is the persistent
// object that keeps fiber storage alive.
//
// Now refactored to use a storage policy pattern for flexibility.
template <class TTraits, template<class> class StoragePolicy = FiberStorageHost>
class TensorCoreHostT {
 public:
  using traits = TTraits;
  using real_t = typename TTraits::real_t;

 private:
  StoragePolicy<TTraits> storage_;

 public:
  TensorCoreHostT() = default;

  TensorCoreHostT(int lr, int dd, int rr) {
    storage_.Allocate(lr, dd, rr);
  }

  // Copy constructor delegates to storage policy
  TensorCoreHostT(const TensorCoreHostT &other) {
    storage_.CopyFrom(other.storage_);
  }

  TensorCoreHostT &operator=(const TensorCoreHostT &other) {
    storage_.CopyFrom(other.storage_);
    return *this;
  }

  TensorCoreHostT(TensorCoreHostT &&) = default;
  TensorCoreHostT &operator=(TensorCoreHostT &&) = default;

  ~TensorCoreHostT() = default;

  TensorCoreHostT DeepCopy() const {
    TensorCoreHostT out;
    out.storage_ = storage_.DeepCopy();
    return out;
  }

  // Reduce the active rank-space extent of the core while assuming the fibers
  // in the retained range already contain the correct data.
  void ReduceSize(int lr_new, int rr_new) {
    storage_.ReduceSize(lr_new, rr_new);
  }

  int RR() const { return storage_.RR(); }
  int DD() const { return storage_.DD(); }
  int LR() const { return storage_.LR(); }

  // Construct a shallow device descriptor that is safe to place into a device
  // pack. The returned object is valid as long as this TensorCoreHostT remains
  // alive and structurally unchanged.
  TensorCoreDeviceT<TTraits, FiberStorageDevice> GetTensorCoreDevice() const {
    auto device_data = storage_.GetDeviceData();
    return TensorCoreDeviceT<TTraits, FiberStorageDevice>(LR(), DD(), RR(), device_data);
  }
};

// Host-side owning tensor train. This is primarily a lightweight container for
// a sequence of TensorCoreHostT objects with consistent adjacent ranks. It owns
// no device pack state directly; device access happens through TensorPackT.
template <class TTraits>
class TensorTrainT {
 public:
  using traits = TTraits;
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

  auto NCores() const { return cores.size(); }
  auto &GetCoreHost(int c) { return cores[c]; }
  const auto &GetCoreHost(int c) const { return cores[c]; }

  auto &operator()(int c) { return cores[c]; }
  const auto &operator()(int c) const { return cores[c]; }

  int GetPhysicalDimension(int dim) const { return cores[dim].DD(); }

  TensorTrainT DeepCopy() const {
    std::vector<TensorCoreHostT<TTraits>> new_cores;
    new_cores.reserve(cores.size());
    for (const auto &core : cores) {
      new_cores.push_back(core.DeepCopy());
    }
    return TensorTrainT(new_cores);
  }

 private:
  std::vector<TensorCoreHostT<TTraits>> cores;
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

// Packed device-facing view of tensor cores. For now this stores cores directly
// with indices (block, variable, core) rather than introducing a separate
// device-side TensorTrainT descriptor. This is the main aggregate object used in
// kernels.
template <class TTraits>
struct TensorPackT {
  using view_t =
      typename TTraits::template view_t<TensorCoreDeviceT<TTraits>***, ManagedTag>;
  using dims_host_view_t = typename TTraits::template host_view_t<int*, ManagedTag>;

  // View of size (nblocks, nvars, ncores). At the moment nvars is a placeholder
  // and the pack stores tensor cores directly rather than explicit trains.
  view_t cores;
  dims_host_view_t physical_dims_h;
  int ncores_per_train;

  KOKKOS_INLINE_FUNCTION
  int GetNBlocks() const { return cores.extent_int(0); }

  KOKKOS_INLINE_FUNCTION
  int GetNCores() const { return cores.extent_int(2); }

  int GetPhysicalDimension(int dim) const {
    return physical_dims_h(dim);
  }

  std::vector<int> GetPhysicalDimensions() const {
    std::vector<int> dims(GetNCores());
    for (int c = 0; c < GetNCores(); ++c) {
      dims[c] = physical_dims_h(c);
    }
    return dims;
  }

  TensorPackT(const std::vector<TensorTrainT<TTraits>> &trains) {
    PARTHENON_REQUIRE(!trains.empty(),
                      "Cannot construct a TensorPackT from an empty train vector.");
    int nvars{1}; // Placeholder
    ncores_per_train = trains[0].NCores();
    cores = view_t("TensorPackT", trains.size(), nvars, ncores_per_train);
    auto cores_h = Kokkos::create_mirror_view(cores);
    physical_dims_h = dims_host_view_t("TensorPackT physical dims", ncores_per_train);
    for (int c = 0; c < ncores_per_train; ++c) {
      physical_dims_h(c) = trains[0].GetPhysicalDimension(c);
    }

    for (int v = 0; v < nvars; ++v) {
      for (int t = 0; t < trains.size(); ++t) {
        PARTHENON_REQUIRE(trains[t].NCores() == ncores_per_train,
                          "All trains must have the same number of cores.");
        for (int c = 0; c < ncores_per_train; ++c) {
          PARTHENON_REQUIRE(trains[t].GetPhysicalDimension(c) == physical_dims_h(c),
                            "All trains in a pack must have the same physical dimensions.");
          cores_h(t, v, c) = trains[t].GetCoreHost(c).GetTensorCoreDevice();
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
