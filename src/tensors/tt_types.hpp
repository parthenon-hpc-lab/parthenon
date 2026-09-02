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
#include "kokkos_abstraction.hpp"
#include "tt_traits.hpp"
#include "utils/concepts_lite.hpp"

namespace parthenon {
namespace tensor2 {

// Base type for tensor-train field name tags used to index a multi-field pack
// on device, analogous to parthenon::variable_names::var_base_t for SparsePack.
// A tensor-train field is a whole train (no sub-components), so a tag only needs
// to supply a name (used to select the field when building a pack from a
// container). Downstream code defines one tag per field:
//
//   struct my_field : public tensor2::tt_var_base_t {
//     static std::string name() { return "my_field"; }
//   };
//
// and indexes a pack with pack(my_field{}, b, c).
struct tt_var_base_t {
  static std::string name() {
    PARTHENON_FAIL("Tensor-train field tags must implement their own name().");
    return "error";
  }
};

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

// Packed device-facing view of tensor cores, indexed (block, variable, core).
// A pack may hold several fields (variables) so long as they share the same
// shape (identical number of cores and per-core physical dimensions); this is
// validated at construction. Fields can be addressed on device either by
// integer variable index -- pack(b, v, c) -- or, when the pack is templated on
// field name tags Ts..., by tag -- pack(field::I{}, b, c) -- which resolves to
// the tag's position in Ts... at compile time (cf. SparsePack).
template <class TTraits, class... Ts>
struct TensorPackT {
  using device_core_t = std::conditional_t<
    TTraits::d_fastest_moving,
    TensorCoreDeviceT<TTraits, FiberStorageDevice<TTraits>>,
    TensorCoreDeviceT<TTraits, ContiguousStorageDevice<TTraits>>>;

  using view_t = typename TTraits::template view_t<device_core_t***, ManagedTag>;
  using dims_host_view_t = typename TTraits::template host_view_t<int*, ManagedTag>;

  // View of size (nblocks, nvars, ncores).
  view_t cores;
  dims_host_view_t physical_dims_h;
  int ncores_per_train;

  KOKKOS_INLINE_FUNCTION
  int GetNBlocks() const { return cores.extent_int(0); }

  KOKKOS_INLINE_FUNCTION
  int GetNVars() const { return cores.extent_int(1); }

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

  // Construct a single-variable pack from a batch of trains (one per block).
  TensorPackT(const std::vector<TensorTrainT<TTraits>> &trains) {
    PARTHENON_REQUIRE(!trains.empty(),
                      "Cannot construct a TensorPackT from an empty train vector.");
    std::vector<const TensorTrainT<TTraits> *> ptrs;
    ptrs.reserve(trains.size());
    for (const auto &t : trains)
      ptrs.push_back(&t);
    BuildFromTrainPointers_({ptrs});
  }

  // Construct a single-variable pack from a batch of (non-owning) train
  // pointers. Avoids deep-copying trains; used to pack container-owned trains.
  TensorPackT(const std::vector<const TensorTrainT<TTraits> *> &train_ptrs) {
    PARTHENON_REQUIRE(!train_ptrs.empty(),
                      "Cannot construct a TensorPackT from an empty train pointer list.");
    BuildFromTrainPointers_({train_ptrs});
  }

  // Construct a multi-variable pack: vars[v][b] is the train for variable v on
  // block b. All (var, block) trains must share the same shape.
  explicit TensorPackT(
      const std::vector<std::vector<const TensorTrainT<TTraits> *>> &vars) {
    BuildFromTrainPointers_(vars);
  }

  // Integer variable indexing.
  KOKKOS_INLINE_FUNCTION
  auto &operator()(int b, int v, int c) const { return cores(b, v, c); }

  // Compile-time variable index of a field name tag within Ts...
  template <class Tag>
  static constexpr int VarIndex() {
    static_assert(IncludesType<Tag, Ts...>::value,
                  "Field tag is not part of this pack's tag list.");
    return static_cast<int>(GetTypeIdx<Tag, Ts...>::value);
  }

  // Tag-based variable indexing: pack(b, field::I{}, c). Block leads, matching
  // the integer operator()(b, v, c) and SparsePack's (b, tag, ...) convention.
  template <class Tag>
  KOKKOS_INLINE_FUNCTION auto &operator()(int b, const Tag &, int c) const {
    return cores(b, VarIndex<Tag>(), c);
  }

 private:
  void BuildFromTrainPointers_(
      const std::vector<std::vector<const TensorTrainT<TTraits> *>> &vars) {
    PARTHENON_REQUIRE(!vars.empty() && !vars[0].empty(),
                      "Cannot construct a TensorPackT with no variables or blocks.");
    const int nvars = static_cast<int>(vars.size());
    const int nblocks = static_cast<int>(vars[0].size());
    ncores_per_train = vars[0][0]->NCores();
    cores = view_t("TensorPackT", nblocks, nvars, ncores_per_train);
    auto cores_h = Kokkos::create_mirror_view(cores);
    physical_dims_h = dims_host_view_t("TensorPackT physical dims", ncores_per_train);
    for (int c = 0; c < ncores_per_train; ++c)
      physical_dims_h(c) = vars[0][0]->GetPhysicalDimension(c);

    for (int v = 0; v < nvars; ++v) {
      PARTHENON_REQUIRE(static_cast<int>(vars[v].size()) == nblocks,
                        "All variables in a pack must span the same blocks.");
      for (int t = 0; t < nblocks; ++t) {
        PARTHENON_REQUIRE(vars[v][t]->NCores() == ncores_per_train,
                          "All trains in a pack must have the same number of cores.");
        for (int c = 0; c < ncores_per_train; ++c) {
          PARTHENON_REQUIRE(vars[v][t]->GetPhysicalDimension(c) == physical_dims_h(c),
                            "All trains in a pack must have the same physical dimensions.");
          cores_h(t, v, c) = vars[v][t]->GetCoreHost(c).GetTensorCoreDevice();
        }
      }
    }
    Kokkos::deep_copy(cores, cores_h);
  }
};

// Host-side pack over a batch of tensor trains: a list of non-owning pointers to
// the host TensorTrainT objects (one per block), plus the metadata operations a
// tensor-train kernel needs before it can build a device pack -- querying and
// reshaping cores. Unlike regular-field kernels (which only need a device pack
// over fixed-size data), tensor-train operations frequently need to reshape a
// train host-side (e.g. resize the output ranks before a sum) before running
// the device kernel. This host pack is that intermediate layer; the device pack
// (TensorPackT) is built from it via MakeDevicePack().
//
// Reshaping mutates the pointed-to trains in place, so the storage owner (a mesh
// container slot or a test-local vector element) sees the updated train without
// any rebinding.
// The host pack may optionally be templated on the same field name tags Ts... as
// the device pack it produces. A tagged host pack (built via FromContainer<Ts...>)
// remembers its field list, so MakeDevicePack() yields a correctly-tagged device
// pack with no need to re-specify the tags; an untagged pack (FromContainer, no
// tags) gathers all of a container's fields and produces an integer-indexed
// device pack. The generic library ops operate on untagged packs; app kernels
// that want compile-time device indexing use tagged packs.
template <class TTraits, class... Ts>
class TensorTrainHostPackT {
 public:
  using train_t = TensorTrainT<TTraits>;
  static constexpr int num_tags = sizeof...(Ts);

  TensorTrainHostPackT() = default;

  // Construct from an explicit (variable-major) grid of train pointers:
  // trains[v][b] is the train for variable v on block b, with field_names[v]
  // the corresponding field name.
  TensorTrainHostPackT(std::vector<std::string> field_names,
                       std::vector<std::vector<train_t *>> trains)
      : field_names_(std::move(field_names)), trains_(std::move(trains)) {
    PARTHENON_REQUIRE(!trains_.empty() && !trains_.front().empty(),
                      "Cannot build a TensorTrainHostPack with no variables or blocks.");
    PARTHENON_REQUIRE(field_names_.size() == trains_.size(),
                      "field_names and trains must have the same number of variables.");
    static_assert(num_tags == 0 || num_tags >= 1, "");
    if constexpr (num_tags > 0)
      PARTHENON_REQUIRE(static_cast<int>(trains_.size()) == num_tags,
                        "A tagged host pack must hold exactly its tagged fields.");
  }

  // Wrap a vector of owning trains as a single-variable host pack (unit tests).
  static TensorTrainHostPackT FromVector(std::vector<train_t> &trains) {
    std::vector<train_t *> ptrs;
    ptrs.reserve(trains.size());
    for (auto &t : trains)
      ptrs.push_back(&t);
    return TensorTrainHostPackT({std::string{}}, {std::move(ptrs)});
  }

  // Gather fields held by a mesh-partition container (e.g. MeshTTData) into a
  // host pack. The untagged pack gathers all of the container's fields in its
  // field order; a tagged pack (Ts...) gathers exactly {Ts::name()...} in tag
  // order. Templated on the container type so tt_types.hpp stays free of the
  // mesh/interface headers; the container need only provide NumBlocks(),
  // FieldNames(), and GetBlockData(b)->Get(field).
  template <class Container>
  static TensorTrainHostPackT FromContainer(Container &md) {
    std::vector<std::string> names;
    if constexpr (num_tags > 0) {
      names = {Ts::name()...};
    } else {
      names = md.FieldNames();
    }
    PARTHENON_REQUIRE(!names.empty(),
                      "Cannot build a TensorTrainHostPack from a container with no "
                      "tensor-train fields.");
    const int nblocks = md.NumBlocks();
    std::vector<std::vector<train_t *>> trains(names.size());
    for (std::size_t v = 0; v < names.size(); ++v) {
      trains[v].reserve(nblocks);
      for (int b = 0; b < nblocks; ++b)
        trains[v].push_back(md.GetBlockData(b)->Get(names[v]).get());
    }
    return TensorTrainHostPackT(std::move(names), std::move(trains));
  }

  int NumVars() const { return static_cast<int>(trains_.size()); }
  int NumBlocks() const { return static_cast<int>(trains_.front().size()); }
  int NCores() const { return trains_.front().front()->NCores(); }
  const std::vector<std::string> &FieldNames() const { return field_names_; }

  // Compile-time variable index of a field tag within this pack's tags Ts...
  template <class Tag>
  static constexpr int VarIndex() {
    static_assert(IncludesType<Tag, Ts...>::value,
                  "Field tag is not part of this host pack's tag list.");
    return static_cast<int>(GetTypeIdx<Tag, Ts...>::value);
  }

  // Integer variable indexing (block leads, matching the device pack).
  train_t &operator()(int b, int v = 0) { return *trains_[v][b]; }
  const train_t &operator()(int b, int v = 0) const { return *trains_[v][b]; }

  // Tag-based variable indexing: pack(b, field::I{}). Only available on a tagged
  // host pack.
  template <class Tag>
  train_t &operator()(int b, const Tag &) { return *trains_[VarIndex<Tag>()][b]; }
  template <class Tag>
  const train_t &operator()(int b, const Tag &) const {
    return *trains_[VarIndex<Tag>()][b];
  }

  // Reshape variable v on block b in place to the given physical dimensions and
  // internal bond ranks (mutates the owner's train). Used to size an output
  // train before a rank-changing op fills it on device.
  void Reshape(int b, int v, const std::vector<int> &phys_dims,
               const std::vector<int> &ranks) {
    *trains_[v][b] = train_t(phys_dims, ranks);
  }
  void Reshape(int b, const std::vector<int> &phys_dims,
               const std::vector<int> &ranks) {
    Reshape(b, 0, phys_dims, ranks);
  }
  // Reshape by field tag.
  template <class Tag>
  void Reshape(int b, const Tag &, const std::vector<int> &phys_dims,
               const std::vector<int> &ranks) {
    Reshape(b, VarIndex<Tag>(), phys_dims, ranks);
  }

  // Build the device pack over the current (post-reshape) trains. The device
  // pack carries this host pack's tags Ts..., so a tagged host pack yields a
  // tag-indexable device pack and an untagged one yields an integer-indexed pack.
  TensorPackT<TTraits, Ts...> MakeDevicePack() const {
    return TensorPackT<TTraits, Ts...>(ConstView_());
  }

  // Build a single-variable, integer-indexed device pack over variable v.
  TensorPackT<TTraits> MakeDevicePackForVar(int v) const {
    return TensorPackT<TTraits>(VarView_(v));
  }

 private:
  // Non-owning const view of variable v's per-block trains.
  std::vector<const train_t *> VarView_(int v) const {
    return std::vector<const train_t *>(trains_[v].begin(), trains_[v].end());
  }
  std::vector<std::vector<const train_t *>> ConstView_() const {
    std::vector<std::vector<const train_t *>> out;
    out.reserve(trains_.size());
    for (std::size_t v = 0; v < trains_.size(); ++v)
      out.push_back(VarView_(static_cast<int>(v)));
    return out;
  }

  std::vector<std::string> field_names_;
  std::vector<std::vector<train_t *>> trains_; // [var][block]
};

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
using TensorPack = TensorPackT<DefaultTTraits>;
using TensorTrainHostPack = TensorTrainHostPackT<DefaultTTraits>;
// Tagged host pack over DefaultTTraits for a specific set of field tags.
template <class... Ts>
using TensorTrainHostPackFor = TensorTrainHostPackT<DefaultTTraits, Ts...>;

// Contiguous storage variants (explicit TTraits for testing)
using TensorCoreDeviceContiguous = TensorCoreDeviceT<ContiguousTTraits, ContiguousStorageDevice<ContiguousTTraits>>;
using TensorCoreHostContiguous = TensorCoreHostT<ContiguousTTraits, ContiguousStorageHost<ContiguousTTraits>>;
using TensorTrainContiguous = TensorTrainT<ContiguousTTraits>;
using TensorPackContiguous = TensorPackT<ContiguousTTraits>;
using TensorTrainHostPackContiguous = TensorTrainHostPackT<ContiguousTTraits>;

} // namespace tensor2
} // namespace parthenon

#endif // TENSORS_TT_TYPES_HPP
