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

#ifndef TENSORS_TT_PACK_HPP
#define TENSORS_TT_PACK_HPP

#include <string>
#include <memory>
#include <vector>

#include "kokkos_abstraction.hpp"
#include "tt_traits.hpp"
#include "tt_types.hpp"
#include "utils/concepts_lite.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {
namespace tensor2 {

// Base type for tensor-train field name tags, analogous to
// variable_names::var_base_t for SparsePack. A tensor-train field is a whole
// train (no sub-components), so a tag only supplies a name. Downstream code
// defines one tag per field and indexes a pack with pack(b, my_field{}, c):
//
//   struct my_field : public tensor2::tt_var_base_t {
//     static std::string name() { return "my_field"; }
//   };
struct tt_var_base_t {
  static std::string name() {
    PARTHENON_FAIL("Tensor-train field tags must implement their own name().");
    return "error";
  }
};

// A dense (nvars x nblocks) grid of elements backed by a single flat vector.
// Used to hold the (var, block) train pointers of a host pack without a nested
// std::vector; ragged grids are not supported.
template <class T>
class VarBlockGrid {
 public:
  VarBlockGrid() = default;
  VarBlockGrid(int nvars, int nblocks)
      : nvars_(nvars), nblocks_(nblocks), data_(nvars * nblocks) {}

  int NumVars() const { return nvars_; }
  int NumBlocks() const { return nblocks_; }

  T &operator()(int v, int b) { return data_[v * nblocks_ + b]; }
  const T &operator()(int v, int b) const { return data_[v * nblocks_ + b]; }

 private:
  int nvars_{0};
  int nblocks_{0};
  std::vector<T> data_;
};

// Device-facing pack of tensor cores, indexed (block, variable, core). A pack
// may hold several fields provided they share the same shape (same core count
// and per-core physical dimensions), which is validated at construction. Fields
// are addressed by integer -- pack(b, v, c) -- or, when the pack is templated on
// field tags var_ts..., by tag -- pack(b, my_field{}, c) -- with the tag
// resolved to its variable slot at compile time (cf. SparsePack).
template <class TTraits, class... var_ts>
struct TensorPackT {
  using train_t = TensorTrainT<TTraits>;
  using device_core_t = std::conditional_t<
    TTraits::d_fastest_moving,
    TensorCoreDeviceT<TTraits, FiberStorageDevice<TTraits>>,
    TensorCoreDeviceT<TTraits, ContiguousStorageDevice<TTraits>>>;

  using view_t = typename TTraits::template view_t<device_core_t***, ManagedTag>;
  using dims_host_view_t = typename TTraits::template host_view_t<int*, ManagedTag>;

  view_t cores; // (nblocks, nvars, ncores)
  dims_host_view_t physical_dims_h;
  int ncores_per_train;

  KOKKOS_INLINE_FUNCTION int GetNBlocks() const { return cores.extent_int(0); }
  KOKKOS_INLINE_FUNCTION int GetNVars() const { return cores.extent_int(1); }
  KOKKOS_INLINE_FUNCTION int GetNCores() const { return cores.extent_int(2); }

  int GetPhysicalDimension(int dim) const { return physical_dims_h(dim); }
  std::vector<int> GetPhysicalDimensions() const {
    std::vector<int> dims(GetNCores());
    for (int c = 0; c < GetNCores(); ++c)
      dims[c] = physical_dims_h(c);
    return dims;
  }

  // Single-variable pack from a batch of trains (one per block).
  TensorPackT(const std::vector<TensorTrainT<TTraits>> &trains) {
    PARTHENON_REQUIRE(!trains.empty(),
                      "Cannot construct a TensorPackT from an empty train vector.");
    VarBlockGrid<const train_t *> grid(1, trains.size());
    for (std::size_t b = 0; b < trains.size(); ++b)
      grid(0, b) = &trains[b];
    Build_(grid);
  }

  // Single-variable pack from a batch of (non-owning) train pointers.
  TensorPackT(const std::vector<const TensorTrainT<TTraits> *> &train_ptrs) {
    PARTHENON_REQUIRE(!train_ptrs.empty(),
                      "Cannot construct a TensorPackT from an empty train pointer list.");
    VarBlockGrid<const train_t *> grid(1, train_ptrs.size());
    for (std::size_t b = 0; b < train_ptrs.size(); ++b)
      grid(0, b) = train_ptrs[b];
    Build_(grid);
  }

  // Multi-variable pack from a (var, block) grid of train pointers.
  explicit TensorPackT(const VarBlockGrid<const train_t *> &grid) { Build_(grid); }

  KOKKOS_INLINE_FUNCTION
  auto &operator()(int b, int v, int c) const { return cores(b, v, c); }

  // Compile-time variable slot of a field tag within var_ts...
  template <class var_t>
  static constexpr int VarIndex() {
    static_assert(IncludesType<var_t, var_ts...>::value,
                  "Field tag is not part of this pack's tag list.");
    return static_cast<int>(GetTypeIdx<var_t, var_ts...>::value);
  }

  // Tag-based variable indexing: pack(b, my_field{}, c).
  template <class var_t>
  KOKKOS_INLINE_FUNCTION auto &operator()(int b, const var_t &, int c) const {
    return cores(b, VarIndex<var_t>(), c);
  }

 private:
  void Build_(const VarBlockGrid<const train_t *> &grid) {
    const int nvars = grid.NumVars();
    const int nblocks = grid.NumBlocks();
    PARTHENON_REQUIRE(nvars > 0 && nblocks > 0,
                      "Cannot construct a TensorPackT with no variables or blocks.");
    ncores_per_train = grid(0, 0)->NCores();
    cores = view_t("TensorPackT", nblocks, nvars, ncores_per_train);
    auto cores_h = Kokkos::create_mirror_view(cores);
    physical_dims_h = dims_host_view_t("TensorPackT physical dims", ncores_per_train);
    for (int c = 0; c < ncores_per_train; ++c)
      physical_dims_h(c) = grid(0, 0)->GetPhysicalDimension(c);

    for (int v = 0; v < nvars; ++v) {
      for (int b = 0; b < nblocks; ++b) {
        const auto *train = grid(v, b);
        PARTHENON_REQUIRE(train->NCores() == ncores_per_train,
                          "All trains in a pack must have the same number of cores.");
        for (int c = 0; c < ncores_per_train; ++c) {
          PARTHENON_REQUIRE(train->GetPhysicalDimension(c) == physical_dims_h(c),
                            "All trains in a pack must have the same physical dimensions.");
          cores_h(b, v, c) = train->GetCoreHost(c).GetTensorCoreDevice();
        }
      }
    }
    Kokkos::deep_copy(cores, cores_h);
  }
};

// Host-side pack over a (var, block) grid of non-owning train pointers, plus the
// metadata operations a kernel needs before building a device pack -- querying
// and reshaping cores. Tensor-train ops frequently reshape a train host-side
// (e.g. resize output ranks before a sum) before running the device kernel;
// this host pack is that intermediate layer and MakeDevicePack() produces the
// device pack. Reshaping mutates the pointed-to trains in place, so the storage
// owner (a mesh container slot or a test-local vector) sees the update with no
// rebinding.
//
// Optionally templated on the same field tags var_ts... as the device pack it
// produces: a tagged pack (FromContainer<var_ts...>) is tag-indexable and yields
// a tagged device pack; an untagged pack gathers all of a container's fields and
// yields an integer-indexed device pack.
template <class TTraits, class... var_ts>
class TensorTrainHostPackT {
 public:
  using train_t = TensorTrainT<TTraits>;
  static constexpr int num_tags = sizeof...(var_ts);

  TensorTrainHostPackT() = default;
  explicit TensorTrainHostPackT(VarBlockGrid<train_t *> grid) : grid_(std::move(grid)) {
    PARTHENON_REQUIRE(grid_.NumVars() > 0 && grid_.NumBlocks() > 0,
                      "Cannot build a TensorTrainHostPack with no variables or blocks.");
    if constexpr (num_tags > 0)
      PARTHENON_REQUIRE(grid_.NumVars() == num_tags,
                        "A tagged host pack must hold exactly its tagged fields.");
  }

  // Wrap a batch of owning trains as a single-variable host pack (unit tests).
  static TensorTrainHostPackT FromVector(std::vector<train_t> &trains) {
    VarBlockGrid<train_t *> grid(1, trains.size());
    for (std::size_t b = 0; b < trains.size(); ++b)
      grid(0, b) = &trains[b];
    return TensorTrainHostPackT(std::move(grid));
  }

  // Wrap a batch of shared-ptr-owned trains as a single-variable host pack. The trains
  // are not copied; the caller retains ownership and the pack aliases them (as it does
  // for container-owned trains). Used to pack transient boundary addend trains.
  static TensorTrainHostPackT
  FromSharedPtrs(std::vector<std::shared_ptr<train_t>> &trains) {
    VarBlockGrid<train_t *> grid(1, trains.size());
    for (std::size_t b = 0; b < trains.size(); ++b)
      grid(0, b) = trains[b].get();
    return TensorTrainHostPackT(std::move(grid));
  }

  // Gather named fields from a mesh-partition container (e.g. MeshTTData). The
  // untagged pack gathers all of the container's fields in its field order; a
  // tagged pack gathers {var_ts::name()...} in tag order. Templated on the
  // container type so this header stays free of the mesh/interface headers; the
  // container need only provide NumBlocks(), FieldNames(), and
  // GetBlockData(b)->Get(field).
  template <class Container>
  static TensorTrainHostPackT FromContainer(Container &md) {
    if constexpr (num_tags > 0) {
      return FromNames(md, {var_ts::name()...});
    } else {
      return FromNames(md, md.FieldNames());
    }
  }

  // Gather an explicit list of named fields from a container. The number of
  // names must match the pack's tag count (if tagged).
  template <class Container>
  static TensorTrainHostPackT FromNames(Container &md,
                                        const std::vector<std::string> &names) {
    PARTHENON_REQUIRE(!names.empty(),
                      "Cannot build a TensorTrainHostPack from an empty field list.");
    const int nblocks = md.NumBlocks();
    VarBlockGrid<train_t *> grid(static_cast<int>(names.size()), nblocks);
    for (std::size_t v = 0; v < names.size(); ++v)
      for (int b = 0; b < nblocks; ++b)
        grid(v, b) = md.GetBlockData(b)->Get(names[v]).get();
    return TensorTrainHostPackT(std::move(grid));
  }

  int NumVars() const { return grid_.NumVars(); }
  int NumBlocks() const { return grid_.NumBlocks(); }
  int NCores() const { return grid_(0, 0)->NCores(); }

  // Compile-time variable slot of a field tag within var_ts...
  template <class var_t>
  static constexpr int VarIndex() {
    static_assert(IncludesType<var_t, var_ts...>::value,
                  "Field tag is not part of this host pack's tag list.");
    return static_cast<int>(GetTypeIdx<var_t, var_ts...>::value);
  }

  // Integer variable indexing (block leads, matching the device pack).
  train_t &operator()(int b, int v = 0) { return *grid_(v, b); }
  const train_t &operator()(int b, int v = 0) const { return *grid_(v, b); }

  // Tag-based variable indexing: pack(b, my_field{}).
  template <class var_t>
  train_t &operator()(int b, const var_t &) { return *grid_(VarIndex<var_t>(), b); }
  template <class var_t>
  const train_t &operator()(int b, const var_t &) const {
    return *grid_(VarIndex<var_t>(), b);
  }

  // Reshape a train in place to the given physical dimensions and internal bond
  // ranks (mutates the owner's train). Used to size an output train before a
  // rank-changing op fills it on device.
  void Reshape(int b, int v, const std::vector<int> &phys_dims,
               const std::vector<int> &ranks) {
    auto &train = *grid_(v, b);
    train = train_t(phys_dims, ranks, train.label(), train.metadata());
  }
  void Reshape(int b, const std::vector<int> &phys_dims,
               const std::vector<int> &ranks) {
    Reshape(b, 0, phys_dims, ranks);
  }
  template <class var_t>
  void Reshape(int b, const var_t &, const std::vector<int> &phys_dims,
               const std::vector<int> &ranks) {
    Reshape(b, VarIndex<var_t>(), phys_dims, ranks);
  }

  // Build the device pack over the current (post-reshape) trains, carrying this
  // host pack's tags var_ts...
  TensorPackT<TTraits, var_ts...> MakeDevicePack() const {
    return TensorPackT<TTraits, var_ts...>(ConstGrid_(NumVars()));
  }

  // Build a single-variable, integer-indexed device pack over variable v.
  TensorPackT<TTraits> MakeDevicePackForVar(int v) const {
    VarBlockGrid<const train_t *> g(1, NumBlocks());
    for (int b = 0; b < NumBlocks(); ++b)
      g(0, b) = grid_(v, b);
    return TensorPackT<TTraits>(g);
  }

 private:
  VarBlockGrid<const train_t *> ConstGrid_(int nvars) const {
    VarBlockGrid<const train_t *> g(nvars, NumBlocks());
    for (int v = 0; v < nvars; ++v)
      for (int b = 0; b < NumBlocks(); ++b)
        g(v, b) = grid_(v, b);
    return g;
  }

  VarBlockGrid<train_t *> grid_; // (var, block) non-owning train pointers
};

// Default type aliases (DefaultTTraits).
using TensorPack = TensorPackT<DefaultTTraits>;
using TensorTrainHostPack = TensorTrainHostPackT<DefaultTTraits>;
// Tagged host pack over DefaultTTraits for a specific set of field tags.
template <class... var_ts>
using TensorTrainHostPackFor = TensorTrainHostPackT<DefaultTTraits, var_ts...>;

// Contiguous storage variants (explicit TTraits for testing).
using TensorPackContiguous = TensorPackT<ContiguousTTraits>;
using TensorTrainHostPackContiguous = TensorTrainHostPackT<ContiguousTTraits>;

} // namespace tensor2
} // namespace parthenon

#endif // TENSORS_TT_PACK_HPP
