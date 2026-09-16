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

#ifndef TENSORS_TT_VARIABLE_HPP
#define TENSORS_TT_VARIABLE_HPP

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "interface/metadata.hpp"
#include "tensors/tt_field_metadata.hpp"
#include "tensors/tt_types.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

class MeshBlock;

// A tensor-train field instance on a block: the mesh-facing wrapper around a pure
// mathematical TensorTrain. TensorTrainT is deliberately identity-free (it is only cores
// and bonds); TTVariableT pairs it with the field's label and registered TTFieldMetadata
// so it satisfies the "Variable concept" the boundary-comm templates (CalcIndices,
// ForEachBoundary, SendKey/ReceiveKey) require -- label(), metadata(), IsSet(), GetDim().
//
// This is the tensor-train analogue of Variable<T>: the container owns TTVariableT objects
// (one per field per block), boundary comm walks them as variables, and the math library
// operates on the TensorTrain reached via train().
template <class TTraits>
class TTVariableT {
 public:
  using train_t = tensor2::TensorTrainT<TTraits>;

  TTVariableT() = default;

  // Construct from an existing train (e.g. after building/initializing it). The train is
  // installed via set_train, so it is validated against the field's registered structure.
  TTVariableT(std::string label, TTFieldMetadata md, train_t train)
      : label_(std::move(label)), md_(std::move(md)) {
    set_train(std::move(train));
  }

  // Construct the field's train from its registered structure on a given block: the
  // per-core physical shapes come from the metadata (spatial core sized from the block
  // geometry), with the given internal bond ranks. This is the canonical path a container
  // uses to create a fresh field variable. `coarse` sizes the spatial core at the coarse
  // (half-resolution) extents for multilevel (AMR) coarse buffers.
  TTVariableT(std::string label, TTFieldMetadata md, std::weak_ptr<MeshBlock> wpmb,
              const std::vector<int> &ranks, bool coarse = false)
      : label_(std::move(label)), md_(std::move(md)),
        train_(md_.CoreIndexers(wpmb, coarse), ranks) {}

  // Variable-concept surface -------------------------------------------------
  const std::string &label() const { return label_; }
  const Metadata &metadata() const { return md_.metadata; }
  bool IsSet(const MetadataFlag bit) const { return md_.metadata.IsSet(bit); }
  // A TT field exposes no separate tensor-component index dimensions: its fixed extra
  // indices are flattened into the (spatial) first core, so CalcIndices' {GetDim(6),
  // GetDim(5), GetDim(4)} component ranges are all one.
  int GetDim(const int i) const {
    PARTHENON_REQUIRE(0 < i && i <= 6, "Index out of bounds");
    return 1;
  }

  const TTFieldMetadata &field_metadata() const { return md_; }

  // Math access --------------------------------------------------------------
  train_t &train() { return train_; }
  const train_t &train() const { return train_; }

  // Install a train, validating its physical structure against the field's registered
  // metadata: the core count must match, and each non-spatial core's physical dimension
  // must match the registered phys_dims. (The spatial core's dimension is derived per-block
  // and so is not stored on the metadata; only its presence -- core 0 -- is checked.) Only
  // bond ranks may differ, so the field's shape identity is preserved. Checked always (host
  // side, once per field per exchange -- never a hot path).
  void set_train(train_t train) {
    PARTHENON_REQUIRE(train.NCores() == md_.NCores(),
                      "set_train: core count must match the field's registered structure.");
    for (std::size_t c = 0; c < md_.phys_dims.size(); ++c)
      PARTHENON_REQUIRE(
          train(c + 1).DD() == md_.phys_dims[c],
          "set_train: non-spatial core dimensions must match the field's metadata.");
    train_ = std::move(train);
  }

 private:
  std::string label_;
  TTFieldMetadata md_;
  train_t train_;
};

using TTVariable = TTVariableT<DefaultTTraits>;

} // namespace parthenon

#endif // TENSORS_TT_VARIABLE_HPP
