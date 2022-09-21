//========================================================================================
// (C) (or copyright) 2020-2022. Triad National Security, LLC. All rights reserved.
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

#include "interface/sparse_pool.hpp"

#include "interface/metadata.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

SparsePool::SparsePool(const std::string &base_name, const Metadata &metadata,
                       const std::vector<int> &sparse_ids,
                       const std::vector<std::vector<int>> &shapes,
                       const std::vector<MetadataFlag> &vector_tensor_flags,
                       const std::vector<std::vector<std::string>> &component_labels,
                       const std::string &controller_base_name)
    : SparsePool(base_name, metadata, controller_base_name) {
  const auto N = sparse_ids.size();

  const auto internal_shapes = shapes.empty() ? std::vector<std::vector<int>>(N) : shapes;
  PARTHENON_REQUIRE_THROWS(internal_shapes.size() == N, "Got wrong number of shapes");

  PARTHENON_REQUIRE_THROWS(component_labels.empty() || (component_labels.size() == N),
                           "Got wrong number of component labels");

  std::vector<const MetadataFlag *> internal_vector_tensor_flags(N, nullptr);
  if (!vector_tensor_flags.empty()) {
    PARTHENON_REQUIRE_THROWS(vector_tensor_flags.size() == N,
                             "Got wrong number of Vector/Tensor flags");
    for (size_t i = 0; i < N; ++i) {
      internal_vector_tensor_flags[i] = &vector_tensor_flags[i];
    }
  }

  for (size_t i = 0; i < N; ++i) {
    AddImpl(sparse_ids[i], internal_shapes[i], internal_vector_tensor_flags[i],
            component_labels.empty() ? std::vector<std::string>{} : component_labels[i]);
  }
}

const Metadata &SparsePool::AddImpl(int sparse_id, const std::vector<int> &shape,
                                    const MetadataFlag *vector_tensor,
                                    const std::vector<std::string> &component_labels) {
  PARTHENON_REQUIRE_THROWS(sparse_id != InvalidSparseID,
                           "Tried to add InvalidSparseID to sparse pool " + base_name_);

  // copy shared metadata
  Metadata this_metadata(
      shared_metadata_.Flags(), shape.size() > 0 ? shape : shared_metadata_.Shape(),
      component_labels.size() > 0 ? component_labels
                                  : shared_metadata_.getComponentLabels(),
      shared_metadata_.getAssociated());

  this_metadata.SetSparseThresholds(shared_metadata_.GetAllocationThreshold(),
                                    shared_metadata_.GetDeallocationThreshold(),
                                    shared_metadata_.GetDefaultValue());

  // if vector_tensor is set, apply it
  if (vector_tensor != nullptr) {
    if (*vector_tensor == Metadata::Vector) {
      this_metadata.Unset(Metadata::Tensor);
      this_metadata.Set(Metadata::Vector);
    } else if (*vector_tensor == Metadata::Tensor) {
      this_metadata.Unset(Metadata::Vector);
      this_metadata.Set(Metadata::Tensor);
    } else if (*vector_tensor == Metadata::None) {
      this_metadata.Unset(Metadata::Vector);
      this_metadata.Unset(Metadata::Tensor);
    } else {
      PARTHENON_THROW("Expected MetadataFlag Vector, Tensor, or None, but got " +
                      vector_tensor->Name());
    }
  }

  // just in case
  this_metadata.IsValid(true);

  const auto ins = pool_.insert({sparse_id, this_metadata});
  PARTHENON_REQUIRE_THROWS(ins.second, "Tried to add sparse ID " +
                                           std::to_string(sparse_id) +
                                           " to sparse pool '" + base_name_ +
                                           "', but this sparse ID already exists");

  return ins.first->second;
}

const Metadata &SparsePool::Add(int sparse_id, const Metadata &md) {
  PARTHENON_REQUIRE_THROWS(sparse_id != InvalidSparseID,
                           "Tried to add InvalidSparseID to sparse pool " + base_name_);

  const auto ins = pool_.insert({sparse_id, md});
  PARTHENON_REQUIRE_THROWS(ins.second, "Tried to add sparse ID " +
                                           std::to_string(sparse_id) +
                                           " to sparse pool '" + base_name_ +
                                           "', but this sparse ID already exists");

  return ins.first->second;
}

} // namespace parthenon
