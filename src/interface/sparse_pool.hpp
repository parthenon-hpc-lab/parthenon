//========================================================================================
// (C) (or copyright) 2021. Triad National Security, LLC. All rights reserved.
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
#ifndef INTERFACE_SPARSE_POOL_HPP_
#define INTERFACE_SPARSE_POOL_HPP_

#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "metadata.hpp"
#include "utils/error_checking.hpp"
#include "variable.hpp"

namespace parthenon {

class SparsePool {
 public:
  // Create an empty sparse pool
  SparsePool(const std::string &base_name, const Metadata &metadata,
             const std::string &controller_base_name = "")
      : base_name_(base_name), controller_base_name_(controller_base_name),
        shared_metadata_(metadata) {
    PARTHENON_REQUIRE_THROWS(shared_metadata_.IsSet(Metadata::Sparse),
                             "Must set Sparse flag for a SparsePool");
  }

  // Create a copy of the sparse pool with a different name
  SparsePool(const std::string &new_base_name, const SparsePool &src)
      : base_name_(new_base_name), controller_base_name_(src.controller_base_name_),
        shared_metadata_(src.shared_metadata()), pool_(src.pool()) {}

  // Create a sparse pool with given sparse ids, shapes, Vector/Tensor flags, and optional
  // component labels
  SparsePool(const std::string &base_name, const Metadata &metadata,
             const std::vector<int> &sparse_ids,
             const std::vector<std::vector<int>> &shapes,
             const std::vector<MetadataFlag> &vector_tensor_flags,
             const std::vector<std::vector<std::string>> &component_labels = {},
             const std::string &controller_base_name = "");

  // Create a sparse pool with given sparse ids and controlling base name and optional
  // shapes and component labels
  SparsePool(const std::string &base_name, const Metadata &metadata,
             const std::string &controller_base_name, const std::vector<int> &sparse_ids,
             const std::vector<std::vector<int>> &shapes = {},
             const std::vector<std::vector<std::string>> &component_labels = {})
      : SparsePool(base_name, metadata, sparse_ids, shapes, {}, component_labels,
                   controller_base_name) {}

  // Create a sparse pool with given sparse ids and optional shapes and component labels
  SparsePool(const std::string &base_name, const Metadata &metadata,
             const std::vector<int> &sparse_ids,
             const std::vector<std::vector<int>> &shapes = {},
             const std::vector<std::vector<std::string>> &component_labels = {})
      : SparsePool(base_name, metadata, sparse_ids, shapes, {}, component_labels, "") {}

  // Create a sparse pool with given sparse ids and component labels
  SparsePool(const std::string &base_name, const Metadata &metadata,
             const std::vector<int> &sparse_ids,
             const std::vector<std::vector<std::string>> &component_labels)
      : SparsePool(base_name, metadata, sparse_ids, {}, {}, component_labels, "") {}

  const std::string &base_name() const { return base_name_; }
  const std::string &controller_base_name() const { return controller_base_name_; }
  const Metadata &shared_metadata() const { return shared_metadata_; }
  const std::unordered_map<int, Metadata> &pool() const { return pool_; }
  auto size() const { return pool_.size(); }

  // Add a new sparse ID to the pool with optional arguments:
  // shape: use this shape if not {}, otherwise use shape from shared metadata (the
  // Vector/Tensor flag will be copied from shared metadata)
  // component_labels: use these component labels if not {}, otherwise use component
  // labels from shared metadata
  const Metadata &Add(int sparse_id, const std::vector<int> &shape = {},
                      const std::vector<std::string> &component_labels = {}) {
    return AddImpl(sparse_id, shape, nullptr, component_labels);
  }

  // As above, but explicitly set Vector/Tensor metadata flag. Valid values for
  // vector_tensor are: None (unset both Vector and Tensor flag), Vector (set only Vector
  // flag), Tensor (set only Tensor flag)
  const Metadata &Add(int sparse_id, const std::vector<int> &shape,
                      MetadataFlag vector_tensor,
                      const std::vector<std::string> &component_labels = {}) {
    return AddImpl(sparse_id, shape, &vector_tensor, component_labels);
  }

  const Metadata &Add(int sparse_id, const std::vector<std::string> &component_labels) {
    return AddImpl(sparse_id, {}, nullptr, component_labels);
  }

 private:
  // TODO(JL) Once we have C++17 with std::optional, we can use
  // std::optional<MetadataFlag> instead of a pointer. We need to differentiate between
  // the getting a value form the user and not getting a value, but there is no good
  // default value
  const Metadata &AddImpl(int sparse_id, const std::vector<int> &shape,
                          const MetadataFlag *vector_tensor,
                          const std::vector<std::string> &component_labels);

  const std::string base_name_;
  const std::string controller_base_name_;

  Metadata shared_metadata_;
  // Metadata per sparse id
  std::unordered_map<int, Metadata> pool_;
};

} // namespace parthenon

#endif // INTERFACE_SPARSE_POOL_HPP_
