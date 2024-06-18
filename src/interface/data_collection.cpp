//========================================================================================
// (C) (or copyright) 2020-2024. Triad National Security, LLC. All rights reserved.
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

#include <string>

#include "interface/data_collection.hpp"
#include "interface/mesh_data.hpp"
#include "interface/meshblock_data.hpp"
#include "mesh/mesh.hpp"
#include "utils/partition_stl_containers.hpp"

namespace parthenon {

template <typename T>
std::shared_ptr<T> &DataCollection<T>::Add(const std::string &label) {
  // error check for duplicate names
  auto it = containers_.find(label);
  if (it != containers_.end()) {
    return it->second;
  }
  containers_[label] = std::make_shared<T>();
  return containers_[label];
}

template <>
std::shared_ptr<MeshData<Real>> &
DataCollection<MeshData<Real>>::GetOrAdd_impl(const std::string &mbd_label,
                                              const int &partition_id,
                                              const std::optional<int> gmg_level) {
  std::string label = GetKey(mbd_label, partition_id, gmg_level);
  auto it = containers_.find(label);
  if (it == containers_.end()) {
    // TODO(someone) add caching of partitions to Mesh at some point
    const int pack_size = pmy_mesh_->DefaultPackSize();
    auto &block_list =
        gmg_level ? pmy_mesh_->gmg_block_lists[*gmg_level] : pmy_mesh_->block_list;
    auto partitions = partition::ToSizeN(block_list, pack_size);
    // Account for possibly empty block_list
    if (partitions.size() == 0) partitions = std::vector<BlockList_t>(1);
    for (auto i = 0; i < partitions.size(); i++) {
      std::string md_label = GetKey(mbd_label, partition_id, gmg_level);
      containers_[md_label] = std::make_shared<MeshData<Real>>(mbd_label);
      containers_[md_label]->Initialize(partitions[i], pmy_mesh_, gmg_level);
      containers_[md_label]->partition = i;
    }
  }
  return containers_[label];
}

template <>
std::shared_ptr<MeshData<Real>> &
DataCollection<MeshData<Real>>::GetOrAdd(const std::string &mbd_label,
                                         const int &partition_id) {
  return GetOrAdd_impl(mbd_label, partition_id, {});
}

template <>
std::shared_ptr<MeshData<Real>> &
DataCollection<MeshData<Real>>::GetOrAdd(int gmg_level, const std::string &mbd_label,
                                         const int &partition_id) {
  return GetOrAdd_impl(mbd_label, partition_id, gmg_level);
}

template class DataCollection<MeshData<Real>>;
template class DataCollection<MeshBlockData<Real>>;

} // namespace parthenon
