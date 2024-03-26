//========================================================================================
// (C) (or copyright) 2020-2023. Triad National Security, LLC. All rights reserved.
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
std::shared_ptr<T> &
DataCollection<T>::Add(const std::string &name, const std::shared_ptr<T> &src,
                       const std::vector<std::string> &field_names, const bool shallow) {
  auto it = containers_.find(name);
  if (it != containers_.end()) {
    if (!(it->second)->Contains(field_names)) {
      PARTHENON_THROW(name +
                      "already exists in collection but does not contain field names");
    }
    return it->second;
  }

  auto c = std::make_shared<T>(name);
  c->Initialize(src.get(), field_names, shallow);

  Set(name, c);

  return containers_[name];
}
template <typename T>
std::shared_ptr<T> &DataCollection<T>::Add(const std::string &label,
                                           const std::shared_ptr<T> &src,
                                           const std::vector<std::string> &flags) {
  return Add(label, src, flags, false);
}
template <typename T>
std::shared_ptr<T> &DataCollection<T>::AddShallow(const std::string &label,
                                                  const std::shared_ptr<T> &src,
                                                  const std::vector<std::string> &flags) {
  return Add(label, src, flags, true);
}

std::shared_ptr<MeshData<Real>> &
GetOrAdd_impl(Mesh *pmy_mesh_,
              std::map<std::string, std::shared_ptr<MeshData<Real>>> &containers_,
              BlockList_t &block_list, const std::string &mbd_label,
              const int &partition_id, const std::optional<int> gmg_level) {
  std::string label = mbd_label + "_part-" + std::to_string(partition_id);
  if (gmg_level) label = label + "_gmg-" + std::to_string(*gmg_level);
  auto it = containers_.find(label);
  if (it == containers_.end()) {
    // TODO(someone) add caching of partitions to Mesh at some point
    const int pack_size = pmy_mesh_->DefaultPackSize();
    auto partitions = partition::ToSizeN(block_list, pack_size);
    // Account for possibly empty block_list
    if (partitions.size() == 0) partitions = std::vector<BlockList_t>(1);
    for (auto i = 0; i < partitions.size(); i++) {
      std::string md_label = mbd_label + "_part-" + std::to_string(i);
      if (gmg_level) md_label = md_label + "_gmg-" + std::to_string(*gmg_level);
      containers_[md_label] = std::make_shared<MeshData<Real>>(mbd_label);
      containers_[md_label]->Set(partitions[i], pmy_mesh_);
      if (gmg_level) {
        containers_[md_label]->grid =
            GridIdentifier{GridType::two_level_composite, *gmg_level};
      } else {
        containers_[md_label]->grid = GridIdentifier{GridType::leaf, 0};
      }
    }
  }
  return containers_[label];
}

template <>
std::shared_ptr<MeshData<Real>> &
DataCollection<MeshData<Real>>::GetOrAdd(const std::string &mbd_label,
                                         const int &partition_id) {
  return GetOrAdd_impl(pmy_mesh_, containers_, pmy_mesh_->block_list, mbd_label,
                       partition_id, {});
}

template <>
std::shared_ptr<MeshData<Real>> &
DataCollection<MeshData<Real>>::GetOrAdd(int gmg_level, const std::string &mbd_label,
                                         const int &partition_id) {
  return GetOrAdd_impl(pmy_mesh_, containers_, pmy_mesh_->gmg_block_lists[gmg_level],
                       mbd_label, partition_id, gmg_level);
}

template class DataCollection<MeshData<Real>>;
template class DataCollection<MeshBlockData<Real>>;

} // namespace parthenon
