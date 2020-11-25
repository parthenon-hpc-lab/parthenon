//========================================================================================
// (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
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
template <typename F>
std::shared_ptr<T> DataCollection<T>::Add(const std::string &name,
                                          const std::shared_ptr<T> &src,
                                          const std::vector<F> &flags) {
  auto it = containers_.find(name);
  if (it != containers_.end()) {
    // TODO(jcd): figure out how to do error checking here
    return it->second;
  }

  auto c = std::make_shared<T>();
  c->Copy(src, flags);

  containers_[name] = c;
  return containers_[name];
}

template <typename T>
std::shared_ptr<T> DataCollection<T>::Add(const std::string &name,
                                          const std::shared_ptr<T> &src) {
  // error check for duplicate names
  auto it = containers_.find(name);
  if (it != containers_.end()) {
    // check to make sure they are the same
    if (!(*src == *(it->second))) {
      throw std::runtime_error("Error attempting to add a Container to a Collection");
    }
    return it->second;
  }

  auto c = std::make_shared<T>();
  c->Copy(src);

  containers_[name] = c;
  return containers_[name];
}

template <>
std::shared_ptr<MeshData<Real>> &
DataCollection<MeshData<Real>>::GetOrAdd(const std::string &mbd_label,
                                         const int &partition_id) {
  const std::string label = mbd_label + "_part-" + std::to_string(partition_id);
  auto it = containers_.find(label);
  if (it == containers_.end()) {
    // TODO(someone) add caching of partitions to Mesh at some point
    const int pack_size = pmy_mesh_->DefaultPackSize();
    auto partitions = partition::ToSizeN(pmy_mesh_->block_list, pack_size);
    for (auto i = 0; i < partitions.size(); i++) {
      const std::string md_label = mbd_label + "_part-" + std::to_string(i);
      containers_[md_label] = std::make_shared<MeshData<Real>>();
      containers_[md_label]->Set(partitions[i], mbd_label);
    }
  }
  return containers_[label];
}

template class DataCollection<MeshData<Real>>;
template class DataCollection<MeshBlockData<Real>>;

} // namespace parthenon
