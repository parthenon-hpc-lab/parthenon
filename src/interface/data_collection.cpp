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

#include <memory>
#include <string>

#include "interface/data_collection.hpp"
#include "interface/mesh_data.hpp"
#include "interface/meshblock_data.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "utils/partition_stl_containers.hpp"

namespace parthenon {

template <typename T>
std::string
DataCollection<T>::GetKey(const std::string &stage_label,
                          const std::shared_ptr<BlockListPartition> &in) const {
  auto key = stage_label;
  if (in->grid.IsMultigrid())
    key = key + "_gmg-" + std::to_string(in->grid.multigrid_level());
  for (const auto &pmb : in->block_list)
    key += "_" + std::to_string(pmb->gid);
  return key;
}

template <typename T>
std::string DataCollection<T>::GetKey(const std::string &stage_label,
                                      const std::shared_ptr<MeshData<Real>> &in) const {
  auto key = stage_label;
  if (in->grid.IsMultigrid())
    key = key + "_gmg-" + std::to_string(in->grid.multigrid_level());
  for (const auto &pmbd : in->GetAllBlockData())
    key += "_" + std::to_string(pmbd->GetBlockPointer()->gid);
  return key;
}

template <>
std::shared_ptr<MeshData<Real>> &
DataCollection<MeshData<Real>>::GetOrAdd(const std::string &mbd_label,
                                         const int &partition_id) {
  return Add(mbd_label,
             pmy_mesh_->GetDefaultBlockPartitions()[partition_id]);
}

template <>
std::shared_ptr<MeshData<Real>> &
DataCollection<MeshData<Real>>::GetOrAdd(int gmg_level, const std::string &mbd_label,
                                         const int &partition_id) {
  return Add(mbd_label,
             pmy_mesh_->GetMultigridBlockPartitions(gmg_level)[partition_id]);
}

template <class T>
const std::shared_ptr<T> &DataCollection<T>::Get(const std::string &name) const {
  if constexpr (std::is_same_v<T, MeshData<Real>>) {
    // Here we call Get with some arbitrary shared ptr
    // since Get doesn't use the second argument when
    // templated on MeshBlockData. Gross, I know.
    return Get(name, pmy_mesh_->GetBasePartition());
  } else {
    return Get(name, std::make_shared<int>());
  }
}

template <class T>
std::shared_ptr<T> &DataCollection<T>::Get(const std::string &name) {
  if constexpr (std::is_same_v<T, MeshData<Real>>) {
    return Get(name, pmy_mesh_->GetBasePartition());
  } else {
    // Here we call Get with some arbitrary shared ptr
    // since Get doesn't use the second argument when
    // templated on MeshBlockData. Gross, I know.
    return Get(name, std::make_shared<int>());
  }
}

template class DataCollection<MeshData<Real>>;
template class DataCollection<MeshBlockData<Real>>;

} // namespace parthenon
