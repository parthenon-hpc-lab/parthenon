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
std::string DataCollection<T>::GetKey(const std::string &stage_label,
                                      const std::shared_ptr<BlockListPartition> &in) {
  auto key = stage_label;
  if (in->grid.type == GridType::two_level_composite)
    key = key + "_gmg-" + std::to_string(in->grid.logical_level);
  for (const auto &pmb : in->block_list)
    key += "_" + std::to_string(pmb->gid);
  return key;
}

template <typename T>
std::string DataCollection<T>::GetKey(const std::string &stage_label,
                                      const std::shared_ptr<MeshData<Real>> &in) {
  auto key = stage_label;
  if (in->grid.type == GridType::two_level_composite)
    key = key + "_gmg-" + std::to_string(in->grid.logical_level);
  for (const auto &pmbd : in->GetAllBlockData())
    key += "_" + std::to_string(pmbd->GetBlockPointer()->gid);
  return key;
}

template <>
std::shared_ptr<MeshData<Real>> &
DataCollection<MeshData<Real>>::GetOrAdd(const std::string &mbd_label,
                                         const int &partition_id) {
  return Add(mbd_label,
             pmy_mesh_->GetDefaultBlockPartitions(GridIdentifier::leaf())[partition_id]);
}

template <>
std::shared_ptr<MeshData<Real>> &
DataCollection<MeshData<Real>>::GetOrAdd(int gmg_level, const std::string &mbd_label,
                                         const int &partition_id) {
  return Add(mbd_label,
             pmy_mesh_->GetDefaultBlockPartitions(
                 GridIdentifier::two_level_composite(gmg_level))[partition_id]);
}

template class DataCollection<MeshData<Real>>;
template class DataCollection<MeshBlockData<Real>>;

} // namespace parthenon
