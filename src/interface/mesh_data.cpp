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
#include "mesh_data.hpp"

#include "mesh/mesh.hpp"

namespace parthenon {

template <typename T>
void MeshData<T>::Set(BlockList_t blocks, Mesh *pmesh, int ndim) {
  const int nblocks = blocks.size();
  ndim_ = ndim;
  block_data_.resize(nblocks);
  SetMeshPointer(pmesh);
  for (int i = 0; i < nblocks; i++) {
    block_data_[i] = blocks[i]->meshblock_data.Get(stage_name_);
  }
}

template <typename T>
void MeshData<T>::Set(BlockList_t blocks, Mesh *pmesh) {
  int ndim;
  if (pmesh != nullptr) {
    ndim = pmesh->ndim;
  }
  Set(blocks, pmesh, ndim);
}

template <typename T>
bool MeshData<T>::BlockDataIsWholeRank_() const {
  return block_data_.size() == (pmy_mesh_->block_list).size();
}

template class MeshData<Real>;

} // namespace parthenon
