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

#include "interface/mesh_data.hpp"

#include "mesh/mesh.hpp"

#include <string>

namespace parthenon {

/*template <typename T>
MeshData<T>::MeshData(const Mesh *pmesh, const std::string &name) : pmy_mesh(pmesh) {
  const int size = pmesh->block_list.size();
  block_data_.resize(size);
  for (int i = 0; i < size; i++) {
    block_data_[i] = pmesh->block_list[i]->meshblock_data.Get(name);
  }
}
*/

} // namespace parthenon
