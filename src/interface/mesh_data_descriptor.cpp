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

#include <memory>
#include <string>
#include <vector>

#include "basic_types.hpp"
#include "interface/data_collection.hpp"
#include "interface/mesh_data.hpp"
#include "interface/metadata.hpp"
#include "mesh/mesh.hpp"
#include "mesh_data_descriptor.hpp"
#include "utils/unique_id.hpp"
#include "utils/utils.hpp"

namespace parthenon {
std::vector<Uid_t>
MeshDataDescriptor::AddMeshData(Mesh *pmesh, const std::string &name,
                                const std::shared_ptr<MeshData<Real>> &base) {
  std::vector<std::string> resolved_vars =
      pmesh->GetVariableNames(varnames, flags, sparse_ids);
  std::shared_ptr<MeshData<Real>> md;
  if (shallow) {
    md = pmesh->mesh_data.AddShallow(name, base, resolved_vars);
  } else {
    md = pmesh->mesh_data.Add(name, base, resolved_vars);
  }
  // cache the uids for later reference
  if (uids_.size() == 0) {
    auto uids = UidIntersection(base.get(), md.get());
    uids_ = uids;
  }
  return uids_;
}
} // namespace parthenon
