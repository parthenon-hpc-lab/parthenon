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
#ifndef INTERFACE_SUB_MESHDATA_REQUIREMENTS_HPP_
#define INTERFACE_SUB_MESHDATA_REQUIREMENTS_HPP_

#include <memory>
#include <string>
#include <vector>

#include "basic_types.hpp"
#include "interface/metadata.hpp"
#include "utils/unique_id.hpp"

namespace parthenon {
class Mesh;
template <typename T>
class MeshData;

class SubMeshDataRequirements {
 public:
  std::vector<std::string> varnames;
  Metadata::FlagCollection flags;
  std::vector<int> sparse_ids;
  bool shallow = false;

  std::vector<Uid_t> AddMDSubset(Mesh *pmesh, const std::string &name,
                                 const std::shared_ptr<MeshData<Real>> &base);
  const auto &GetUids() const { return uids_; }

 private:
  std::vector<Uid_t> uids_;
};
} // namespace parthenon

#endif // INTERFACE_SUB_MESHDATA_REQUIREMENTS_HPP_
