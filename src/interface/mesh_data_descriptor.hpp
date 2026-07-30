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
#ifndef INTERFACE_MESH_DATA_DESCRIPTOR_HPP_
#define INTERFACE_MESH_DATA_DESCRIPTOR_HPP_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "basic_types.hpp"
#include "interface/metadata.hpp"
#include "utils/error_checking.hpp"
#include "utils/type_list.hpp"
#include "utils/unique_id.hpp"

namespace parthenon {
class Mesh;
template <typename T>
class MeshData;

class MeshDataDescriptor {
 public:
  std::vector<std::string> varnames;
  Metadata::FlagCollection flags;
  std::vector<int> sparse_ids;
  std::string origin = "base";
  bool shallow = false;
  bool request_all = false;

  template <typename... Ts>
  void RegisterVariables() {
    const bool contains_regex = (Ts::regex() || ...);
    PARTHENON_REQUIRE(!contains_regex, "Can't add variable types that have a regex");
    std::vector<std::string> new_names = {Ts::name()...};
    RegisterVariables(new_names);
  }
  template <typename... Ts>
  void RegisterVariables(const TypeList<Ts...> &tl) {
    RegisterVariables<Ts...>();
  }
  void RegisterVariables(const std::vector<std::string> &new_names) {
    varnames.insert(varnames.end(), new_names.begin(), new_names.end());
  }
  void RegisterVariables(const std::string &new_name) { varnames.push_back(new_name); }
  template <typename... Args>
  void RegisterVariables(const std::string &new_name, Args... args) {
    varnames.push_back(new_name);
    RegisterVariables(std::forward<Args>(args)...);
  }

  std::shared_ptr<MeshData<Real>>
  AddMeshData(Mesh *pmesh, const std::string &name,
              const std::shared_ptr<MeshData<Real>> &base);
  std::shared_ptr<MeshData<Real>> AddMeshData(Mesh *pmesh, const std::string &name);
  std::shared_ptr<MeshData<Real>> AddMeshData(Mesh *pmesh, const std::string &name,
                                              const int i);
  std::shared_ptr<MeshData<Real>> GetMeshData(Mesh *pmesh, const std::string &name);

  const auto &GetUids() const { return uids_; }

 private:
  std::vector<Uid_t> uids_;
};
} // namespace parthenon

#endif // INTERFACE_MESH_DATA_DESCRIPTOR_HPP_
