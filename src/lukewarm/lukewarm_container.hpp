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
#ifndef LUKEWARM_LUKEWARM_CONTAINER_HPP_
#define LUKEWARM_LUKEWARM_CONTAINER_HPP_

// This file was made in part with generative AI.

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "lukewarm/lukewarm.hpp"
#include "lukewarm/lukewarm_comms.hpp"
#include "lukewarm/lukewarm_geometry.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

class StateDescriptor;

// The lukewarms owned by one MeshData, by name
class LukewarmContainer {
 public:
  LukewarmContainer() = default;
  // Build every lukewarm declared in the resolved state, with all of its fields. Each
  // lukewarm draws ids from the counter of the same name, created here if missing.
  LukewarmContainer(const StateDescriptor *resolved_packages,
                    std::map<std::string, LukewarmIdCounter> &id_counters);

  void Add(const std::shared_ptr<Lukewarm> &lukewarm);

  bool Contains(const std::string &name) const { return lukewarms_.count(name) > 0; }
  const std::shared_ptr<Lukewarm> &GetLukewarmPtr(const std::string &name) const {
    const auto itr = lukewarms_.find(name);
    PARTHENON_REQUIRE_THROWS(itr != lukewarms_.end(), "No lukewarm named " + name);
    return itr->second;
  }
  Lukewarm &GetLukewarm(const std::string &name) const { return *GetLukewarmPtr(name); }
  const auto &GetLukewarms() const { return lukewarms_; }

  LukewarmComms &GetComms(const std::string &name) {
    const auto itr = comms_.find(name);
    PARTHENON_REQUIRE_THROWS(itr != comms_.end(), "No lukewarm named " + name);
    return itr->second;
  }

  const LukewarmGeometry &GetGeometry() const { return geometry_; }
  void SetGeometry(const LukewarmGeometry &geometry) { geometry_ = geometry; }

  // The id of this MeshData, which is the smallest gid of its blocks, and the MeshData
  // next to it, sorted by id. Set by BuildLukewarmComms().
  int GetId() const { return id_; }
  const std::vector<LukewarmNeighbor> &GetNeighbors() const { return neighbors_; }
  void SetNeighbors(const int id, const std::vector<LukewarmNeighbor> &neighbors) {
    id_ = id;
    neighbors_ = neighbors;
  }

 private:
  std::map<std::string, std::shared_ptr<Lukewarm>> lukewarms_;
  std::map<std::string, LukewarmComms> comms_;
  LukewarmGeometry geometry_;
  int id_ = -1;
  std::vector<LukewarmNeighbor> neighbors_;
};

} // namespace parthenon

#endif // LUKEWARM_LUKEWARM_CONTAINER_HPP_
