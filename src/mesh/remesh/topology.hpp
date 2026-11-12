#ifndef MESH_REMESH_TOPOLOGY_HPP_
#define MESH_REMESH_TOPOLOGY_HPP_

#include <vector>

#include "mesh/remesh/remesh_plan.hpp"

namespace parthenon::remesh {

TopologyPlan PlanTopology(const forest::Forest &active,
                          const std::vector<TaggedLeaf> &tags,
                          const std::vector<int> &old_owners, int ndim,
                          std::uint64_t old_epoch = 0);

} // namespace parthenon::remesh

#endif // MESH_REMESH_TOPOLOGY_HPP_
