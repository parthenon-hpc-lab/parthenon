#ifndef MESH_REMESH_PLACEMENT_HPP_
#define MESH_REMESH_PLACEMENT_HPP_

#include <vector>

#include "mesh/remesh/remesh_plan.hpp"

namespace parthenon::remesh {

OwnershipMap PlaceBlocksContiguously(const std::vector<double> &costs, int nranks);
void ValidateContiguousOwnership(const OwnershipMap &ownership);

} // namespace parthenon::remesh

#endif // MESH_REMESH_PLACEMENT_HPP_
