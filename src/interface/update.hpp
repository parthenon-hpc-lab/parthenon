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
#ifndef INTERFACE_UPDATE_HPP_
#define INTERFACE_UPDATE_HPP_

#include <memory>
#include <string>
#include <vector>

#include "defs.hpp"
#include "interface/container.hpp"
#include "mesh/mesh.hpp"

namespace parthenon {

namespace Update {
// this does not belong here and should be moved to a better place
// currently causes problems from recursive includes (from mesh.hpp)
using BlockList_t = std::vector<std::shared_ptr<MeshBlock>>;

TaskStatus FluxDivergence(std::shared_ptr<Container<Real>> &in,
                          std::shared_ptr<Container<Real>> &dudt_cont);
auto FluxDivergenceMesh(const MeshBlockVarFluxPack<Real> &in_pack,
                        MeshBlockVarPack<Real> &dudt_pack) -> TaskStatus;
void UpdateContainer(std::shared_ptr<Container<Real>> &in,
                     std::shared_ptr<Container<Real>> &dudt_cont, const Real dt,
                     std::shared_ptr<Container<Real>> &out);
void UpdateContainer(const MeshBlockVarPack<Real> &in_pack,
                     const MeshBlockVarPack<Real> &dudt_pack, const Real dt,
                     MeshBlockVarPack<Real> &out_pack);
void AverageContainers(MeshBlockVarPack<Real> &c1_pack,
                       const MeshBlockVarPack<Real> &c2_pack, const Real wgt1);
Real EstimateTimestep(std::shared_ptr<Container<Real>> &rc);

} // namespace Update

namespace FillDerivedVariables {

using FillDerivedFunc = void(std::shared_ptr<Container<Real>> &);
void SetFillDerivedFunctions(FillDerivedFunc *pre, FillDerivedFunc *post);
TaskStatus FillDerived(std::shared_ptr<Container<Real>> &rc);

} // namespace FillDerivedVariables

} // namespace parthenon

#endif // INTERFACE_UPDATE_HPP_
