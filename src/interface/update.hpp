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
#include "interface/mesh_data.hpp"
#include "interface/meshblock_data.hpp"
#include "mesh/mesh.hpp"

#include "kokkos_abstraction.hpp"

namespace parthenon {

namespace Update {

TaskStatus FluxDivergenceBlock(std::shared_ptr<MeshBlockData<Real>> &in,
                               std::shared_ptr<MeshBlockData<Real>> &dudt_cont);
TaskStatus FluxDivergenceMesh(std::shared_ptr<MeshData<Real>> &in_pack,
                              std::shared_ptr<MeshData<Real>> &dudt_pack);
void UpdateContainer(std::shared_ptr<MeshBlockData<Real>> &in,
                     std::shared_ptr<MeshBlockData<Real>> &dudt_cont, const Real dt,
                     std::shared_ptr<MeshBlockData<Real>> &out);
void UpdateContainer(std::shared_ptr<MeshData<Real>> &in,
                     std::shared_ptr<MeshData<Real>> &dudt, const Real dt,
                     std::shared_ptr<MeshData<Real>> &out);
void AverageContainers(std::shared_ptr<MeshData<Real>> &c1_pack,
                       std::shared_ptr<MeshData<Real>> &c2_pack, const Real wgt1);
Real EstimateTimestep(std::shared_ptr<MeshBlockData<Real>> &rc);

} // namespace Update

namespace FillDerivedVariables {

using FillDerivedFunc = void(std::shared_ptr<MeshBlockData<Real>> &);
void SetFillDerivedFunctions(FillDerivedFunc *pre, FillDerivedFunc *post);
TaskStatus FillDerived(std::shared_ptr<MeshBlockData<Real>> &rc);

} // namespace FillDerivedVariables

} // namespace parthenon

#endif // INTERFACE_UPDATE_HPP_
