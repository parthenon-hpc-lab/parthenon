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
void UpdateMeshBlockData(std::shared_ptr<MeshBlockData<Real>> &in,
                         std::shared_ptr<MeshBlockData<Real>> &dudt_cont, const Real dt,
                         std::shared_ptr<MeshBlockData<Real>> &out);
void AverageMeshBlockData(std::shared_ptr<MeshBlockData<Real>> &c1,
                          std::shared_ptr<MeshBlockData<Real>> &c2, const Real wgt);
Real EstimateTimestep(std::shared_ptr<MeshBlockData<Real>> &rc);

template <typename T>
void UpdateIndependentData(T &in, T &dudt, const Real dt, T &out) {
  std::vector<MetadataFlag> flags({Metadata::Independent});
  auto in_pack = in->PackVariables(flags);
  auto out_pack = out->PackVariables(flags);
  auto dudt_pack = dudt->PackVariables(flags);
  const IndexDomain interior = IndexDomain::interior;
  IndexRange ib = in->GetBoundsI(interior);
  IndexRange jb = in->GetBoundsJ(interior);
  IndexRange kb = in->GetBoundsK(interior);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "UpdateMeshData", DevExecSpace(), 0, in_pack.GetDim(5) - 1, 0,
      in_pack.GetDim(4) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int l, const int k, const int j, const int i) {
        out_pack(b, l, k, j, i) = in_pack(b, l, k, j, i) + dt * dudt_pack(b, l, k, j, i);
      });
}

template <typename T>
void AverageIndependentData(T &c1, T &c2, const Real wgt1) {
  std::vector<MetadataFlag> flags({Metadata::Independent});
  auto c1_pack = c1->PackVariables(flags);
  auto c2_pack = c2->PackVariables(flags);

  const IndexDomain interior = IndexDomain::interior;
  IndexRange ib = c1->GetBoundsI(interior);
  IndexRange jb = c1->GetBoundsJ(interior);
  IndexRange kb = c1->GetBoundsK(interior);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "AverageMeshData", DevExecSpace(), 0, c1_pack.GetDim(5) - 1,
      0, c1_pack.GetDim(4) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int l, const int k, const int j, const int i) {
        c1_pack(b, l, k, j, i) =
            wgt1 * c1_pack(b, l, k, j, i) + (1 - wgt1) * c2_pack(b, l, k, j, i);
      });
}

} // namespace Update

namespace FillDerivedVariables {

using FillDerivedFunc = void(std::shared_ptr<MeshBlockData<Real>> &);
void SetFillDerivedFunctions(FillDerivedFunc *pre, FillDerivedFunc *post);
TaskStatus FillDerived(std::shared_ptr<MeshBlockData<Real>> &rc);

} // namespace FillDerivedVariables

} // namespace parthenon

#endif // INTERFACE_UPDATE_HPP_
