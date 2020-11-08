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

#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "defs.hpp"
#include "interface/metadata.hpp"
#include "interface/params.hpp"
#include "interface/state_descriptor.hpp"

#include "kokkos_abstraction.hpp"

namespace parthenon {

template <typename T>
class MeshBlockData;

namespace Update {

template <typename T>
TaskStatus FluxDivergence(std::shared_ptr<T> &in, std::shared_ptr<T> &dudt_obj);

template <typename T>
void UpdateIndependentData(T &in, T &dudt, const Real dt, T &out) {
  std::vector<MetadataFlag> flags({Metadata::Independent});
  auto in_pack = in->PackVariables(flags);
  auto out_pack = out->PackVariables(flags);
  auto dudt_pack = dudt->PackVariables(flags);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "UpdateMeshData", DevExecSpace(), 0, in_pack.GetDim(5) - 1, 0,
      in_pack.GetDim(4) - 1, 0, in_pack.GetDim(3) - 1, 0, in_pack.GetDim(2) - 1, 0,
      in_pack.GetDim(1) - 1,
      KOKKOS_LAMBDA(const int b, const int l, const int k, const int j, const int i) {
        out_pack(b, l, k, j, i) = in_pack(b, l, k, j, i) + dt * dudt_pack(b, l, k, j, i);
      });
}

template <typename T>
void AverageIndependentData(T &c1, T &c2, const Real wgt1) {
  std::vector<MetadataFlag> flags({Metadata::Independent});
  auto c1_pack = c1->PackVariables(flags);
  auto c2_pack = c2->PackVariables(flags);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "AverageMeshData", DevExecSpace(), 0, c1_pack.GetDim(5) - 1,
      0, c1_pack.GetDim(4) - 1, 0, c1_pack.GetDim(3) - 1, 0, c1_pack.GetDim(2) - 1, 0,
      c1_pack.GetDim(1) - 1,
      KOKKOS_LAMBDA(const int b, const int l, const int k, const int j, const int i) {
        c1_pack(b, l, k, j, i) =
            wgt1 * c1_pack(b, l, k, j, i) + (1 - wgt1) * c2_pack(b, l, k, j, i);
      });
}

template <typename T>
Real EstimateTimestep(T &rc) {
  auto pm = rc->GetGridPointer();
  Real dt_min = std::numeric_limits<Real>::max();
  for (auto &pkg : pm->packages) {
    auto &desc = pkg.second;
    if (desc->EstimateTimestep != nullptr) {
      Real dt = desc->EstimateTimestep(rc);
      dt_min = std::min(dt_min, dt);
    }
  }
  return dt_min;
}

template <typename T>
using DeriveFuncType = void(std::shared_ptr<T> &);

template <typename T>
TaskStatus FillDerived(std::shared_ptr<T> &rc) {
  using DeriveFunc_t = DeriveFuncType<T>;
  using Desc_t = std::shared_ptr<StateDescriptor>;
  Desc_t &app_pkg = rc->GetGridPointer()->packages["AppInput"];
  auto &params = app_pkg->AllParams();
  std::string mytype = typeid(rc).name();
  std::string suffix = (mytype.find("Block") != std::string::npos ? "Block" : "Mesh");
  if (params.hasKey("PreFillDerived" + suffix)) {
    params.Get<DeriveFunc_t *>("PreFillDerived" + suffix)(rc);
  }
  auto gp = rc->GetGridPointer();
  // type deduction fails if auto is used below
  for (const std::pair<std::string, Desc_t> &pkg : gp->packages) {
    auto &p = pkg.second->AllParams();
    if (p.hasKey("FillDerived" + suffix)) {
      p.Get<DeriveFunc_t *>("FillDerived" + suffix)(rc);
    }
  }
  if (params.hasKey("PostFillDerived" + suffix)) {
    params.Get<DeriveFunc_t *>("PostFillDerived" + suffix)(rc);
  }
  return TaskStatus::complete;
}

} // namespace Update

namespace FillDerivedVariables {
/*
using FillDerivedFunc = void(std::shared_ptr<MeshBlockData<Real>> &);
void SetFillDerivedFunctions(FillDerivedFunc *pre, FillDerivedFunc *post);
TaskStatus FillDerived(std::shared_ptr<MeshBlockData<Real>> &rc);
*/
} // namespace FillDerivedVariables

} // namespace parthenon

#endif // INTERFACE_UPDATE_HPP_
