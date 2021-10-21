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

#include "interface/update.hpp"

#include <memory>

#include "config.hpp"
#include "coordinates/coordinates.hpp"
#include "interface/meshblock_data.hpp"
#include "interface/metadata.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"

#include "kokkos_abstraction.hpp"
#include "mesh/meshblock_pack.hpp"

namespace parthenon {

namespace Update {

KOKKOS_FORCEINLINE_FUNCTION
Real FluxDiv_(const int l, const int k, const int j, const int i, const int ndim,
              const Coordinates_t &coords, const VariableFluxPack<Real> &v) {
  Real du = (coords.Area(X1DIR, k, j, i + 1) * v.flux(X1DIR, l, k, j, i + 1) -
             coords.Area(X1DIR, k, j, i) * v.flux(X1DIR, l, k, j, i));
  if (ndim >= 2) {
    du += (coords.Area(X2DIR, k, j + 1, i) * v.flux(X2DIR, l, k, j + 1, i) -
           coords.Area(X2DIR, k, j, i) * v.flux(X2DIR, l, k, j, i));
  }
  if (ndim == 3) {
    du += (coords.Area(X3DIR, k + 1, j, i) * v.flux(X3DIR, l, k + 1, j, i) -
           coords.Area(X3DIR, k, j, i) * v.flux(X3DIR, l, k, j, i));
  }
  return -du / coords.Volume(k, j, i);
}

template <>
TaskStatus FluxDivergence(MeshBlockData<Real> *in, MeshBlockData<Real> *dudt_cont) {
  std::shared_ptr<MeshBlock> pmb = in->GetBlockPointer();

  const IndexDomain interior = IndexDomain::interior;
  const IndexRange ib = in->GetBoundsI(interior);
  const IndexRange jb = in->GetBoundsJ(interior);
  const IndexRange kb = in->GetBoundsK(interior);

  const auto &vin = in->PackVariablesAndFluxes({Metadata::Independent});
  auto dudt = dudt_cont->PackVariables({Metadata::Independent});

  const auto &coords = pmb->coords;
  const int ndim = pmb->pmy_mesh->ndim;
  pmb->par_for(
      "FluxDivergenceBlock", 0, vin.GetDim(4) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int l, const int k, const int j, const int i) {
        dudt(l, k, j, i) = FluxDiv_(l, k, j, i, ndim, coords, vin);
      });

  return TaskStatus::complete;
}

template <>
TaskStatus FluxDivergence(MeshData<Real> *in_obj, MeshData<Real> *dudt_obj) {
  const IndexDomain interior = IndexDomain::interior;

  std::vector<MetadataFlag> flags({Metadata::Independent});
  const auto &vin = in_obj->PackVariablesAndFluxes(flags);
  auto dudt = dudt_obj->PackVariables(flags);
  const IndexRange ib = in_obj->GetBoundsI(interior);
  const IndexRange jb = in_obj->GetBoundsJ(interior);
  const IndexRange kb = in_obj->GetBoundsK(interior);

  const int ndim = vin.GetNdim();
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "FluxDivergenceMesh", DevExecSpace(), 0, vin.GetDim(5) - 1, 0,
      vin.GetDim(4) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int m, const int l, const int k, const int j, const int i) {
        const auto &coords = vin.coords(m);
        const auto &v = vin(m);
        dudt(m, l, k, j, i) = FluxDiv_(l, k, j, i, ndim, coords, v);
      });
  return TaskStatus::complete;
}

template <>
TaskStatus UpdateWithFluxDivergence(MeshBlockData<Real> *u0_data,
                                    MeshBlockData<Real> *u1_data, const Real gam0,
                                    const Real gam1, const Real beta_dt) {
  std::shared_ptr<MeshBlock> pmb = u0_data->GetBlockPointer();

  const IndexDomain interior = IndexDomain::interior;
  const IndexRange ib = u0_data->GetBoundsI(interior);
  const IndexRange jb = u0_data->GetBoundsJ(interior);
  const IndexRange kb = u0_data->GetBoundsK(interior);

  auto u0 = u0_data->PackVariablesAndFluxes({Metadata::Independent});
  const auto &u1 = u1_data->PackVariables({Metadata::Independent});

  const auto &coords = pmb->coords;
  const int ndim = pmb->pmy_mesh->ndim;
  pmb->par_for(
      "UpdateWithFluxDivergenceBlock", 0, u0.GetDim(4) - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e, KOKKOS_LAMBDA(const int l, const int k, const int j, const int i) {
        u0(l, k, j, i) = gam0 * u0(l, k, j, i) + gam1 * u1(l, k, j, i) +
                         beta_dt * FluxDiv_(l, k, j, i, ndim, coords, u0);
      });

  return TaskStatus::complete;
}

template <>
TaskStatus UpdateWithFluxDivergence(MeshData<Real> *u0_data, MeshData<Real> *u1_data,
                                    const Real gam0, const Real gam1,
                                    const Real beta_dt) {
  const IndexDomain interior = IndexDomain::interior;

  std::vector<MetadataFlag> flags({Metadata::Independent});
  auto u0_pack = u0_data->PackVariablesAndFluxes(flags);
  const auto &u1_pack = u1_data->PackVariables(flags);
  const IndexRange ib = u0_data->GetBoundsI(interior);
  const IndexRange jb = u0_data->GetBoundsJ(interior);
  const IndexRange kb = u0_data->GetBoundsK(interior);

  const int ndim = u0_pack.GetNdim();
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "UpdateWithFluxDivergenceMesh", DevExecSpace(), 0,
      u0_pack.GetDim(5) - 1, 0, u0_pack.GetDim(4) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int m, const int l, const int k, const int j, const int i) {
        const auto &coords = u0_pack.coords(m);
        const auto &u0 = u0_pack(m);
        u0_pack(m, l, k, j, i) = gam0 * u0(l, k, j, i) + gam1 * u1_pack(m, l, k, j, i) +
                                 beta_dt * FluxDiv_(l, k, j, i, ndim, coords, u0);
      });
  return TaskStatus::complete;
}

} // namespace Update

} // namespace parthenon
