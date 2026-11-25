//========================================================================================
// (C) (or copyright) 2020-2024. Triad National Security, LLC. All rights reserved.
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
#include <vector>

#include "config.hpp"
#include "coordinates/coordinates.hpp"
#include "globals.hpp"
#include "interface/meshblock_data.hpp"
#include "interface/metadata.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "pack/sparse_pack/make_pack_descriptor.hpp"
#include "pack/sparse_pack/sparse_pack.hpp"
#include "sparse/sparse_management.hpp"

namespace parthenon {

namespace Update {

namespace impl {
template <typename T>
TaskStatus FluxDivergence(T *in, T *dudt_data) {
  if (in->NumBlocks() == 0) return TaskStatus::complete;
  const auto desc = MakeUpdatePackDescriptor(
      std::vector<MetadataFlag>{Metadata::WithFluxes, Metadata::Cell}, in);
  const auto flux_desc = MakeUpdatePackDescriptor(
      std::vector<MetadataFlag>{Metadata::WithFluxes, Metadata::Cell}, in,
      {PDOpt::WithFluxes});
  const auto vin = flux_desc.GetPack(in, false);
  const auto dudt = desc.GetPack(dudt_data, false);
  CheckPackLayouts(desc, vin, dudt);

  const IndexRange ib = in->GetBoundsI(IndexDomain::interior);
  const IndexRange jb = in->GetBoundsJ(IndexDomain::interior);
  const IndexRange kb = in->GetBoundsK(IndexDomain::interior);
  const auto entire_j = in->GetBoundsJ(IndexDomain::entire);
  const auto entire_k = in->GetBoundsK(IndexDomain::entire);
  const int ndim = entire_k.e > entire_k.s ? 3 : (entire_j.e > entire_j.s ? 2 : 1);
  ForEachCell(
      vin, ib, jb, kb,
      KOKKOS_LAMBDA(const int b, const int l, const int k, const int j, const int i) {
        dudt(b, l, k, j, i) =
            FluxDivHelper(b, l, k, j, i, ndim, vin.GetCoordinates(b), vin);
      });
  return TaskStatus::complete;
}

template <typename T>
TaskStatus UpdateWithFluxDivergence(T *u0_data, T *u1_data, const Real gam0,
                                    const Real gam1, const Real beta_dt) {
  if (u0_data->NumBlocks() == 0) return TaskStatus::complete;
  const auto desc = MakeUpdatePackDescriptor(
      std::vector<MetadataFlag>{Metadata::WithFluxes, Metadata::Cell}, u0_data);
  const auto flux_desc = MakeUpdatePackDescriptor(
      std::vector<MetadataFlag>{Metadata::WithFluxes, Metadata::Cell}, u0_data,
      {PDOpt::WithFluxes});
  const auto u0 = flux_desc.GetPack(u0_data, false);
  const auto u1 = desc.GetPack(u1_data, false);
  CheckPackLayouts(desc, u0, u1);

  const IndexRange ib = u0_data->GetBoundsI(IndexDomain::interior);
  const IndexRange jb = u0_data->GetBoundsJ(IndexDomain::interior);
  const IndexRange kb = u0_data->GetBoundsK(IndexDomain::interior);
  const auto entire_j = u0_data->GetBoundsJ(IndexDomain::entire);
  const auto entire_k = u0_data->GetBoundsK(IndexDomain::entire);
  const int ndim = entire_k.e > entire_k.s ? 3 : (entire_j.e > entire_j.s ? 2 : 1);
  ForEachCell(
      u0, ib, jb, kb,
      KOKKOS_LAMBDA(const int b, const int l, const int k, const int j, const int i) {
        u0(b, l, k, j, i) =
            gam0 * u0(b, l, k, j, i) + gam1 * u1(b, l, k, j, i) +
            beta_dt * FluxDivHelper(b, l, k, j, i, ndim, u0.GetCoordinates(b), u0);
      });
  return TaskStatus::complete;
}
} // namespace impl

template <>
TaskStatus FluxDivergence(MeshBlockData<Real> *in, MeshBlockData<Real> *dudt_data) {
  return impl::FluxDivergence(in, dudt_data);
}

template <>
TaskStatus FluxDivergence(MeshData<Real> *in, MeshData<Real> *dudt_data) {
  return impl::FluxDivergence(in, dudt_data);
}

template <>
TaskStatus UpdateWithFluxDivergence(MeshBlockData<Real> *u0_data,
                                    MeshBlockData<Real> *u1_data, const Real gam0,
                                    const Real gam1, const Real beta_dt) {
  return impl::UpdateWithFluxDivergence(u0_data, u1_data, gam0, gam1, beta_dt);
}

template <>
TaskStatus UpdateWithFluxDivergence(MeshData<Real> *u0_data, MeshData<Real> *u1_data,
                                    const Real gam0, const Real gam1,
                                    const Real beta_dt) {
  return impl::UpdateWithFluxDivergence(u0_data, u1_data, gam0, gam1, beta_dt);
}

} // namespace Update

} // namespace parthenon
