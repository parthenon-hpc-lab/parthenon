//========================================================================================
// (C) (or copyright) 2020-2021. Triad National Security, LLC. All rights reserved.
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
#include "globals.hpp"
#include "interface/meshblock_data.hpp"
#include "interface/metadata.hpp"
#include "interface/variable_pack.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"

#include "kokkos_abstraction.hpp"
#include "mesh/meshblock_pack.hpp"

namespace parthenon {

namespace Update {

template <>
TaskStatus FluxDivergence(MeshBlockData<Real> *in, MeshBlockData<Real> *dudt_cont) {
  std::shared_ptr<MeshBlock> pmb = in->GetBlockPointer();

  const IndexDomain interior = IndexDomain::interior;
  const IndexRange ib = in->GetBoundsI(interior);
  const IndexRange jb = in->GetBoundsJ(interior);
  const IndexRange kb = in->GetBoundsK(interior);

  const auto &vin = in->PackVariablesAndFluxes({Metadata::WithFluxes});
  auto dudt = dudt_cont->PackVariables({Metadata::WithFluxes});

  const auto &coords = pmb->coords;
  const int ndim = pmb->pmy_mesh->ndim;
  pmb->par_for(
      "FluxDivergenceBlock", 0, vin.GetDim(4) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int l, const int k, const int j, const int i) {
        if (dudt.IsAllocated(l) && vin.IsAllocated(l)) {
          dudt(l, k, j, i) = FluxDivHelper(l, k, j, i, ndim, coords, vin);
        }
      });

  return TaskStatus::complete;
}

template <>
TaskStatus FluxDivergence(MeshData<Real> *in_obj, MeshData<Real> *dudt_obj) {
  const IndexDomain interior = IndexDomain::interior;

  std::vector<MetadataFlag> flags({Metadata::WithFluxes});
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
        if (dudt.IsAllocated(m, l) && vin.IsAllocated(m, l)) {
          const auto &coords = vin.GetCoords(m);
          const auto &v = vin(m);
          dudt(m, l, k, j, i) = FluxDivHelper(l, k, j, i, ndim, coords, v);
        }
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

  auto u0 = u0_data->PackVariablesAndFluxes({Metadata::WithFluxes});
  const auto &u1 = u1_data->PackVariables({Metadata::WithFluxes});

  const auto &coords = pmb->coords;
  const int ndim = pmb->pmy_mesh->ndim;
  pmb->par_for(
      "UpdateWithFluxDivergenceBlock", 0, u0.GetDim(4) - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e, KOKKOS_LAMBDA(const int l, const int k, const int j, const int i) {
        if (u0.IsAllocated(l) && u1.IsAllocated(l)) {
          u0(l, k, j, i) = gam0 * u0(l, k, j, i) + gam1 * u1(l, k, j, i) +
                           beta_dt * FluxDivHelper(l, k, j, i, ndim, coords, u0);
        }
      });

  return TaskStatus::complete;
}

template <>
TaskStatus UpdateWithFluxDivergence(MeshData<Real> *u0_data, MeshData<Real> *u1_data,
                                    const Real gam0, const Real gam1,
                                    const Real beta_dt) {
  const IndexDomain interior = IndexDomain::interior;

  std::vector<MetadataFlag> flags({Metadata::WithFluxes});
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
        if (u0_pack.IsAllocated(m, l) && u1_pack.IsAllocated(m, l)) {
          const auto &coords = u0_pack.GetCoords(m);
          const auto &u0 = u0_pack(m);
          u0_pack(m, l, k, j, i) = gam0 * u0(l, k, j, i) + gam1 * u1_pack(m, l, k, j, i) +
                                   beta_dt * FluxDivHelper(l, k, j, i, ndim, coords, u0);
        }
      });
  return TaskStatus::complete;
}

TaskStatus SparseDealloc(MeshData<Real> *md) {
  if (!Globals::sparse_config.enabled || (md->NumBlocks() == 0)) {
    return TaskStatus::complete;
  }

  Kokkos::Profiling::pushRegion("Task_SparseDealloc");

  const IndexRange ib = md->GetBoundsI(IndexDomain::entire);
  const IndexRange jb = md->GetBoundsJ(IndexDomain::entire);
  const IndexRange kb = md->GetBoundsK(IndexDomain::entire);

  PackIndexMap map;
  const std::vector<MetadataFlag> flags({Metadata::Sparse});
  const auto &pack = md->PackVariables(flags, map);

  const int num_blocks = pack.GetDim(5);
  const int num_vars = pack.GetDim(4);
  ParArray2D<bool> is_zero("IsZero", num_blocks, num_vars);

  Kokkos::parallel_for(
      "SparseDealloc",
      Kokkos::TeamPolicy<>(parthenon::DevExecSpace(), num_blocks * num_vars,
                           Kokkos::AUTO),
      KOKKOS_LAMBDA(parthenon::team_mbr_t team_member) {
        const int idx = team_member.league_rank();
        const int b = idx / num_vars;     // block index
        const int v = idx - b * num_vars; // variable index

        const int Nj = jb.e + 1 - jb.s;
        const int NkNj = (kb.e + 1 - kb.s) * Nj;

        is_zero(b, v) = true;

        if (!pack.IsAllocated(b, v)) {
          // setting this to false so that dealloc counter will remain at 0
          is_zero(b, v) = false;
          return;
        }

        const auto &var = pack(b, v);
        const Real threshold = var.deallocation_threshold;
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange<>(team_member, NkNj), [&](const int inner_idx) {
              const int k = inner_idx / Nj;
              const int j = inner_idx - k * Nj;

              Kokkos::parallel_for(Kokkos::ThreadVectorRange(team_member, ib.s, ib.e + 1),
                                   [&](const int i) {
                                     const Real &val = var(k, j, i);
                                     if (std::abs(val) > threshold) {
                                       is_zero(b, v) = false;
                                       return;
                                     }
                                   });

              if (!is_zero(b, v)) {
                return;
              }
            });
      });

  auto is_zero_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), is_zero);

  for (int b = 0; b < num_blocks; ++b) {
    for (auto var_itr : map.Map()) {
      const auto label = var_itr.first;
      // skip the entry in the map for the sparse base name
      if (md->GetBlockData(b)->HasCellVariable(label)) {
        auto &counter = md->GetBlockData(b)->Get(label).dealloc_count;
        bool all_zero = true;
        for (int v = var_itr.second.first; v <= var_itr.second.second; ++v) {
          if (!is_zero_h(b, v)) {
            all_zero = false;
            break;
          }
        }

        if (all_zero) {
          // all components of this var are zero, increment dealloc counter
          counter += 1;
        } else {
          counter = 0;
        }

        if (counter > Globals::sparse_config.deallocation_count) {
          // this variable has been flagged for deallocation deallocation_count times in
          // a row, now deallocate it
          md->GetBlockData(b)->GetBlockPointer()->DeallocateSparse(label);
        }
      }
    }
  }

  Kokkos::Profiling::popRegion(); // Task_SparseDealloc
  return TaskStatus::complete;
}

} // namespace Update

} // namespace parthenon
