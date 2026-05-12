//========================================================================================
// (C) (or copyright) 2020-2025. Triad National Security, LLC. All rights reserved.
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

#include <string>
#include <unordered_set>

#include "sparse/sparse_management.hpp"

namespace parthenon {

TaskStatus SparseDealloc(MeshData<Real> *md) {
  PARTHENON_INSTRUMENT
  if (Globals::sparse_config.enabled && (md->NumBlocks() > 0)) {
    SparseDeallocOnCount(md, Globals::sparse_config.deallocation_count);
  }
  return TaskStatus::complete;
}

template <typename T>
TaskStatus InitNewlyAllocatedVars(T *rc) {
  PARTHENON_INSTRUMENT
  if (!rc->AllVariablesInitialized()) {
    const IndexDomain interior = IndexDomain::interior;
    const IndexRange ib = rc->GetBoundsI(interior);
    const IndexRange jb = rc->GetBoundsJ(interior);
    const IndexRange kb = rc->GetBoundsK(interior);
    const int Ni = ib.e + 1 - ib.s;
    const int Nj = jb.e + 1 - jb.s;
    const int Nk = kb.e + 1 - kb.s;
    const int NjNi = Nj * Ni;
    const int NkNjNi = Nk * NjNi;

    // This pack will always be freshly built, since we only get here if sparse data
    // was allocated and hasn't been initialized, which in turn implies the cached
    // pack must be stale.
    auto desc =
        parthenon::MakePackDescriptor<variable_names::any>(rc, {Metadata::Sparse});
    auto v = desc.GetPack(rc);

    Kokkos::parallel_for(
        PARTHENON_AUTO_LABEL,
        Kokkos::TeamPolicy<>(parthenon::DevExecSpace(), v.GetNBlocks(), Kokkos::AUTO),
        KOKKOS_LAMBDA(parthenon::team_mbr_t team_member) {
          const int b = team_member.league_rank();
          int lo = v.GetLowerBound(b, variable_names::any());
          int hi = v.GetUpperBound(b, variable_names::any());

          for (int vidx = lo; vidx <= hi; ++vidx) {
            if (!v(b, vidx).initialized) {
              Real val = v(b, vidx).sparse_default_val;
              Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team_member, NkNjNi),
                                   [&](const int idx) {
                                     const int k = kb.s + idx / NjNi;
                                     const int j = jb.s + (idx % NjNi) / Ni;
                                     const int i = ib.s + idx % Ni;
                                     v(b, vidx, k, j, i) = val;
                                   });
            }
          }
        });

    // Set initialized here since everything has been filled with default values,
    // user defined functions may overwrite these in the next step but that doesn't
    // change initialization status of the interior
    rc->SetAllVariablesToInitialized();
  }

  // Do user defined initializations if present
  // This has to be done even in the case where no blocks have been allocated
  // since the boundaries of allocated blocks could have received default data
  // In any case
  auto pm = rc->GetParentPointer();
  for (const auto &pkg : pm->packages.AllPackages()) {
    pkg.second->InitNewlyAllocatedVars(rc);
  }

  // Don't worry about flagging variables as initialized
  // since they will be flagged at the beginning of the
  // next step in the evolution driver

  return TaskStatus::complete;
}

template <typename T>
void SparseDeallocOnCount(T *rc, std::size_t count,
                          const std::unordered_set<std::string> &exclude) {
  PARTHENON_INSTRUMENT
  auto control_vars = rc->GetMeshPointer()->resolved_packages->GetControlVariables();
  static auto desc = MakePackDescriptor(rc, control_vars, {Metadata::Sparse});
  auto pack = desc.GetPack(rc);
  auto packIdx = desc.GetMap();
  if (pack.GetNBlocks() < 1) return;

  const IndexRange ib = rc->GetBoundsI(IndexDomain::entire);
  const IndexRange jb = rc->GetBoundsJ(IndexDomain::entire);
  const IndexRange kb = rc->GetBoundsK(IndexDomain::entire);

  ParArray2D<bool> is_zero("IsZero", pack.GetNBlocks(), pack.GetMaxNumberOfVars());
  Kokkos::parallel_for(
      PARTHENON_AUTO_LABEL,
      Kokkos::TeamPolicy<>(parthenon::DevExecSpace(), pack.GetNBlocks(), Kokkos::AUTO),
      KOKKOS_LAMBDA(parthenon::team_mbr_t team_member) {
        const int b = team_member.league_rank();

        const int lo = pack.GetLowerBound(b);
        const int hi = pack.GetUpperBound(b);

        for (int v = lo; v <= hi; ++v) {
          const auto &var = pack(b, v);
          const Real threshold = var.deallocation_threshold;
          bool all_zero = true;
          const auto &var_raw = var.data();
          Kokkos::parallel_reduce(
              Kokkos::TeamThreadRange<>(team_member, var.size()),
              [&](const int idx, bool &lall_zero) {
                if (std::abs(var_raw[idx]) > threshold) {
                  lall_zero = false;
                  return;
                }
              },
              Kokkos::LAnd<bool, DevMemSpace>(all_zero));
          Kokkos::single(Kokkos::PerTeam(team_member),
                         [&]() { is_zero(b, v) = all_zero; });
        }
      });

  auto is_zero_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), is_zero);

  for (int b = 0; b < pack.GetNBlocks(); ++b) {
    auto pmbdata = rc->GetBlockDataRawPointer(b);
    auto pmb = pmbdata->GetBlockPointer();
    // Per group, update each member's deallocation counter using the old logic.
    // If every member in the group is ready, deallocate the whole group together.
    for (const auto &control_group :
         rc->GetMeshPointer()->resolved_packages->GetControlGroups()) {
      PARTHENON_REQUIRE_THROWS(!control_group.empty(),
                               "Encountered an empty sparse control group");
      const auto &representative = *control_group.begin();

      bool group_excluded = false;
      bool deallocate_group = true;
      for (const auto &control_var : control_group) {
        if (exclude.count(control_var.label()) > 0) {
          group_excluded = true;
          break;
        }

        int lo = pack.GetLowerBoundHost(b, PackIdx(packIdx[control_var.label()]));
        int hi = pack.GetUpperBoundHost(b, PackIdx(packIdx[control_var.label()]));
        if (lo > hi) continue; // This controller variable is not present in the pack.

        auto &counter = pmbdata->Get(control_var.label()).dealloc_count;
        bool all_zero = true;
        for (int iv = lo; iv <= hi; ++iv) all_zero = all_zero && is_zero_h(b, iv);
        if (all_zero) {
          counter++;
        } else {
          counter = 0;
        }
        deallocate_group = deallocate_group && (counter > count);
      }

      if (group_excluded || !deallocate_group) continue;

      for (const auto &control_var : control_group) {
        pmbdata->Get(control_var.label()).dealloc_count = 0;
      }
      pmb->DeallocateSparse(representative.label());
    }
  }
}

template TaskStatus InitNewlyAllocatedVars<MeshBlockData<Real>>(MeshBlockData<Real> *rc);
template TaskStatus InitNewlyAllocatedVars<MeshData<Real>>(MeshData<Real> *rc);

template void
SparseDeallocOnCount<MeshData<Real>>(MeshData<Real> *, std::size_t,
                                     const std::unordered_set<std::string> &);
template void
SparseDeallocOnCount<MeshBlockData<Real>>(MeshBlockData<Real> *, std::size_t,
                                          const std::unordered_set<std::string> &);

} // namespace parthenon
