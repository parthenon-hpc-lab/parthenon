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

#include <tuple>
#include <utility>

#include "defs.hpp"
#include "interface/metadata.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/domain.hpp"
#include "pack/make_pack_descriptor.hpp"
#include "pack/sparse_pack.hpp"

#ifndef SPARSE_SPARSE_MANAGEMENT_HPP_
#define SPARSE_SPARSE_MANAGEMENT_HPP_

namespace parthenon {
template <typename T>
auto SparseCheckIsZero(T *rc) {
  PARTHENON_INSTRUMENT
  auto control_vars = rc->GetMeshPointer()->resolved_packages->GetControlVariables();
  static auto desc = MakePackDescriptor(rc, control_vars, {Metadata::Sparse});
  auto pack = desc.GetPack(rc);
  auto packIdx = desc.GetMap();

  const IndexRange ib = rc->GetBoundsI(IndexDomain::entire);
  const IndexRange jb = rc->GetBoundsJ(IndexDomain::entire);
  const IndexRange kb = rc->GetBoundsK(IndexDomain::entire);

  ParArray2D<bool> is_zero("IsZero", pack.GetNBlocks(), pack.GetMaxNumberOfVars());
  if (pack.GetNBlocks() > 0) {
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
  }
  auto is_zero_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), is_zero);
  return std::make_tuple(pack, packIdx, control_vars, is_zero_h);
}
template <typename T>
void SparseDeallocOnCount(T *rc, std::size_t count);

extern template void SparseDeallocOnCount<MeshData<Real>>(MeshData<Real> *, std::size_t);
extern template void SparseDeallocOnCount<MeshBlockData<Real>>(MeshBlockData<Real> *,
                                                               std::size_t);

} // namespace parthenon
#endif // SPARSE_SPARSE_MANAGEMENT_HPP_
