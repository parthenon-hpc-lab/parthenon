//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2026 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
#ifndef BVALS_COMMS_BOUNDARY_FLUX_HPP_
#define BVALS_COMMS_BOUNDARY_FLUX_HPP_

#include "bvals/comms/bnd_info.hpp"
#include "bvals/comms/bvals_in_one.hpp"
#include "interface/mesh_data.hpp"

namespace parthenon {
TaskStatus PrepareBoundaryFluxBuffers(std::shared_ptr<MeshData<Real>> &md, bool send);

// For a single BoundaryFlux field, evaluate flux(block, dir, t, u, v, k, j, i)
// at fine faces and area-restrict directly into communication buffers. The
// device callback returns positive-coordinate flux density from valid ghosts.
template <class Flux>
TaskStatus LoadAndSendBoundaryFluxes(std::shared_ptr<MeshData<Real>> &md, Flux flux) {
  if (PrepareBoundaryFluxBuffers(md, true) != TaskStatus::complete)
    return TaskStatus::incomplete;
  auto &cache = md->GetBvarsCache().GetSubCache(BoundaryType::flxcor_send, true);
  auto info = cache.bnd_info;
  Kokkos::parallel_for(
      "RestrictBoundaryFluxes",
      Kokkos::TeamPolicy<>(DevExecSpace(), info.size(), Kokkos::AUTO),
      KOKKOS_LAMBDA(team_mbr_t team) {
        const auto &bi = info(team.league_rank());
        const auto &idxer = bi.idxer[0];
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team, idxer.size()), [&](const int index) {
              const auto [t, u, v, ck, cj, ci] = idxer(index);
              const int i = bi.fine_offset[0] + bi.refinement_factor[0] * ci;
              const int j = bi.fine_offset[1] + bi.refinement_factor[1] * cj;
              const int k = bi.fine_offset[2] + bi.refinement_factor[2] * ck;
              const int ni = bi.dir == X1DIR ? 1 : bi.refinement_factor[0];
              const int nj = bi.dir == X2DIR ? 1 : bi.refinement_factor[1];
              const int nk = bi.dir == X3DIR ? 1 : bi.refinement_factor[2];
              Real area = 0.0, integral = 0.0;
              for (int dk = 0; dk < nk; ++dk)
                for (int dj = 0; dj < nj; ++dj)
                  for (int di = 0; di < ni; ++di) {
                    const Real a = bi.coords.FaceArea(bi.dir, k + dk, j + dj, i + di);
                    area += a;
                    integral +=
                        a * flux(bi.block_index, bi.dir, t, u, v, k + dk, j + dj, i + di);
                  }
              bi.buf(index) = area != 0.0 ? integral / area : 0.0;
            });
      });
  cache.boundary_flux_loaded = true;
  return SendBoundBufsNoRestrict<BoundaryType::flxcor_send>(md);
}

// Device accessor for all corrected faces touching one coarse cell. Component
// values remain in the communication buffers; this holds only six small indices.
struct CellBoundaryFluxes {
  KOKKOS_DEFAULTED_FUNCTION CellBoundaryFluxes() = default;
  BndInfoArr_t info;
  int buffer[6]{-1, -1, -1, -1, -1, -1};
  int offset[6]{}, stride[6]{};
  Real sign[6]{};

  KOKKOS_INLINE_FUNCTION bool HasFace(int face) const { return buffer[face] >= 0; }
  KOKKOS_INLINE_FUNCTION Real operator()(int face, int component) const {
    return sign[face] * info(buffer[face]).buf(offset[face] + component * stride[face]);
  }
};

struct BoundaryFluxes {
  BndInfoArr_t info;
  ParArray1D<int> head;

  KOKKOS_INLINE_FUNCTION CellBoundaryFluxes ForCell(int block, int k, int j, int i) const {
    CellBoundaryFluxes cell;
    cell.info = info;
    if (head.size() == 0) return cell;
    for (int n = head(block); n >= 0; n = info(n).next_boundary_flux) {
      const auto &bi = info(n);
      const auto &idx = bi.idxer[0];
      const auto orientation = bi.lcoord_trans.InverseTransform(bi.topo_idx[0]);
      const int dir = static_cast<int>(std::get<0>(orientation)) % 3 + 1;
      const int side = bi.face_side;
      const int il = i + (side > 0 && dir == X1DIR);
      const int jl = j + (side > 0 && dir == X2DIR);
      const int kl = k + (side > 0 && dir == X3DIR);
      const auto [ib, jb, kb] = bi.lcoord_trans.Transform(std::array<int, 3>{il, jl, kl});
      if (ib < idx.StartIdx<5>() || ib > idx.EndIdx<5>() ||
          jb < idx.StartIdx<4>() || jb > idx.EndIdx<4>() ||
          kb < idx.StartIdx<3>() || kb > idx.EndIdx<3>() ||
          !idx.IsActive(kl, jl, il)) continue;
      const int face = 2 * (dir - 1) + (side > 0);
      cell.buffer[face] = n;
      cell.offset[face] = idx.GetFlatIdx(0, 0, 0, kb, jb, ib);
      cell.stride[face] = idx.GetFlatIdx(0, 0, 1, kb, jb, ib) - cell.offset[face];
      cell.sign[face] = std::get<1>(orientation);
    }
    return cell;
  }
};

// After ReceiveFluxCorrections, launch a cell-owned kernel through the host
// callback. The device accessor is valid until this call releases the buffers.
template <class Apply>
TaskStatus ApplyBoundaryFluxes(std::shared_ptr<MeshData<Real>> &md, Apply apply) {
  PrepareBoundaryFluxBuffers(md, false);
  auto &cache = md->GetBvarsCache().GetSubCache(BoundaryType::flxcor_recv, false);
  apply(BoundaryFluxes{cache.bnd_info, cache.boundary_flux_head});
  // A local sender can reuse its buffer as soon as it is marked stale.
  Kokkos::fence();
  for (auto *buffer : cache.buf_vec)
    buffer->Stale();
  return TaskStatus::complete;
}
} // namespace parthenon
#endif
