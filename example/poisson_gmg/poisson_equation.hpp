//========================================================================================
// (C) (or copyright) 2023. Triad National Security, LLC. All rights reserved.
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
#ifndef EXAMPLE_POISSON_GMG_POISSON_EQUATION_HPP_
#define EXAMPLE_POISSON_GMG_POISSON_EQUATION_HPP_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>
#include "poisson_package.hpp"

namespace poisson_package {

// This class implement methods for calculating A.x = y and returning the diagonal of A, where A is the 
// the matrix representing the discretized Poisson equation on the grid. Here we implement the Laplace 
// operator in terms of a flux divergence to (potentially) to consistently deal with coarse fine 
// boundaries on the grid.
class PoissonEquation {
 public:
  template <class x_t, class out_t, bool only_md_level = false, class TL_t>
  parthenon::TaskID Ax(TL_t &tl, parthenon::TaskID depends_on, std::shared_ptr<parthenon::MeshData<Real>> &md, bool only_interior, 
            bool do_flux_cor = false) {
    auto flux_res = tl.AddTask(depends_on, CalculateFluxes<x_t, only_md_level>,  md);
    if (do_flux_cor && !only_md_level) {
      auto start_flxcor = tl.AddTask(flux_res, parthenon::StartReceiveFluxCorrections, md);
      auto send_flxcor = tl.AddTask(flux_res, parthenon::LoadAndSendFluxCorrections, md);
      auto recv_flxcor = tl.AddTask(send_flxcor, parthenon::ReceiveFluxCorrections, md);
      flux_res = tl.AddTask(recv_flxcor, parthenon::SetFluxCorrections, md);
    }
    return tl.AddTask(flux_res, FluxMultiplyMatrix<x_t, out_t, only_md_level>, md,
                             only_interior);
  }
  
  template <class diag_t, bool only_md_level = false>
  parthenon::TaskStatus SetDiagonal(std::shared_ptr<parthenon::MeshData<Real>> &md) {
    using namespace parthenon;
    const int ndim = md->GetMeshPointer()->ndim;
    using TE = parthenon::TopologicalElement;
    TE te = TE::CC;
    IndexRange ib = md->GetBoundsI(IndexDomain::interior, te);
    IndexRange jb = md->GetBoundsJ(IndexDomain::interior, te);
    IndexRange kb = md->GetBoundsK(IndexDomain::interior, te);
  
    auto pkg = md->GetMeshPointer()->packages.Get("poisson_package");
    const auto alpha = pkg->Param<Real>("diagonal_alpha");
  
    int nblocks = md->NumBlocks();
    std::vector<bool> include_block(nblocks, true);
  
    if (only_md_level) {
      for (int b = 0; b < nblocks; ++b)
        include_block[b] =
            (md->grid.logical_level == md->GetBlockData(b)->GetBlockPointer()->loc.level());
    }
  
    auto desc = parthenon::MakePackDescriptor<diag_t, D>(md.get());
    auto pack = desc.GetPack(md.get(), include_block);
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "StoreDiagonal", DevExecSpace(), 0, pack.GetNBlocks() - 1,
        kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          const auto &coords = pack.GetCoordinates(b);
          // Build the unigrid diagonal of the matrix
          Real dx1 = coords.template Dxc<X1DIR>(k, j, i);
          Real diag_elem =
              -(pack(b, TE::F1, D(), k, j, i) + pack(b, TE::F1, D(), k, j, i + 1)) /
                  (dx1 * dx1) -
              alpha;
          if (ndim > 1) {
            Real dx2 = coords.template Dxc<X2DIR>(k, j, i);
            diag_elem -=
                (pack(b, TE::F2, D(), k, j, i) + pack(b, TE::F2, D(), k, j + 1, i)) /
                (dx2 * dx2);
          }
          if (ndim > 2) {
            Real dx3 = coords.template Dxc<X3DIR>(k, j, i);
            diag_elem -=
                (pack(b, TE::F3, D(), k, j, i) + pack(b, TE::F3, D(), k + 1, j, i)) /
                (dx3 * dx3);
          }
          pack(b, te, diag_t(), k, j, i) = diag_elem;
        });
    return TaskStatus::complete;
  }
 
 private:
  template <class var_t, bool only_md_level = false>
  static parthenon::TaskStatus CalculateFluxes(std::shared_ptr<parthenon::MeshData<Real>> &md) {
    using namespace parthenon;
    const int ndim = md->GetMeshPointer()->ndim;
    using TE = parthenon::TopologicalElement;
    TE te = TE::CC;
    IndexRange ib = md->GetBoundsI(IndexDomain::interior, te);
    IndexRange jb = md->GetBoundsJ(IndexDomain::interior, te);
    IndexRange kb = md->GetBoundsK(IndexDomain::interior, te);
  
    using TE = parthenon::TopologicalElement;
  
    int nblocks = md->NumBlocks();
    std::vector<bool> include_block(nblocks, true);
  
    if (only_md_level) {
      for (int b = 0; b < nblocks; ++b)
        include_block[b] =
            (md->grid.logical_level == md->GetBlockData(b)->GetBlockPointer()->loc.level());
    }
  
    auto desc = parthenon::MakePackDescriptor<var_t, D>(md.get(), {}, {PDOpt::WithFluxes});
    auto pack = desc.GetPack(md.get(), include_block);
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "CaclulateFluxes", DevExecSpace(), 0, pack.GetNBlocks() - 1,
        kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          const auto &coords = pack.GetCoordinates(b);
          Real dx1 = coords.template Dxc<X1DIR>(k, j, i);
          pack.flux(b, X1DIR, var_t(), k, j, i) =
              pack(b, TE::F1, D(), k, j, i) / dx1 *
              (pack(b, te, var_t(), k, j, i - 1) - pack(b, te, var_t(), k, j, i));
          if (i == ib.e)
            pack.flux(b, X1DIR, var_t(), k, j, i + 1) =
                pack(b, TE::F1, D(), k, j, i + 1) / dx1 *
                (pack(b, te, var_t(), k, j, i) - pack(b, te, var_t(), k, j, i + 1));

          if (ndim > 1) {
            Real dx2 = coords.template Dxc<X2DIR>(k, j, i);
            pack.flux(b, X2DIR, var_t(), k, j, i) =
                pack(b, TE::F2, D(), k, j, i) *
                (pack(b, te, var_t(), k, j - 1, i) - pack(b, te, var_t(), k, j, i)) / dx2;
            if (j == jb.e)
              pack.flux(b, X2DIR, var_t(), k, j + 1, i) =
                  pack(b, TE::F2, D(), k, j + 1, i) *
                  (pack(b, te, var_t(), k, j, i) - pack(b, te, var_t(), k, j + 1, i)) / dx2;
          }

          if (ndim > 2) {
            Real dx3 = coords.template Dxc<X3DIR>(k, j, i);
            pack.flux(b, X3DIR, var_t(), k, j, i) =
                pack(b, TE::F3, D(), k, j, i) *
                (pack(b, te, var_t(), k - 1, j, i) - pack(b, te, var_t(), k, j, i)) / dx3;
            if (k == kb.e)
              pack.flux(b, X2DIR, var_t(), k + 1, j, i) =
                  pack(b, TE::F3, D(), k + 1, j, i) *
                  (pack(b, te, var_t(), k, j, i) - pack(b, te, var_t(), k + 1, j, i)) / dx3;
          }
        });
    return TaskStatus::complete;
  }
  
  template <class in_t, class out_t, bool only_md_level = false>
  static parthenon::TaskStatus FluxMultiplyMatrix(std::shared_ptr<parthenon::MeshData<Real>> &md, bool only_interior) {
    using namespace parthenon;
    const int ndim = md->GetMeshPointer()->ndim;
    using TE = parthenon::TopologicalElement;
    TE te = TE::CC;
    IndexRange ib = md->GetBoundsI(IndexDomain::interior, te);
    IndexRange jb = md->GetBoundsJ(IndexDomain::interior, te);
    IndexRange kb = md->GetBoundsK(IndexDomain::interior, te);
  
    auto pkg = md->GetMeshPointer()->packages.Get("poisson_package");
    const auto alpha = pkg->Param<Real>("diagonal_alpha");
  
    int nblocks = md->NumBlocks();
    std::vector<bool> include_block(nblocks, true);
    if (only_interior) {
      for (int b = 0; b < nblocks; ++b)
        include_block[b] = md->GetBlockData(b)->GetBlockPointer()->neighbors.size() == 0;
    }
  
    if (only_md_level) {
      for (int b = 0; b < nblocks; ++b)
        include_block[b] =
            include_block[b] &&
            (md->grid.logical_level == md->GetBlockData(b)->GetBlockPointer()->loc.level());
    }
  
    auto desc =
        parthenon::MakePackDescriptor<in_t, out_t>(md.get(), {}, {PDOpt::WithFluxes});
    auto pack = desc.GetPack(md.get(), include_block);
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "FluxMultiplyMatrix", DevExecSpace(), 0,
        pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          const auto &coords = pack.GetCoordinates(b);
          Real dx1 = coords.template Dxc<X1DIR>(k, j, i);
          pack(b, te, out_t(), k, j, i) = -alpha * pack(b, te, in_t(), k, j, i);
          pack(b, te, out_t(), k, j, i) += (pack.flux(b, X1DIR, in_t(), k, j, i) -
                                            pack.flux(b, X1DIR, in_t(), k, j, i + 1)) /
                                           dx1;
  
          if (ndim > 1) {
            Real dx2 = coords.template Dxc<X2DIR>(k, j, i);
            pack(b, te, out_t(), k, j, i) += (pack.flux(b, X2DIR, in_t(), k, j, i) -
                                              pack.flux(b, X2DIR, in_t(), k, j + 1, i)) /
                                             dx2;
          }
  
          if (ndim > 2) {
            Real dx3 = coords.template Dxc<X3DIR>(k, j, i);
            pack(b, te, out_t(), k, j, i) += (pack.flux(b, X3DIR, in_t(), k, j, i) -
                                              pack.flux(b, X3DIR, in_t(), k + 1, j, i)) /
                                             dx3;
          }
        });
    return TaskStatus::complete;
  }


};

} // namespace poisson_package

#endif // EXAMPLE_POISSON_GMG_POISSON_EQUATION_HPP_
