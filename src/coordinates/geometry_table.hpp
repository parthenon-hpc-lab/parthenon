//========================================================================================
// (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
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
// This file was made in part with generative AI.

#ifndef COORDINATES_GEOMETRY_TABLE_HPP_
#define COORDINATES_GEOMETRY_TABLE_HPP_

#include <array>

#include "basic_types.hpp"
#include "defs.hpp"
#include "kokkos_types.hpp"

#include <Kokkos_Core.hpp>

namespace parthenon {

// Geometric quantities q(n, k, j, i) of the cells, faces and nodes (k, j, i) of a block,
// tabulated by mapped coordinates whose geometry is costly to evaluate.  The coordinates
// own the table, so it is built with them when a block (or its coarse buffer) is made,
// which includes the blocks made by remeshing, load balancing and restarts, and copies
// of the coordinates share it.  A default table is empty, Has() is then false and the
// coordinates evaluate the geometry directly.  The table is kept in device memory and
// mirrored on the host, so the geometry can be read from host code (e.g., problem
// generators and boundary setup) as well as from kernels.
class GeometryTable {
 public:
  GeometryTable() = default;
  // a table of nq quantities over the (nk, nj, ni) indices from 0
  GeometryTable(const char *label, const int nq, const int nk, const int nj, const int ni)
      : q_(label, nq, nk, nj, ni), q_h_(Kokkos::create_mirror_view(q_)), nk_(nk),
        nj_(nj), ni_(ni) {}

  // Whether (k, j, i) is in the table
  KOKKOS_FORCEINLINE_FUNCTION bool Has(const int k, const int j, const int i) const {
    return static_cast<unsigned>(i) < static_cast<unsigned>(ni_) &&
           static_cast<unsigned>(j) < static_cast<unsigned>(nj_) &&
           static_cast<unsigned>(k) < static_cast<unsigned>(nk_);
  }
  KOKKOS_FORCEINLINE_FUNCTION Real operator()(const int n, const int k, const int j,
                                              const int i) const {
    KOKKOS_IF_ON_DEVICE((return q_(n, k, j, i);))
    KOKKOS_IF_ON_HOST((return q_h_(n, k, j, i);))
  }

  // Fills the table, f(q, k, j, i) setting q(n, k, j, i) for every n
  template <typename F>
  void Fill(const F &f) {
    auto q = q_;
    Kokkos::parallel_for(
        "GeometryTable::Fill",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>, DevExecSpace>({0, 0, 0}, {nk_, nj_, ni_}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) { f(q, k, j, i); });
    Kokkos::fence("GeometryTable::Fill");
    Kokkos::deep_copy(q_h_, q_);
  }

  // The number of cells (with ghosts) along each direction of the block of the region
  // rs with the start indices istart, and of its coarse buffer (coarsened by 2 in the
  // directions with ghosts)
  static std::array<int, 3> BlockCells(const RegionSize &rs,
                                       const std::array<int, 3> &istart) {
    return {rs.nx(X1DIR) + 2 * istart[0], rs.nx(X2DIR) + 2 * istart[1],
            rs.nx(X3DIR) + 2 * istart[2]};
  }
  static std::array<int, 3> CoarseCells(const std::array<int, 3> &ncells,
                                        const std::array<int, 3> &istart,
                                        const int coarsen) {
    std::array<int, 3> nc;
    for (int d = 0; d < 3; ++d)
      nc[d] = istart[d] > 0 ? (ncells[d] - 2 * istart[d]) / coarsen + 2 * istart[d]
                            : ncells[d];
    return nc;
  }

 private:
  ParArray4DRaw<Real> q_;
  typename ParArray4DRaw<Real>::host_mirror_type q_h_;
  int nk_ = 0, nj_ = 0, ni_ = 0;
};

} // namespace parthenon

#endif // COORDINATES_GEOMETRY_TABLE_HPP_
