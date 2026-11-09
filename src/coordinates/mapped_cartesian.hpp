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

#ifndef COORDINATES_MAPPED_CARTESIAN_HPP_
#define COORDINATES_MAPPED_CARTESIAN_HPP_

#include <array>
#include <cmath>

#include "geometry_table.hpp"
#include "mapped_patch_3d.hpp"
#include "uniform_coordinates.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

// 3D Cartesian coordinates on a logically rectangular mapped mesh.  Each block (tree)
// carries a MappedPatch3D from its logical coordinates (X1, X2, X3) to physical (x, y, z).
// Cells are the images of the logical cells under the map, with curved edges and faces.
//   - Face vectors are int n dA = (1/2) loop int x cross dx around the face boundary,
//     which depends only on the boundary, as a sum of edge integrals evaluated by Gauss
//     quadrature.  Each edge integral is shared by the faces it bounds, so the outward
//     face vectors of a cell sum to zero to roundoff (a uniform state is preserved).
//   - Volumes and first moments are Gauss quadratures of the Jacobian over the cell.
// As these take many evaluations of the map, the (k, j, i) geometry of the block and its
// ghosts is tabulated when the coordinates are made (see GeometryTable).
//
// Conventions (as for MappedCylindrical):
//   - Single-index accessors (Xf<dir>(i), Xc<dir>(i), Dx, Dxc, Dxf, X<dir, el>) return
//     *logical* coordinates; they are used for indexing and prolongation.
//   - (k, j, i) accessors (Xc, Xf, FaceArea, CellVolume, CellWidth, EdgeLength) return
//     *physical* quantities.
//   - FaceNormal<dir>(k, j, i) returns the unit normal of the dir-face at the lower side
//     of cell (k, j, i) in the global (x, y, z) basis, pointing towards increasing
//     logical index.
class MappedCartesian : public UniformCoordinates<MappedCartesian> {
  using base_t = UniformCoordinates<MappedCartesian>;

 public:
  using base_t::CellWidth;
  using base_t::Dx;
  using base_t::Dxc;
  using base_t::FaceArea;
  using base_t::Scale;
  using base_t::Volume;
  using base_t::Xc;
  using base_t::Xf;
  MappedCartesian() = default;
  MappedCartesian(const RegionSize &rs, ParameterInput *pin)
      : MappedCartesian(rs, pin, MappedPatch3D()) {}
  MappedCartesian(const RegionSize &rs, ParameterInput *pin, const MappedPatch3D &patch)
      : base_t(rs, pin), patch_(patch), orient_(patch.Orientation()),
        ncells_(GeometryTable::BlockCells(rs, GetStartIndex())) {
    Tabulate();
  }
  MappedCartesian(const MappedCartesian &src, int coarsen)
      : base_t(src, coarsen), patch_(src.patch_), orient_(src.orient_),
        ncells_(GeometryTable::CoarseCells(src.ncells_, src.GetStartIndex(), coarsen)) {
    Tabulate();
  }
  constexpr static const char *name_ = "MappedCartesian";

  KOKKOS_INLINE_FUNCTION const MappedPatch3D &Patch() const { return patch_; }

  //----------------------------------------
  // Map: physical (x, y, z) of the logical point (x1, x2, x3)
  //----------------------------------------
  KOKKOS_FORCEINLINE_FUNCTION void Map(const Real x1, const Real x2, const Real x3,
                                       Real &x, Real &y, Real &z) const {
    Real p[3];
    patch_.Map(x1, x2, x3, p);
    x = p[0];
    y = p[1];
    z = p[2];
  }

  //----------------------------------------
  // PhysicalToLogical: logical (x1, x2, x3) of the physical point (x, y, z), the inverse
  // of Map
  //----------------------------------------
  KOKKOS_FORCEINLINE_FUNCTION void PhysicalToLogical(const Real x, const Real y,
                                                     const Real z, Real &x1, Real &x2,
                                                     Real &x3) const {
    const Real p[3] = {x, y, z};
    Real l[3];
    patch_.InverseMap(p, l);
    x1 = l[0];
    x2 = l[1];
    x3 = l[2];
  }

  //----------------------------------------
  // Vertex: physical (x, y, z) of the node at the lower corner of cell (k, j, i)
  //----------------------------------------
  KOKKOS_FORCEINLINE_FUNCTION void Vertex(const int k, const int j, const int i, Real &x,
                                          Real &y, Real &z) const {
    if (geom_.Has(k, j, i)) {
      x = geom_(G_V, k, j, i);
      y = geom_(G_V + 1, k, j, i);
      z = geom_(G_V + 2, k, j, i);
      return;
    }
    Map(base_t::Xf<X1DIR>(i), base_t::Xf<X2DIR>(j), base_t::Xf<X3DIR>(k), x, y, z);
  }

  //----------------------------------------
  // LogicalBoxMoments: m[0] = int dV and m[1 + d] = int x_d dV over the image of the
  // logical box [lo, hi]
  //----------------------------------------
  KOKKOS_FORCEINLINE_FUNCTION void LogicalBoxMoments(const Real lo[3], const Real hi[3],
                                                     Real m[4]) const {
    // 3-point Gauss-Legendre on [0, 1]
    const Real gx[3] = {0.5 - 0.5 * 0.7745966692414834, 0.5,
                        0.5 + 0.5 * 0.7745966692414834};
    const Real gw[3] = {5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0};
    const Real d0 = hi[0] - lo[0], d1 = hi[1] - lo[1], d2 = hi[2] - lo[2];
    const Real lvol = d0 * d1 * d2;
    for (int n = 0; n < 4; ++n)
      m[n] = 0.0;
    for (int qk = 0; qk < 3; ++qk) {
      for (int qj = 0; qj < 3; ++qj) {
        for (int qi = 0; qi < 3; ++qi) {
          const Real p1 = lo[0] + gx[qi] * d0, p2 = lo[1] + gx[qj] * d1;
          const Real p3 = lo[2] + gx[qk] * d2;
          Real x[3], J[3][3];
          patch_.Map(p1, p2, p3, x);
          patch_.Jacobian(p1, p2, p3, J);
          const Real det = J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1]) -
                           J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0]) +
                           J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0]);
          const Real w = gw[qi] * gw[qj] * gw[qk] * orient_ * det * lvol;
          m[0] += w;
          m[1] += w * x[0];
          m[2] += w * x[1];
          m[3] += w * x[2];
        }
      }
    }
  }

  //----------------------------------------
  // Xc: physical volume centroid
  //----------------------------------------
  template <int dir>
  KOKKOS_FORCEINLINE_FUNCTION Real Xc(const int k, const int j, const int i) const {
    static_assert(dir > 0 && dir < 4);
    if (geom_.Has(k, j, i)) return geom_(G_XC + dir - 1, k, j, i);
    Real m[4];
    Moments_(k, j, i, m);
    return m[dir] / m[0];
  }

  //----------------------------------------
  // Xf: physical coordinate dir of the (logical) midpoint of the face-face of cell
  // (k, j, i)
  //----------------------------------------
  template <int dir, int face>
  KOKKOS_FORCEINLINE_FUNCTION Real Xf(const int k, const int j, const int i) const {
    static_assert(dir > 0 && dir < 4);
    static_assert(face > 0 && face < 4);
    const Real x1 = (face == X1DIR) ? base_t::Xf<X1DIR>(i) : base_t::Xc<X1DIR>(i);
    const Real x2 = (face == X2DIR) ? base_t::Xf<X2DIR>(j) : base_t::Xc<X2DIR>(j);
    const Real x3 = (face == X3DIR) ? base_t::Xf<X3DIR>(k) : base_t::Xc<X3DIR>(k);
    Real p[3];
    patch_.Map(x1, x2, x3, p);
    return p[dir - 1];
  }
  template <int dir>
  KOKKOS_FORCEINLINE_FUNCTION Real Xf(const int k, const int j, const int i) const {
    return Xf<dir, dir>(k, j, i);
  }

  template <int dir, TopologicalElement el>
  KOKKOS_FORCEINLINE_FUNCTION Real Scale(const int k, const int j, const int i) const {
    static_assert(dir > 0 && dir < 4);
    return 1.0;
  }

  //----------------------------------------
  // CellWidth: volume over the mean area of the two dir faces
  //----------------------------------------
  template <int dir>
  KOKKOS_FORCEINLINE_FUNCTION Real CellWidth(const int k, const int j,
                                             const int i) const {
    static_assert(dir > 0 && dir < 4);
    if (geom_.Has(k, j, i)) return geom_(G_CW + dir - 1, k, j, i);
    constexpr int di = (dir == X1DIR), dj = (dir == X2DIR), dk = (dir == X3DIR);
    return CellVolume(k, j, i) /
           (0.5 * (FaceArea<dir>(k, j, i) + FaceArea<dir>(k + dk, j + dj, i + di)));
  }
  KOKKOS_FORCEINLINE_FUNCTION Real CellWidth(const int dir, const int k, const int j,
                                             const int i) const {
    if (dir == X1DIR) return CellWidth<X1DIR>(k, j, i);
    if (dir == X2DIR) return CellWidth<X2DIR>(k, j, i);
    return CellWidth<X3DIR>(k, j, i);
  }

  //----------------------------------------
  // EdgeLength: arc length of the dir edge from the node at the lower corner of cell
  // (k, j, i)
  //----------------------------------------
  template <int dir>
  KOKKOS_FORCEINLINE_FUNCTION Real EdgeLength(const int k, const int j,
                                              const int i) const {
    static_assert(dir > 0 && dir < 4);
    if (geom_.Has(k, j, i)) return geom_(G_EL + dir - 1, k, j, i);
    Real e[4];
    NodeEdge_<dir>(k, j, i, e);
    return e[3];
  }
  KOKKOS_FORCEINLINE_FUNCTION Real EdgeLength(const int dir, const int k, const int j,
                                              const int i) const {
    if (dir == X1DIR) return EdgeLength<X1DIR>(k, j, i);
    if (dir == X2DIR) return EdgeLength<X2DIR>(k, j, i);
    return EdgeLength<X3DIR>(k, j, i);
  }

  //----------------------------------------
  // FaceArea: magnitude of the face vector
  //----------------------------------------
  template <int dir>
  KOKKOS_FORCEINLINE_FUNCTION Real FaceArea(const int k, const int j, const int i) const {
    static_assert(dir > 0 && dir < 4);
    if (geom_.Has(k, j, i)) return geom_(G_FA + dir - 1, k, j, i);
    Real s[3];
    FaceVector_<dir>(k, j, i, s);
    return std::sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]);
  }

  // +1 if the logical (x1, x2, x3) ordering is right-handed in (x, y, z), -1 if mirrored
  KOKKOS_FORCEINLINE_FUNCTION Real Orientation() const { return orient_; }

  //----------------------------------------
  // FaceNormal: unit normal (x, y, z) of the dir face, pointing to increasing index, the
  // direction of the face vector
  //----------------------------------------
  template <int dir>
  KOKKOS_FORCEINLINE_FUNCTION void FaceNormal(const int k, const int j, const int i,
                                              Real &n1, Real &n2, Real &n3) const {
    static_assert(dir > 0 && dir < 4);
    if (geom_.Has(k, j, i)) {
      constexpr int g = G_N + 3 * (dir - 1);
      n1 = geom_(g, k, j, i);
      n2 = geom_(g + 1, k, j, i);
      n3 = geom_(g + 2, k, j, i);
      return;
    }
    Real s[3];
    FaceVector_<dir>(k, j, i, s);
    const Real smag = std::sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]);
    n1 = s[0] / smag;
    n2 = s[1] / smag;
    n3 = s[2] / smag;
  }

  //----------------------------------------
  // CellVolume
  //----------------------------------------
  KOKKOS_FORCEINLINE_FUNCTION Real CellVolume(const int k, const int j,
                                              const int i) const {
    if (geom_.Has(k, j, i)) return geom_(G_VOL, k, j, i);
    Real m[4];
    Moments_(k, j, i, m);
    return m[0];
  }

  KOKKOS_FORCEINLINE_FUNCTION
  Real Volume(CellLevel cl, TopologicalElement el, const int k, const int j,
              const int i) const {
    using TE = TopologicalElement;
    if (cl == CellLevel::same) {
      if (el == TE::CC)
        return CellVolume(k, j, i);
      else if (el == TE::F1)
        return FaceArea<X1DIR>(k, j, i);
      else if (el == TE::F2)
        return FaceArea<X2DIR>(k, j, i);
      else if (el == TE::F3)
        return FaceArea<X3DIR>(k, j, i);
      else if (el == TE::E1)
        return EdgeLength<X1DIR>(k, j, i);
      else if (el == TE::E2)
        return EdgeLength<X2DIR>(k, j, i);
      else if (el == TE::E3)
        return EdgeLength<X3DIR>(k, j, i);
      else if (el == TE::NN)
        return 1.0;
    } else {
      PARTHENON_FAIL(
          "Have not yet implemented fine fields for MappedCartesian coordinates.");
    }
    PARTHENON_FAIL("If you reach this point, someone has added a new value to the the "
                   "TopologicalElement enum.");
    return 0.0;
  }

  //----------------------------------------
  // Tabulate: tabulates the geometry over the cells, and the faces, edges and nodes up
  // to the upper boundary of the block's ghosts, by the direct evaluation of an
  // untabulated copy (public, as it defines a device lambda)
  //----------------------------------------
  void Tabulate() {
    const MappedCartesian eval = Untabulated_();
    geom_ = GeometryTable("MappedCartesian::geom", G_NQ, ncells_[2] + 1, ncells_[1] + 1,
                          ncells_[0] + 1);
    geom_.Fill(KOKKOS_LAMBDA(const ParArray4DRaw<Real> &q, const int k, const int j,
                             const int i) {
      q(G_VOL, k, j, i) = eval.CellVolume(k, j, i);
      q(G_XC, k, j, i) = eval.Xc<X1DIR>(k, j, i);
      q(G_XC + 1, k, j, i) = eval.Xc<X2DIR>(k, j, i);
      q(G_XC + 2, k, j, i) = eval.Xc<X3DIR>(k, j, i);
      q(G_CW, k, j, i) = eval.CellWidth<X1DIR>(k, j, i);
      q(G_CW + 1, k, j, i) = eval.CellWidth<X2DIR>(k, j, i);
      q(G_CW + 2, k, j, i) = eval.CellWidth<X3DIR>(k, j, i);
      q(G_FA, k, j, i) = eval.FaceArea<X1DIR>(k, j, i);
      q(G_FA + 1, k, j, i) = eval.FaceArea<X2DIR>(k, j, i);
      q(G_FA + 2, k, j, i) = eval.FaceArea<X3DIR>(k, j, i);
      eval.FaceNormal<X1DIR>(k, j, i, q(G_N, k, j, i), q(G_N + 1, k, j, i),
                             q(G_N + 2, k, j, i));
      eval.FaceNormal<X2DIR>(k, j, i, q(G_N + 3, k, j, i), q(G_N + 4, k, j, i),
                             q(G_N + 5, k, j, i));
      eval.FaceNormal<X3DIR>(k, j, i, q(G_N + 6, k, j, i), q(G_N + 7, k, j, i),
                             q(G_N + 8, k, j, i));
      q(G_EL, k, j, i) = eval.EdgeLength<X1DIR>(k, j, i);
      q(G_EL + 1, k, j, i) = eval.EdgeLength<X2DIR>(k, j, i);
      q(G_EL + 2, k, j, i) = eval.EdgeLength<X3DIR>(k, j, i);
      eval.Vertex(k, j, i, q(G_V, k, j, i), q(G_V + 1, k, j, i), q(G_V + 2, k, j, i));
    });
  }

 private:
  MappedPatch3D patch_;
  Real orient_ = 1.0;
  // cells (with ghosts) along each direction of the block, and the tabulated geometry
  // of its (k, j, i) cells, faces, edges and nodes
  std::array<int, 3> ncells_ = {0, 0, 0};
  GeometryTable geom_;
  enum : int {
    G_VOL = 0, // CellVolume
    G_XC = 1,  // Xc<dir>, at G_XC + dir - 1
    G_CW = 4,  // CellWidth<dir>
    G_FA = 7,  // FaceArea<dir>
    G_N = 10,  // FaceNormal<dir>, at G_N + 3 (dir - 1) + (0, 1, 2)
    G_EL = 19, // EdgeLength<dir>
    G_V = 22,  // Vertex
    G_NQ = 25
  };

  MappedCartesian Untabulated_() const {
    MappedCartesian c = *this;
    c.geom_ = GeometryTable();
    return c;
  }

  // Integrals along the image of the logical segment xa -> xb: e[0..2] = (1/2) int x
  // cross dx and e[3] = arc length, by 5-point Gauss-Legendre quadrature
  KOKKOS_FORCEINLINE_FUNCTION void EdgeIntegrals_(const Real xa[3], const Real xb[3],
                                                  Real e[4]) const {
    const Real gx[5] = {-0.9061798459386640, -0.5384693101056831, 0.0,
                        0.5384693101056831, 0.9061798459386640};
    const Real gw[5] = {0.2369268850561891, 0.4786286704993665, 0.5688888888888889,
                        0.4786286704993665, 0.2369268850561891};
    const Real dl[3] = {xb[0] - xa[0], xb[1] - xa[1], xb[2] - xa[2]};
    for (int n = 0; n < 4; ++n)
      e[n] = 0.0;
    for (int q = 0; q < 5; ++q) {
      const Real s = 0.5 * (1.0 + gx[q]), w = 0.5 * gw[q];
      const Real p1 = xa[0] + s * dl[0], p2 = xa[1] + s * dl[1], p3 = xa[2] + s * dl[2];
      Real x[3], J[3][3], dx[3];
      patch_.Map(p1, p2, p3, x);
      patch_.Jacobian(p1, p2, p3, J);
      for (int d = 0; d < 3; ++d)
        dx[d] = J[d][0] * dl[0] + J[d][1] * dl[1] + J[d][2] * dl[2];
      e[0] += 0.5 * w * (x[1] * dx[2] - x[2] * dx[1]);
      e[1] += 0.5 * w * (x[2] * dx[0] - x[0] * dx[2]);
      e[2] += 0.5 * w * (x[0] * dx[1] - x[1] * dx[0]);
      e[3] += w * std::sqrt(dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2]);
    }
  }

  // Edge integrals of the dir edge from the node at the lower corner of cell (k, j, i),
  // along increasing logical index
  template <int dir>
  KOKKOS_FORCEINLINE_FUNCTION void NodeEdge_(const int k, const int j, const int i,
                                             Real e[4]) const {
    constexpr int di = (dir == X1DIR), dj = (dir == X2DIR), dk = (dir == X3DIR);
    const Real xa[3] = {base_t::Xf<X1DIR>(i), base_t::Xf<X2DIR>(j),
                        base_t::Xf<X3DIR>(k)};
    const Real xb[3] = {base_t::Xf<X1DIR>(i + di), base_t::Xf<X2DIR>(j + dj),
                        base_t::Xf<X3DIR>(k + dk)};
    EdgeIntegrals_(xa, xb, e);
  }

  // Face vector of the dir-face at the lower side of cell (k, j, i), pointing to
  // increasing index: the edges are combined counter-clockwise about the face's logical
  // normal, which orient_ makes counter-clockwise about its physical normal
  template <int dir>
  KOKKOS_FORCEINLINE_FUNCTION void FaceVector_(const int k, const int j, const int i,
                                               Real s[3]) const {
    Real e0[4], e1[4], e2[4], e3[4];
    if constexpr (dir == X1DIR) {
      // (x2, x3) loop
      NodeEdge_<X2DIR>(k, j, i, e0);
      NodeEdge_<X3DIR>(k, j + 1, i, e1);
      NodeEdge_<X2DIR>(k + 1, j, i, e2);
      NodeEdge_<X3DIR>(k, j, i, e3);
    } else if constexpr (dir == X2DIR) {
      // (x3, x1) loop
      NodeEdge_<X3DIR>(k, j, i, e0);
      NodeEdge_<X1DIR>(k + 1, j, i, e1);
      NodeEdge_<X3DIR>(k, j, i + 1, e2);
      NodeEdge_<X1DIR>(k, j, i, e3);
    } else {
      // (x1, x2) loop
      NodeEdge_<X1DIR>(k, j, i, e0);
      NodeEdge_<X2DIR>(k, j, i + 1, e1);
      NodeEdge_<X1DIR>(k, j + 1, i, e2);
      NodeEdge_<X2DIR>(k, j, i, e3);
    }
    for (int d = 0; d < 3; ++d)
      s[d] = orient_ * ((e0[d] + e1[d]) - (e2[d] + e3[d]));
  }

  KOKKOS_FORCEINLINE_FUNCTION void Moments_(const int k, const int j, const int i,
                                            Real m[4]) const {
    const Real lo[3] = {base_t::Xf<X1DIR>(i), base_t::Xf<X2DIR>(j),
                        base_t::Xf<X3DIR>(k)};
    const Real hi[3] = {base_t::Xf<X1DIR>(i + 1), base_t::Xf<X2DIR>(j + 1),
                        base_t::Xf<X3DIR>(k + 1)};
    LogicalBoxMoments(lo, hi, m);
  }
};

} // namespace parthenon

#endif // COORDINATES_MAPPED_CARTESIAN_HPP_
