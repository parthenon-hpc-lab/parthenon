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
#ifndef COORDINATES_MAPPED_CYLINDRICAL_HPP_
#define COORDINATES_MAPPED_CYLINDRICAL_HPP_

#include <array>
#include <cmath>

#include "geometry_table.hpp"
#include "mapped_patch.hpp"
#include "uniform_coordinates.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

// Axisymmetric coordinates on a logically rectangular mapped mesh.  Each block (tree)
// carries a MappedPatch from its logical coordinates (X1, X2) to the physical meridional
// plane (R, z); X3 is phi.  Cells are the images of the logical cells under the map,
// with curved edges, revolved about the z axis.  Areas, volumes and moments are line
// integrals along the cell edges (Green's theorem) evaluated by Gauss quadrature, with
// each edge integral shared by the faces and cells it bounds.  The geometry then nests
// under refinement (a coarse cell is the union of its children) and the closed-surface
// identities hold to roundoff (sum of outward face vectors = (planar area, 0) dphi).
// As these take many evaluations of the map, the (k, j, i) geometry of the block and its
// ghosts is tabulated when the coordinates are made (see GeometryTable).
//
// Conventions:
//   - Single-index accessors (Xf<dir>(i), Xc<dir>(i), Dx, Dxc, Dxf, X<dir, el>) return
//     *logical* coordinates; they are used for indexing and prolongation.
//   - (k, j, i) accessors (Xc, Xf, FaceArea, CellVolume, CellWidth, EdgeLength) return
//     *physical* quantities, with areas and volumes including the factor of dphi.
//   - FaceNormal<dir>(k, j, i) returns the unit normal of the dir-face at the lower
//     side of cell (k, j, i) in the global (R, z) basis, pointing towards increasing
//     logical index.
class MappedCylindrical : public UniformCoordinates<MappedCylindrical> {
  using base_t = UniformCoordinates<MappedCylindrical>;

 public:
  using base_t::CellWidth;
  using base_t::Dx;
  using base_t::Dxc;
  using base_t::FaceArea;
  using base_t::Scale;
  using base_t::Volume;
  using base_t::Xc;
  using base_t::Xf;
  MappedCylindrical() = default;
  MappedCylindrical(const RegionSize &rs, ParameterInput *pin)
      : MappedCylindrical(rs, pin, MappedPatch()) {}
  MappedCylindrical(const RegionSize &rs, ParameterInput *pin, const MappedPatch &patch)
      : base_t(rs, pin), patch_(patch), orient_(patch.Orientation()),
        ncells_(GeometryTable::BlockCells(rs, GetStartIndex())) {
    PARTHENON_REQUIRE(rs.xmin(X3DIR) >= 0.0, "Min phi must be >= 0.0.");
    PARTHENON_REQUIRE(rs.xmax(X3DIR) <= 2.0 * M_PI + 1.e-15, "Max phi must be <= 2pi");
    Tabulate();
  }
  MappedCylindrical(const MappedCylindrical &src, int coarsen)
      : base_t(src, coarsen), patch_(src.patch_), orient_(src.orient_),
        ncells_(GeometryTable::CoarseCells(src.ncells_, src.GetStartIndex(), coarsen)) {
    Tabulate();
  }
  constexpr static const char *name_ = "MappedCylindrical";

  KOKKOS_INLINE_FUNCTION const MappedPatch &Patch() const { return patch_; }

  //----------------------------------------
  // Map: physical (R, z) of the logical point (x1, x2)
  //----------------------------------------
  KOKKOS_FORCEINLINE_FUNCTION void Map(const Real x1, const Real x2, Real &r,
                                       Real &z) const {
    patch_.Map(x1, x2, r, z);
    r = std::abs(r);
  }

  //----------------------------------------
  // PhysicalToLogical: logical (x1, x2, x3) of the physical point (R, z, phi), the
  // inverse of Map in (R, z)
  //----------------------------------------
  KOKKOS_FORCEINLINE_FUNCTION void PhysicalToLogical(const Real r, const Real z,
                                                     const Real phi, Real &x1, Real &x2,
                                                     Real &x3) const {
    patch_.InverseMap(r, z, x1, x2);
    x3 = phi;
  }

  //----------------------------------------
  // Vertex: physical (R, z) of the node at the lower corner of cell (j, i)
  //----------------------------------------
  KOKKOS_FORCEINLINE_FUNCTION void Vertex(const int j, const int i, Real &r,
                                          Real &z) const {
    if (geom_.Has(0, j, i)) {
      r = geom_(G_VR, 0, j, i);
      z = geom_(G_VZ, 0, j, i);
      return;
    }
    Map(base_t::Xf<X1DIR>(i), base_t::Xf<X2DIR>(j), r, z);
  }

  //----------------------------------------
  // LogicalQuadMoments: m0 = int R dA, m1 = int R^2 dA, m2 = int R z dA over the image
  // of the logical rectangle [x1lo, x1hi] x [x2lo, x2hi]
  //----------------------------------------
  KOKKOS_FORCEINLINE_FUNCTION void LogicalQuadMoments(const Real x1lo, const Real x2lo,
                                                      const Real x1hi, const Real x2hi,
                                                      Real &m0, Real &m1,
                                                      Real &m2) const {
    Real a;
    RectMoments_(x1lo, x2lo, x1hi, x2hi, a, m0, m1, m2);
  }

  //----------------------------------------
  // Xc: physical volume centroid (R and z); phi is uniform
  //----------------------------------------
  template <int dir>
  KOKKOS_FORCEINLINE_FUNCTION Real Xc(const int k, const int j, const int i) const {
    static_assert(dir > 0 && dir < 4);
    if constexpr (dir == X3DIR) {
      return base_t::Xc<X3DIR>(k);
    } else {
      if (geom_.Has(0, j, i)) return geom_((dir == X1DIR) ? G_XC1 : G_XC2, 0, j, i);
      Real m0, m1, m2;
      Moments_(j, i, m0, m1, m2);
      // m1 = int R^2 dA, m2 = int R z dA, m0 = int R dA
      return (dir == X1DIR) ? m1 / m0 : m2 / m0;
    }
  }

  //----------------------------------------
  // Xf: physical coordinate dir of the (logical) midpoint of the face-face of cell
  // (k, j, i)
  //----------------------------------------
  template <int dir, int face>
  KOKKOS_FORCEINLINE_FUNCTION Real Xf(const int k, const int j, const int i) const {
    static_assert(dir > 0 && dir < 4);
    static_assert(face > 0 && face < 4);
    if constexpr (dir == X3DIR) {
      return (face == X3DIR) ? base_t::Xf<X3DIR>(k) : base_t::Xc<X3DIR>(k);
    } else if constexpr (face == X3DIR) {
      return Xc<dir>(k, j, i);
    } else {
      Real r, z;
      if constexpr (face == X1DIR) {
        Map(base_t::Xf<X1DIR>(i), base_t::Xc<X2DIR>(j), r, z);
      } else {
        Map(base_t::Xc<X1DIR>(i), base_t::Xf<X2DIR>(j), r, z);
      }
      return (dir == X1DIR) ? r : z;
    }
  }
  template <int dir>
  KOKKOS_FORCEINLINE_FUNCTION Real Xf(const int k, const int j, const int i) const {
    return Xf<dir, dir>(k, j, i);
  }

  template <int dir, TopologicalElement el>
  KOKKOS_FORCEINLINE_FUNCTION Real Scale(const int k, const int j, const int i) const {
    static_assert(dir > 0 && dir < 4);
    if constexpr (dir == X1DIR || dir == X2DIR) return 1.0;
    // phi direction scale factor is R
    return Xc<X1DIR>(k, j, i);
  }

  //----------------------------------------
  // CellWidth: volume over the mean area of the two dir faces; R dphi for phi
  //----------------------------------------
  template <int dir>
  KOKKOS_FORCEINLINE_FUNCTION Real CellWidth(const int k, const int j,
                                             const int i) const {
    static_assert(dir > 0 && dir < 4);
    if constexpr (dir == X3DIR) {
      return Xc<X1DIR>(k, j, i) * Dx<X3DIR>();
    } else {
      if (geom_.Has(0, j, i)) return geom_((dir == X1DIR) ? G_CW1 : G_CW2, 0, j, i);
      // use the planar measure so the width is finite next to the axis
      Real ea[5], eb[5];
      FaceIntegrals_<dir>(j, i, ea);
      if constexpr (dir == X1DIR) {
        FaceIntegrals_<dir>(j, i + 1, eb);
      } else {
        FaceIntegrals_<dir>(j + 1, i, eb);
      }
      return PlanarArea_(j, i) / (0.5 * (ea[4] + eb[4]));
    }
  }
  KOKKOS_FORCEINLINE_FUNCTION Real CellWidth(const int dir, const int k, const int j,
                                             const int i) const {
    if (dir == X1DIR) return CellWidth<X1DIR>(k, j, i);
    if (dir == X2DIR) return CellWidth<X2DIR>(k, j, i);
    return CellWidth<X3DIR>(k, j, i);
  }

  //----------------------------------------
  // EdgeLength: (arc) length of cell edges
  //----------------------------------------
  template <int dir>
  KOKKOS_FORCEINLINE_FUNCTION Real EdgeLength(const int k, const int j,
                                              const int i) const {
    static_assert(dir > 0 && dir < 4);
    if constexpr (dir == X3DIR) {
      Real ra, za;
      Vertex(j, i, ra, za);
      return ra * Dx<X3DIR>();
    } else {
      if (geom_.Has(0, j, i)) return geom_((dir == X1DIR) ? G_EL1 : G_EL2, 0, j, i);
      // the dir edge from node (j, i) bounds the face normal to the other direction
      Real e[5];
      FaceIntegrals_<(dir == X1DIR) ? X2DIR : X1DIR>(j, i, e);
      return e[4];
    }
  }
  KOKKOS_FORCEINLINE_FUNCTION Real EdgeLength(const int dir, const int k, const int j,
                                              const int i) const {
    if (dir == X1DIR) return EdgeLength<X1DIR>(k, j, i);
    if (dir == X2DIR) return EdgeLength<X2DIR>(k, j, i);
    return EdgeLength<X3DIR>(k, j, i);
  }

  //----------------------------------------
  // FaceArea: magnitude of the face vector int n R dl dphi for meridional faces, planar
  // area for phi faces
  //----------------------------------------
  template <int dir>
  KOKKOS_FORCEINLINE_FUNCTION Real FaceArea(const int k, const int j, const int i) const {
    static_assert(dir > 0 && dir < 4);
    if constexpr (dir == X3DIR) {
      return PlanarArea_(j, i);
    } else {
      if (geom_.Has(0, j, i)) return geom_((dir == X1DIR) ? G_FA1 : G_FA2, 0, j, i);
      Real sr, sz;
      FaceVector_<dir>(j, i, sr, sz);
      return std::sqrt(sr * sr + sz * sz) * Dx<X3DIR>();
    }
  }

  // +1 if the logical (x1, x2) ordering is right-handed in (R, z), -1 if mirrored
  KOKKOS_FORCEINLINE_FUNCTION Real Orientation() const { return orient_; }

  //----------------------------------------
  // FaceNormal: unit normal (R, z, phi) of the dir face, pointing to increasing index.
  // This is the direction of the face vector, or of the chord on the axis where the face
  // vector vanishes
  //----------------------------------------
  template <int dir>
  KOKKOS_FORCEINLINE_FUNCTION void FaceNormal(const int k, const int j, const int i,
                                              Real &n1, Real &n2, Real &n3) const {
    static_assert(dir > 0 && dir < 4);
    if constexpr (dir == X3DIR) {
      n1 = 0.0;
      n2 = 0.0;
      n3 = 1.0;
    } else {
      n3 = 0.0;
      if (geom_.Has(0, j, i)) {
        n1 = geom_((dir == X1DIR) ? G_N1R : G_N2R, 0, j, i);
        n2 = geom_((dir == X1DIR) ? G_N1Z : G_N2Z, 0, j, i);
        return;
      }
      Real sr, sz;
      FaceVector_<dir>(j, i, sr, sz);
      const Real smag = std::sqrt(sr * sr + sz * sz);
      if (smag > 0.0) {
        n1 = sr / smag;
        n2 = sz / smag;
        return;
      }
      Real ra, za, rb, zb;
      FaceVertices_<dir>(j, i, ra, za, rb, zb);
      const Real tr = rb - ra, tz = zb - za;
      const Real s = orient_ / std::sqrt(tr * tr + tz * tz);
      if constexpr (dir == X1DIR) {
        n1 = s * tz;
        n2 = -s * tr;
      } else {
        n1 = -s * tz;
        n2 = s * tr;
      }
    }
  }

  //----------------------------------------
  // CellVolume: int R dA dphi over the cell
  //----------------------------------------
  KOKKOS_FORCEINLINE_FUNCTION Real CellVolume(const int k, const int j,
                                              const int i) const {
    if (geom_.Has(0, j, i)) return geom_(G_VOL, 0, j, i);
    Real m0, m1, m2;
    Moments_(j, i, m0, m1, m2);
    return m0 * Dx<X3DIR>();
  }

  KOKKOS_FORCEINLINE_FUNCTION
  Real Volume(CellLevel cl, TopologicalElement el, const int k, const int j,
              const int i) {
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
          "Have not yet implemented fine fields for MappedCylindrical coordinates.");
    }
    PARTHENON_FAIL("If you reach this point, someone has added a new value to the the "
                   "TopologicalElement enum.");
    return 0.0;
  }

  //----------------------------------------
  // Tabulate: tabulates the geometry over the cells, and the faces and nodes up to the
  // upper boundary of the block's ghosts, by the direct evaluation of an untabulated
  // copy (public, as it defines a device lambda)
  //----------------------------------------
  void Tabulate() {
    const MappedCylindrical eval = Untabulated_();
    geom_ =
        GeometryTable("MappedCylindrical::geom", G_NQ, 1, ncells_[1] + 1, ncells_[0] + 1);
    geom_.Fill(
        KOKKOS_LAMBDA(const ParArray4DRaw<Real> &q, const int, const int j, const int i) {
          q(G_VOL, 0, j, i) = eval.CellVolume(0, j, i);
          q(G_XC1, 0, j, i) = eval.Xc<X1DIR>(0, j, i);
          q(G_XC2, 0, j, i) = eval.Xc<X2DIR>(0, j, i);
          q(G_PA, 0, j, i) = eval.PlanarArea_(j, i);
          q(G_CW1, 0, j, i) = eval.CellWidth<X1DIR>(0, j, i);
          q(G_CW2, 0, j, i) = eval.CellWidth<X2DIR>(0, j, i);
          q(G_FA1, 0, j, i) = eval.FaceArea<X1DIR>(0, j, i);
          q(G_FA2, 0, j, i) = eval.FaceArea<X2DIR>(0, j, i);
          Real n3;
          eval.FaceNormal<X1DIR>(0, j, i, q(G_N1R, 0, j, i), q(G_N1Z, 0, j, i), n3);
          eval.FaceNormal<X2DIR>(0, j, i, q(G_N2R, 0, j, i), q(G_N2Z, 0, j, i), n3);
          q(G_EL1, 0, j, i) = eval.EdgeLength<X1DIR>(0, j, i);
          q(G_EL2, 0, j, i) = eval.EdgeLength<X2DIR>(0, j, i);
          eval.Vertex(j, i, q(G_VR, 0, j, i), q(G_VZ, 0, j, i));
        });
  }

 private:
  MappedPatch patch_;
  Real orient_ = 1.0;
  // cells (with ghosts) along each direction of the block, and the tabulated geometry
  // of its (j, i) cells, faces and nodes (independent of phi)
  std::array<int, 3> ncells_ = {0, 0, 0};
  GeometryTable geom_;
  enum : int {
    G_VOL, // CellVolume
    G_XC1, // Xc<X1DIR>
    G_XC2, // Xc<X2DIR>
    G_PA,  // PlanarArea_
    G_CW1, // CellWidth<X1DIR>
    G_CW2, // CellWidth<X2DIR>
    G_FA1, // FaceArea<X1DIR>
    G_FA2, // FaceArea<X2DIR>
    G_N1R, // FaceNormal<X1DIR>
    G_N1Z,
    G_N2R, // FaceNormal<X2DIR>
    G_N2Z,
    G_EL1, // EdgeLength<X1DIR>
    G_EL2, // EdgeLength<X2DIR>
    G_VR,  // Vertex
    G_VZ,
    G_NQ
  };

  MappedCylindrical Untabulated_() const {
    MappedCylindrical c = *this;
    c.geom_ = GeometryTable();
    return c;
  }

  // Vertices bounding the dir-face at the lower side of cell (j, i), ordered along
  // increasing logical index in the transverse direction
  template <int dir>
  KOKKOS_FORCEINLINE_FUNCTION void FaceVertices_(const int j, const int i, Real &ra,
                                                 Real &za, Real &rb, Real &zb) const {
    Vertex(j, i, ra, za);
    if constexpr (dir == X1DIR) {
      Vertex(j + 1, i, rb, zb);
    } else {
      Vertex(j, i + 1, rb, zb);
    }
  }

  // Integrals along the image of the logical segment (x1a, x2a) -> (x1b, x2b):
  // e[0] = int R dz, e[1] = int R^2/2 dz, e[2] = int R^3/3 dz, e[3] = int R^2 z/2 dz and
  // e[4] = arc length, by 5-point Gauss-Legendre quadrature
  KOKKOS_FORCEINLINE_FUNCTION void EdgeIntegrals_(const Real x1a, const Real x2a,
                                                  const Real x1b, const Real x2b,
                                                  Real e[5]) const {
    const Real gx[5] = {-0.9061798459386640, -0.5384693101056831, 0.0,
                        0.5384693101056831, 0.9061798459386640};
    const Real gw[5] = {0.2369268850561891, 0.4786286704993665, 0.5688888888888889,
                        0.4786286704993665, 0.2369268850561891};
    const Real d1 = x1b - x1a, d2 = x2b - x2a;
    for (int n = 0; n < 5; ++n)
      e[n] = 0.0;
    for (int q = 0; q < 5; ++q) {
      const Real s = 0.5 * (1.0 + gx[q]), w = 0.5 * gw[q];
      const Real p1 = x1a + s * d1, p2 = x2a + s * d2;
      Real r, z, x_xi, x_eta, y_xi, y_eta;
      Map(p1, p2, r, z);
      patch_.Jacobian(p1, p2, x_xi, x_eta, y_xi, y_eta);
      const Real dr = x_xi * d1 + x_eta * d2, dz = y_xi * d1 + y_eta * d2;
      const Real wr = w * r * dz;
      e[0] += wr;
      e[1] += 0.5 * r * wr;
      e[2] += r * r * wr / 3.0;
      e[3] += 0.5 * r * z * wr;
      e[4] += w * std::sqrt(dr * dr + dz * dz);
    }
  }

  // Edge integrals of the dir-face at the lower side of cell (j, i), along increasing
  // logical index in the transverse direction
  template <int dir>
  KOKKOS_FORCEINLINE_FUNCTION void FaceIntegrals_(const int j, const int i,
                                                  Real e[5]) const {
    const Real x1 = base_t::Xf<X1DIR>(i), x2 = base_t::Xf<X2DIR>(j);
    if constexpr (dir == X1DIR) {
      EdgeIntegrals_(x1, x2, x1, base_t::Xf<X2DIR>(j + 1), e);
    } else {
      EdgeIntegrals_(x1, x2, base_t::Xf<X1DIR>(i + 1), x2, e);
    }
  }

  // Face vector int n R dl (without dphi) of the dir-face at the lower side of cell (j, i),
  // pointing to increasing index
  template <int dir>
  KOKKOS_FORCEINLINE_FUNCTION void FaceVector_(const int j, const int i, Real &sr,
                                               Real &sz) const {
    Real e[5], ra, za, rb, zb;
    FaceIntegrals_<dir>(j, i, e);
    FaceVertices_<dir>(j, i, ra, za, rb, zb);
    // int R dR is exact from the end points
    const Real rdr = 0.5 * (rb * rb - ra * ra);
    const Real s = (dir == X1DIR) ? orient_ : -orient_;
    sr = s * e[0];
    sz = -s * rdr;
  }

  // Planar area a = int dA and moments m0 = int R dA, m1 = int R^2 dA, m2 = int R z dA
  // over the image of the logical rectangle [x1lo, x1hi] x [x2lo, x2hi].  The edges are
  // integrated along increasing logical index, as in FaceIntegrals_, and combined
  // counter-clockwise in logical space, which orient_ makes counter-clockwise in (R, z)
  KOKKOS_FORCEINLINE_FUNCTION void RectMoments_(const Real x1lo, const Real x2lo,
                                                const Real x1hi, const Real x2hi, Real &a,
                                                Real &m0, Real &m1, Real &m2) const {
    Real eb[5], er[5], et[5], el[5];
    EdgeIntegrals_(x1lo, x2lo, x1hi, x2lo, eb);
    EdgeIntegrals_(x1hi, x2lo, x1hi, x2hi, er);
    EdgeIntegrals_(x1lo, x2hi, x1hi, x2hi, et);
    EdgeIntegrals_(x1lo, x2lo, x1lo, x2hi, el);
    a = orient_ * ((eb[0] + er[0]) - (et[0] + el[0]));
    m0 = orient_ * ((eb[1] + er[1]) - (et[1] + el[1]));
    m1 = orient_ * ((eb[2] + er[2]) - (et[2] + el[2]));
    m2 = orient_ * ((eb[3] + er[3]) - (et[3] + el[3]));
  }

  KOKKOS_FORCEINLINE_FUNCTION void CellMoments_(const int j, const int i, Real &a,
                                                Real &m0, Real &m1, Real &m2) const {
    RectMoments_(base_t::Xf<X1DIR>(i), base_t::Xf<X2DIR>(j), base_t::Xf<X1DIR>(i + 1),
                 base_t::Xf<X2DIR>(j + 1), a, m0, m1, m2);
  }

  // Planar (meridional) area of cell (j, i)
  KOKKOS_FORCEINLINE_FUNCTION Real PlanarArea_(const int j, const int i) const {
    if (geom_.Has(0, j, i)) return geom_(G_PA, 0, j, i);
    Real a, m0, m1, m2;
    CellMoments_(j, i, a, m0, m1, m2);
    return a;
  }

  // Moments of cell (j, i): m0 = int R dA, m1 = int R^2 dA, m2 = int R z dA
  KOKKOS_FORCEINLINE_FUNCTION void Moments_(const int j, const int i, Real &m0, Real &m1,
                                            Real &m2) const {
    Real a;
    CellMoments_(j, i, a, m0, m1, m2);
  }
};

} // namespace parthenon

#endif // COORDINATES_MAPPED_CYLINDRICAL_HPP_
