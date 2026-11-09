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

#ifndef COORDINATES_MAPPED_PATCH_3D_HPP_
#define COORDINATES_MAPPED_PATCH_3D_HPP_

#include <cmath>

#include "basic_types.hpp"

#include <Kokkos_Macros.hpp>

namespace parthenon {

// A MappedPatch3D maps the logical coordinates (xi, eta, zeta) of a tree in a 3D forest
// to physical Cartesian coordinates (x, y, z).  It is a small POD so it can live inside
// device-side coordinate objects.  The patches build a box-in-sphere mesh:
//
//   Identity: (x, y, z) = (xi, eta, zeta)
//   Box:      x_d = lo_d + xi_d (hi_d - lo_d), the central cube
//   Wedge:    blends the cube face {a c(A, B)} to the sphere {r1 s(A, B)} linearly in zeta
//   Shell:    the spherical shell r s(A, B), r linear (or geometric) in zeta from r0 to r1
//
// where A = 2 xi - 1 and B = 2 eta - 1 are in [-1, 1], c = n + A u + B v is the point of
// the unit cube face with outward normal n and tangents u, v, and s = g / |g| with
// g = n + tan(pi A / 4) u + tan(pi B / 4) v is the equiangular direction of that face.
// The faces shared by the patches of different cube faces are evaluated by the same
// formula from both sides (up to roundoff), so the mesh is conforming.
struct MappedPatch3D {
  enum Type : int { Identity = 0, Box = 1, Wedge = 2, Shell = 3 };

  int type = Identity;
  Real lo[3] = {0.0, 0.0, 0.0};
  Real hi[3] = {1.0, 1.0, 1.0};
  // outward normal and tangents (along xi and eta) of the cube face
  Real n[3] = {0.0, 0.0, 1.0};
  Real u[3] = {1.0, 0.0, 0.0};
  Real v[3] = {0.0, 1.0, 0.0};
  // cube half-width (Wedge) and radii
  Real a = 1.0, r0 = 1.0, r1 = 1.0;
  int rlog = 0;

  static MappedPatch3D MakeBox(const Real xlo[3], const Real xhi[3]) {
    MappedPatch3D p;
    p.type = Box;
    for (int d = 0; d < 3; ++d) {
      p.lo[d] = xlo[d];
      p.hi[d] = xhi[d];
    }
    return p;
  }

  static MappedPatch3D MakeWedge(const Real nn[3], const Real uu[3], const Real vv[3],
                                 Real half_width, Real rmax) {
    MappedPatch3D p;
    p.type = Wedge;
    p.SetFace_(nn, uu, vv);
    p.a = half_width;
    p.r1 = rmax;
    return p;
  }

  static MappedPatch3D MakeShell(const Real nn[3], const Real uu[3], const Real vv[3],
                                 Real rmin, Real rmax, bool log_radial) {
    MappedPatch3D p;
    p.type = Shell;
    p.SetFace_(nn, uu, vv);
    p.r0 = rmin;
    p.r1 = rmax;
    p.rlog = log_radial;
    return p;
  }

  // Sign of the Jacobian d(x, y, z)/d(xi, eta, zeta), assumed constant over the patch
  Real Orientation() const {
    Real J[3][3];
    Jacobian(0.5, 0.5, 0.5, J);
    const Real det = J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1]) -
                     J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0]) +
                     J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0]);
    return det >= 0.0 ? 1.0 : -1.0;
  }

  KOKKOS_INLINE_FUNCTION void Map(const Real xi, const Real eta, const Real zeta,
                                  Real x[3]) const {
    if (type == Identity) {
      x[0] = xi;
      x[1] = eta;
      x[2] = zeta;
    } else if (type == Box) {
      const Real l[3] = {xi, eta, zeta};
      for (int d = 0; d < 3; ++d)
        x[d] = lo[d] + l[d] * (hi[d] - lo[d]);
    } else {
      const Real A = 2.0 * xi - 1.0, B = 2.0 * eta - 1.0;
      Real s[3], gnorm, ta, tb;
      Direction_(A, B, s, gnorm, ta, tb);
      if (type == Wedge) {
        for (int d = 0; d < 3; ++d) {
          const Real c = n[d] + A * u[d] + B * v[d];
          x[d] = (1.0 - zeta) * a * c + zeta * r1 * s[d];
        }
      } else {
        const Real r = Radius_(zeta);
        for (int d = 0; d < 3; ++d)
          x[d] = r * s[d];
      }
    }
  }

  // Jacobian J[i][d] = d x_i / d xi_d
  KOKKOS_INLINE_FUNCTION void Jacobian(const Real xi, const Real eta, const Real zeta,
                                       Real J[3][3]) const {
    if (type == Identity || type == Box) {
      for (int i = 0; i < 3; ++i) {
        for (int d = 0; d < 3; ++d)
          J[i][d] = (i != d) ? 0.0 : ((type == Box) ? hi[d] - lo[d] : 1.0);
      }
      return;
    }
    const Real A = 2.0 * xi - 1.0, B = 2.0 * eta - 1.0;
    Real s[3], gnorm, ta, tb;
    Direction_(A, B, s, gnorm, ta, tb);
    // d g / d xi = dta u and d g / d eta = dtb v, with tan' = (pi / 4) (1 + tan^2) and
    // dA / dxi = 2
    const Real dta = 0.5 * M_PI * (1.0 + ta * ta), dtb = 0.5 * M_PI * (1.0 + tb * tb);
    Real su = 0.0, sv = 0.0;
    for (int d = 0; d < 3; ++d) {
      su += s[d] * u[d];
      sv += s[d] * v[d];
    }
    Real ds_dxi[3], ds_deta[3];
    for (int d = 0; d < 3; ++d) {
      ds_dxi[d] = dta * (u[d] - s[d] * su) / gnorm;
      ds_deta[d] = dtb * (v[d] - s[d] * sv) / gnorm;
    }
    if (type == Wedge) {
      for (int d = 0; d < 3; ++d) {
        const Real c = n[d] + A * u[d] + B * v[d];
        J[d][0] = (1.0 - zeta) * a * 2.0 * u[d] + zeta * r1 * ds_dxi[d];
        J[d][1] = (1.0 - zeta) * a * 2.0 * v[d] + zeta * r1 * ds_deta[d];
        J[d][2] = r1 * s[d] - a * c;
      }
    } else {
      const Real r = Radius_(zeta);
      const Real dr = rlog ? r * std::log(r1 / r0) : r1 - r0;
      for (int d = 0; d < 3; ++d) {
        J[d][0] = r * ds_dxi[d];
        J[d][1] = r * ds_deta[d];
        J[d][2] = dr * s[d];
      }
    }
  }

  // Inverse of Map: the logical l = (xi, eta, zeta) of the physical point x, also
  // outside the unit cube where the map extends smoothly. Exact for Identity, Box and
  // Shell, Newton iteration from the patch center for Wedge.
  KOKKOS_INLINE_FUNCTION void InverseMap(const Real x[3], Real l[3]) const {
    if (type == Identity) {
      for (int d = 0; d < 3; ++d)
        l[d] = x[d];
    } else if (type == Box) {
      for (int d = 0; d < 3; ++d)
        l[d] = (x[d] - lo[d]) / (hi[d] - lo[d]);
    } else if (type == Shell) {
      // s = g / |g| with g = n + tan(pi A / 4) u + tan(pi B / 4) v
      Real r2 = 0.0, sn = 0.0, su = 0.0, sv = 0.0;
      for (int d = 0; d < 3; ++d) {
        r2 += x[d] * x[d];
        sn += x[d] * n[d];
        su += x[d] * u[d];
        sv += x[d] * v[d];
      }
      const Real r = std::sqrt(r2);
      l[0] = 0.5 * (1.0 + 4.0 / M_PI * std::atan2(su, sn));
      l[1] = 0.5 * (1.0 + 4.0 / M_PI * std::atan2(sv, sn));
      l[2] = rlog ? std::log(r / r0) / std::log(r1 / r0) : (r - r0) / (r1 - r0);
    } else {
      for (int d = 0; d < 3; ++d)
        l[d] = 0.5;
      for (int it = 0; it < 50; ++it) {
        Real p[3], J[3][3];
        Map(l[0], l[1], l[2], p);
        Jacobian(l[0], l[1], l[2], J);
        const Real f[3] = {p[0] - x[0], p[1] - x[1], p[2] - x[2]};
        // dl = J^-1 f by Cramer's rule
        const Real det = J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1]) -
                         J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0]) +
                         J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0]);
        Real dl[3];
        for (int c = 0; c < 3; ++c) {
          Real M[3][3];
          for (int i = 0; i < 3; ++i) {
            for (int d = 0; d < 3; ++d)
              M[i][d] = (d == c) ? f[i] : J[i][d];
          }
          dl[c] = (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) -
                   M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) +
                   M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0])) /
                  det;
        }
        // limit the step far from the solution
        Real s = 0.0;
        for (int d = 0; d < 3; ++d)
          s = std::abs(dl[d]) > s ? std::abs(dl[d]) : s;
        const Real scale = s > 0.5 ? 0.5 / s : 1.0;
        for (int d = 0; d < 3; ++d)
          l[d] -= scale * dl[d];
        if (s < 1.0e-14) break;
      }
    }
  }

 private:
  void SetFace_(const Real nn[3], const Real uu[3], const Real vv[3]) {
    for (int d = 0; d < 3; ++d) {
      n[d] = nn[d];
      u[d] = uu[d];
      v[d] = vv[d];
    }
  }

  // tan(pi A / 4), exact at the edges of the face, where it meets the neighboring face
  KOKKOS_INLINE_FUNCTION static Real EquiTan_(const Real A) {
    if (A == 1.0) return 1.0;
    if (A == -1.0) return -1.0;
    return std::tan(0.25 * M_PI * A);
  }

  // equiangular direction s = g / |g| of the face point (A, B)
  KOKKOS_INLINE_FUNCTION void Direction_(const Real A, const Real B, Real s[3],
                                         Real &gnorm, Real &ta, Real &tb) const {
    ta = EquiTan_(A);
    tb = EquiTan_(B);
    Real g[3], g2 = 0.0;
    for (int d = 0; d < 3; ++d) {
      g[d] = n[d] + ta * u[d] + tb * v[d];
      g2 += g[d] * g[d];
    }
    gnorm = std::sqrt(g2);
    for (int d = 0; d < 3; ++d)
      s[d] = g[d] / gnorm;
  }

  KOKKOS_INLINE_FUNCTION Real Radius_(const Real zeta) const {
    return rlog ? r0 * std::pow(r1 / r0, zeta) : r0 + zeta * (r1 - r0);
  }
};

} // namespace parthenon

#endif // COORDINATES_MAPPED_PATCH_3D_HPP_
