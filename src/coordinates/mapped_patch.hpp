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
#ifndef COORDINATES_MAPPED_PATCH_HPP_
#define COORDINATES_MAPPED_PATCH_HPP_

#include <cmath>

#include "basic_types.hpp"

#include <Kokkos_Macros.hpp>

namespace parthenon {

// A MappedPatch maps the logical coordinates (xi, eta) of a tree in a forest to physical
// planar coordinates (x, y).  It is a small POD so it can live inside device-side
// coordinate objects.  Corner ordering is c0 = (0, 0), c1 = (1, 0), c2 = (0, 1),
// c3 = (1, 1) in (xi, eta), matching the node ordering of forest::Face.
//
//   Identity: (x, y) = (xi, eta), i.e. logical coordinates are physical coordinates
//   Coons:    transfinite interpolation of four boundary curves, each of which is a
//             straight line (uniform in arc length), a circular arc about the origin
//             (uniform in angle), or a straight line uniform in angle about the origin
//   Polar:    exact polar patch, theta linear in xi and r linear (or geometric) in eta
struct MappedPatch {
  enum Type : int { Identity = 0, Coons = 1, Polar = 2 };
  enum Edge : int { Line = 0, Arc = 1, EquiAngle = 2 };
  // Edge indices: 0: eta = 0 (c0->c1), 1: eta = 1 (c2->c3), 2: xi = 0 (c0->c2),
  // 3: xi = 1 (c1->c3)

  int type = Identity;
  Real cx[4] = {0.0, 1.0, 0.0, 1.0};
  Real cy[4] = {0.0, 0.0, 1.0, 1.0};
  int edge[4] = {Line, Line, Line, Line};
  // Polar parameters (also used to cache corner angles/radii for Coons arcs)
  Real r0 = 0.0, r1 = 1.0, th0 = 0.0, th1 = 1.0;
  int rlog = 0;
  Real cth[4] = {0.0, 0.0, 0.0, 0.0};
  Real crad[4] = {0.0, 0.0, 0.0, 0.0};

  static MappedPatch MakeCoons(const Real x[4], const Real y[4], const int edges[4]) {
    MappedPatch p;
    p.type = Coons;
    for (int n = 0; n < 4; ++n) {
      p.cx[n] = x[n];
      p.cy[n] = y[n];
      p.edge[n] = edges[n];
      p.cth[n] = std::atan2(y[n], x[n]);
      p.crad[n] = std::sqrt(x[n] * x[n] + y[n] * y[n]);
    }
    return p;
  }

  static MappedPatch MakePolar(Real rmin, Real rmax, Real thmin, Real thmax,
                               bool log_radial) {
    MappedPatch p;
    p.type = Polar;
    p.r0 = rmin;
    p.r1 = rmax;
    p.th0 = thmin;
    p.th1 = thmax;
    p.rlog = log_radial;
    const Real rr[4] = {rmin, rmin, rmax, rmax};
    const Real tt[4] = {thmin, thmax, thmin, thmax};
    for (int n = 0; n < 4; ++n) {
      p.cx[n] = rr[n] * std::cos(tt[n]);
      p.cy[n] = rr[n] * std::sin(tt[n]);
      p.cth[n] = tt[n];
      p.crad[n] = rr[n];
      p.edge[n] = (n < 2) ? Arc : Line;
    }
    return p;
  }

  // Sign of the Jacobian d(x, y)/d(xi, eta), assumed constant over the patch
  Real Orientation() const {
    if (type == Identity) return 1.0;
    const Real ax = cx[1] - cx[0], ay = cy[1] - cy[0];
    const Real bx = cx[2] - cx[0], by = cy[2] - cy[0];
    return (ax * by - ay * bx) >= 0.0 ? 1.0 : -1.0;
  }

  KOKKOS_INLINE_FUNCTION void EdgePoint(const int e, const Real s, Real &x,
                                        Real &y) const {
    const int a = (e == 1) ? 2 : ((e == 3) ? 1 : 0);
    const int b = (e == 0) ? 1 : ((e == 2) ? 2 : 3);
    if (edge[e] == Arc) {
      const Real r = 0.5 * (crad[a] + crad[b]);
      const Real th = cth[a] + s * (cth[b] - cth[a]);
      x = r * std::cos(th);
      y = r * std::sin(th);
    } else if (edge[e] == EquiAngle) {
      const Real th = cth[a] + s * (cth[b] - cth[a]);
      const Real dx = std::cos(th), dy = std::sin(th);
      const Real ex = cx[b] - cx[a], ey = cy[b] - cy[a];
      // point a + t (b - a) on the ray through the origin with direction (dx, dy)
      const Real t = -(cx[a] * dy - cy[a] * dx) / (ex * dy - ey * dx);
      x = cx[a] + t * ex;
      y = cy[a] + t * ey;
    } else {
      x = cx[a] + s * (cx[b] - cx[a]);
      y = cy[a] + s * (cy[b] - cy[a]);
    }
  }

  // Derivative of EdgePoint(e, s) with respect to s
  KOKKOS_INLINE_FUNCTION void EdgeTangent(const int e, const Real s, Real &dx,
                                          Real &dy) const {
    const int a = (e == 1) ? 2 : ((e == 3) ? 1 : 0);
    const int b = (e == 0) ? 1 : ((e == 2) ? 2 : 3);
    const Real dth = cth[b] - cth[a];
    if (edge[e] == Arc) {
      const Real r = 0.5 * (crad[a] + crad[b]);
      const Real th = cth[a] + s * dth;
      dx = -r * std::sin(th) * dth;
      dy = r * std::cos(th) * dth;
    } else if (edge[e] == EquiAngle) {
      const Real th = cth[a] + s * dth;
      const Real c = std::cos(th), sn = std::sin(th);
      const Real ex = cx[b] - cx[a], ey = cy[b] - cy[a];
      // t = num / den from EdgePoint, differentiated in th
      const Real num = cy[a] * c - cx[a] * sn, den = ex * sn - ey * c;
      const Real dnum = -cy[a] * sn - cx[a] * c, dden = ex * c + ey * sn;
      const Real dt = (dnum * den - num * dden) / (den * den) * dth;
      dx = dt * ex;
      dy = dt * ey;
    } else {
      dx = cx[b] - cx[a];
      dy = cy[b] - cy[a];
    }
  }

  // Jacobian of Map: (x_xi, x_eta, y_xi, y_eta)
  KOKKOS_INLINE_FUNCTION void Jacobian(const Real xi, const Real eta, Real &x_xi,
                                       Real &x_eta, Real &y_xi, Real &y_eta) const {
    if (type == Identity) {
      x_xi = 1.0;
      x_eta = 0.0;
      y_xi = 0.0;
      y_eta = 1.0;
    } else if (type == Polar) {
      const Real th = th0 + xi * (th1 - th0);
      const Real r = rlog ? r0 * std::pow(r1 / r0, eta) : r0 + eta * (r1 - r0);
      const Real dr = rlog ? r * std::log(r1 / r0) : r1 - r0;
      const Real c = std::cos(th), sn = std::sin(th);
      x_xi = -r * sn * (th1 - th0);
      y_xi = r * c * (th1 - th0);
      x_eta = dr * c;
      y_eta = dr * sn;
    } else {
      Real bx, by, tx, ty, lx, ly, rx, ry;
      EdgePoint(0, xi, bx, by);
      EdgePoint(1, xi, tx, ty);
      EdgePoint(2, eta, lx, ly);
      EdgePoint(3, eta, rx, ry);
      Real dbx, dby, dtx, dty, dlx, dly, drx, dry;
      EdgeTangent(0, xi, dbx, dby);
      EdgeTangent(1, xi, dtx, dty);
      EdgeTangent(2, eta, dlx, dly);
      EdgeTangent(3, eta, drx, dry);
      x_xi = (1.0 - eta) * dbx + eta * dtx - lx + rx -
             ((1.0 - eta) * (cx[1] - cx[0]) + eta * (cx[3] - cx[2]));
      y_xi = (1.0 - eta) * dby + eta * dty - ly + ry -
             ((1.0 - eta) * (cy[1] - cy[0]) + eta * (cy[3] - cy[2]));
      x_eta = -bx + tx + (1.0 - xi) * dlx + xi * drx -
              ((1.0 - xi) * (cx[2] - cx[0]) + xi * (cx[3] - cx[1]));
      y_eta = -by + ty + (1.0 - xi) * dly + xi * dry -
              ((1.0 - xi) * (cy[2] - cy[0]) + xi * (cy[3] - cy[1]));
    }
  }

  KOKKOS_INLINE_FUNCTION void Map(const Real xi, const Real eta, Real &x, Real &y) const {
    if (type == Identity) {
      x = xi;
      y = eta;
    } else if (type == Polar) {
      const Real th = th0 + xi * (th1 - th0);
      const Real r = rlog ? r0 * std::pow(r1 / r0, eta) : r0 + eta * (r1 - r0);
      x = r * std::cos(th);
      y = r * std::sin(th);
    } else {
      Real bx, by, tx, ty, lx, ly, rx, ry;
      EdgePoint(0, xi, bx, by);
      EdgePoint(1, xi, tx, ty);
      EdgePoint(2, eta, lx, ly);
      EdgePoint(3, eta, rx, ry);
      const Real w0 = (1.0 - xi) * (1.0 - eta), w1 = xi * (1.0 - eta);
      const Real w2 = (1.0 - xi) * eta, w3 = xi * eta;
      x = (1.0 - eta) * bx + eta * tx + (1.0 - xi) * lx + xi * rx -
          (w0 * cx[0] + w1 * cx[1] + w2 * cx[2] + w3 * cx[3]);
      y = (1.0 - eta) * by + eta * ty + (1.0 - xi) * ly + xi * ry -
          (w0 * cy[0] + w1 * cy[1] + w2 * cy[2] + w3 * cy[3]);
    }
  }

  // Inverse of Map: the logical (xi, eta) of the physical point (x, y), also outside the
  // unit square where the map extends smoothly. Exact for Identity and Polar, Newton
  // iteration from the patch center for Coons.
  KOKKOS_INLINE_FUNCTION void InverseMap(const Real x, const Real y, Real &xi,
                                         Real &eta) const {
    if (type == Identity) {
      xi = x;
      eta = y;
    } else if (type == Polar) {
      // angle measured from the middle of the patch, so it does not wrap inside it
      const Real thm = 0.5 * (th0 + th1);
      const Real th = thm + std::atan2(y * std::cos(thm) - x * std::sin(thm),
                                       x * std::cos(thm) + y * std::sin(thm));
      const Real r = std::sqrt(x * x + y * y);
      xi = (th - th0) / (th1 - th0);
      eta = rlog ? std::log(r / r0) / std::log(r1 / r0) : (r - r0) / (r1 - r0);
    } else {
      xi = 0.5;
      eta = 0.5;
      for (int it = 0; it < 50; ++it) {
        Real px, py, a, b, c, d;
        Map(xi, eta, px, py);
        Jacobian(xi, eta, a, b, c, d);
        const Real fx = px - x, fy = py - y, det = a * d - b * c;
        Real dxi = (d * fx - b * fy) / det, deta = (a * fy - c * fx) / det;
        // limit the step far from the solution
        const Real s = std::abs(dxi) > std::abs(deta) ? std::abs(dxi) : std::abs(deta);
        if (s > 0.5) {
          dxi *= 0.5 / s;
          deta *= 0.5 / s;
        }
        xi -= dxi;
        eta -= deta;
        if (s < 1.0e-14) break;
      }
    }
  }
};

} // namespace parthenon

#endif // COORDINATES_MAPPED_PATCH_HPP_
