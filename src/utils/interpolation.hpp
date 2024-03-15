//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2024 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// Interpolation copied/refactored from
// https://github.com/lanl/phoebus and https://github.com/lanl/spiner
//========================================================================================
// © 2022. Triad National Security, LLC. All rights reserved.  This
// program was produced under U.S. Government contract
// 89233218CNA000001 for Los Alamos National Laboratory (LANL), which
// is operated by Triad National Security, LLC for the U.S.
// Department of Energy/National Nuclear Security Administration. All
// rights in the program are reserved by Triad National Security, LLC,
// and the U.S. Department of Energy/National Nuclear Security
// Administration. The Government is granted for itself and others
// acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works,
// distribute copies to the public, perform publicly and display
// publicly, and to permit others to do so.

#ifndef UTILS_INTERPOLATION_HPP_
#define UTILS_INTERPOLATION_HPP_

// Parthenon includes
#include <coordinates/coordinates.hpp>
#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>
#include <utils/error_checking.hpp>
#include <utils/robust.hpp>

namespace parthenon {
namespace interpolation {

// using namespace parthenon::package::prelude;
// using parthenon::Coordinates_t;

// From https://github.com/lanl/spiner/blob/main/spiner/regular_grid_1d.hpp
// a poor-man's std::pair
struct weights_t {
  Real first, second;
  KOKKOS_INLINE_FUNCTION Real &operator[](const int i) {
    assert(0 <= i && i <= 1);
    return i == 0 ? first : second;
  }
};

/// Base class for providing interpolation methods on uniformly spaced data.
/// Constructor is provided with spacing, number of support points, and desired
/// shift. GetIndicesAndWeights then updates arrays of indices and weights for
/// calculating the interpolated data. These arrays are of size StencilSize().
/// Data is forced to zero outside the boundaries.
class Interpolation {
 public:
  KOKKOS_FUNCTION
  Interpolation(const int n_support, const Real dx, const Real shift)
      : n_support_(n_support), dx_(dx), shift_(shift), ishift_(std::round(shift)) {}

  KOKKOS_INLINE_FUNCTION
  virtual void GetIndicesAndWeights(const int i, int *idx, Real *wgt) const {}
  KOKKOS_INLINE_FUNCTION
  virtual int StencilSize() const { return 0; }

  static constexpr int maxStencilSize = 2;

 protected:
  const int n_support_;
  const Real dx_;
  Real shift_;
  int ishift_;
};

class PiecewiseConstant : public Interpolation {
 public:
  KOKKOS_FUNCTION
  PiecewiseConstant(const int n_support, const Real dx, const Real shift)
      : Interpolation(n_support, dx, shift) {}

  KOKKOS_INLINE_FUNCTION
  void GetIndicesAndWeights(const int i, int *idx, Real *wgt) const override {
    idx[0] = i + ishift_;
    wgt[0] = 1.;
    if (idx[0] < 0 || idx[0] >= n_support_) {
      idx[0] = 0;
      wgt[0] = 0.;
    }
  }

  KOKKOS_INLINE_FUNCTION
  int StencilSize() const override { return 1; }
};

class Linear : public Interpolation {
 public:
  KOKKOS_FUNCTION
  Linear(const int n_support, const Real dx, const Real shift)
      : Interpolation(n_support, dx, shift) {
    PARTHENON_FAIL("Not written yet!");
  }

  KOKKOS_INLINE_FUNCTION
  void GetIndicesAndWeights(const int i, int *idx, Real *wgt) const override {
    idx[0] = std::floor(i + shift_);
    idx[1] = idx[0] + 1;

    wgt[0] = wgt[1] = 1. - wgt[0];

    for (int nsup = 0; nsup < 2; nsup++) {
      if (idx[nsup] < 0 || idx[nsup] >= n_support_) {
        idx[nsup] = 0;
        wgt[nsup] = 0.;
      }
    }
  }

  KOKKOS_INLINE_FUNCTION
  int StencilSize() const override { return 2; }
};

// TODO(JMM): Is this interpolation::Do syntax reasonable? An
// alternative path would be a class called "LCInterp with all
// static functions. Then it could have an `operator()` which would
// be maybe nicer?
// TODO(JMM): Merge this w/ what Ben has done.
namespace Cent {
namespace Linear {

/*
 * Get interpolation weights for linear interpolation
 * PARAM[IN] - x - location to interpolate to
 * PARAM[IN] - nx - number of points along this direction. Used for sanity checks.
 * PARAM[IN] - coords - parthenon coords object
 * PARAM[OUT] - ix - index of points to interpolate
 * PARAM[OUT] - w - weights
 */
template <int DIR>
KOKKOS_INLINE_FUNCTION void GetWeights(const Real x, const int nx,
                                       const Coordinates_t &coords, int &ix,
                                       weights_t &w) {
  PARTHENON_DEBUG_REQUIRE(
      typeid(Coordinates_t) == typeid(UniformCartesian),
      "Interpolation routines currently only work for UniformCartesian");
  const Real min = coords.Xc<DIR>(0); // assume uniform Cartesian
  const Real dx = coords.CellWidthFA(DIR);
  ix = std::min(std::max(0, static_cast<int>(robust::ratio(x - min, dx))), nx - 2);
  const Real floor = min + ix * dx;
  w[1] = robust::ratio(x - floor, dx);
  w[0] = 1. - w[1];
}

/*
 * Trilinear interpolation on a variable or meshblock pack
 * PARAM[IN] - b - Meshblock index
 * PARAM[IN] - X1, X2, X3 - Coordinate locations
 * PARAM[IN] - p - Variable or MeshBlockPack
 * PARAM[IN] - v - variable index
 */
template <typename Pack>
KOKKOS_INLINE_FUNCTION Real Do3D(int b, const Real X1, const Real X2, const Real X3,
                                 const Pack &p, int v) {
  const auto &coords = p.GetCoords(b);
  int ix[3];
  weights_t w[3];
  GetWeights<X1DIR>(X1, p.GetDim(1), coords, ix[0], w[0]);
  GetWeights<X2DIR>(X2, p.GetDim(2), coords, ix[1], w[1]);
  GetWeights<X3DIR>(X3, p.GetDim(3), coords, ix[2], w[2]);
  return (w[2][0] * (w[1][0] * (w[0][0] * p(b, v, ix[2], ix[1], ix[0]) +
                                w[0][1] * p(b, v, ix[2], ix[1], ix[0] + 1)) +
                     w[1][1] * (w[0][0] * p(b, v, ix[2], ix[1] + 1, ix[0]) +
                                w[0][1] * p(b, v, ix[2], ix[1] + 1, ix[0] + 1))) +
          w[2][1] * (w[1][0] * (w[0][0] * p(b, v, ix[2] + 1, ix[1], ix[0]) +
                                w[0][1] * p(b, v, ix[2] + 1, ix[1], ix[0] + 1)) +
                     w[1][1] * (w[0][0] * p(b, v, ix[2] + 1, ix[1] + 1, ix[0]) +
                                w[0][1] * p(b, v, ix[2] + 1, ix[1] + 1, ix[0] + 1))));
}

/*
 * Bilinear interpolation on a variable or meshblock pack
 * PARAM[IN] - b - Meshblock index
 * PARAM[IN] - X1, X2 - Coordinate locations
 * PARAM[IN] - p - Variable or MeshBlockPack
 * PARAM[IN] - v - variable index
 */
template <typename Pack>
KOKKOS_INLINE_FUNCTION Real Do2D(int b, const Real X1, const Real X2, const Pack &p,
                                 int v) {
  const auto &coords = p.GetCoords(b);
  int ix1, ix2;
  weights_t w1, w2;
  GetWeights<X1DIR>(X1, p.GetDim(1), coords, ix1, w1);
  GetWeights<X2DIR>(X2, p.GetDim(2), coords, ix2, w2);
  return (w2[0] * (w1[0] * p(b, v, 0, ix2, ix1) + w1[1] * p(b, v, 0, ix2, ix1 + 1)) +
          w2[1] *
              (w1[0] * p(b, v, 0, ix2 + 1, ix1) + w1[1] * p(b, v, 0, ix2 + 1, ix1 + 1)));
}

/*
 * Linear interpolation on a variable or meshblock pack
 * PARAM[IN] - b - Meshblock index
 * PARAM[IN] - X1 - Coordinate location
 * PARAM[IN] - p - Variable or MeshBlockPack
 * PARAM[IN] - v - variable index
 */
template <typename Pack>
KOKKOS_INLINE_FUNCTION Real Do1D(int b, const Real X1, const Pack &p, int v) {
  const auto &coords = p.GetCoords(b);
  int ix;
  weights_t w;
  GetWeights<X1DIR>(X1, p.GetDim(1), coords, ix, w);
  return w[0] * p(b, v, 0, 0, ix) + w[1] * p(b, v, 0, 0, ix + 1);
}

/*
 * Trilinear or bilinear interpolation on a variable or meshblock pack
 * PARAM[IN] - axisymmetric
 * PARAM[IN] - b - Meshblock index
 * PARAM[IN] - X1, X2, X3 - Coordinate locations
 * PARAM[IN] - p - Variable or MeshBlockPack
 * PARAM[IN] - v - variable index
 */
// JMM: I know this won't vectorize because of the switch, but it
// probably won't anyway, since we're doing trilinear
// interpolation, which will kill memory locality.  Doing it this
// way means we can do trilinear vs bilinear which I think is a
// sufficient win at minimum code bloat.
template <typename Pack>
KOKKOS_INLINE_FUNCTION Real Do(int b, const Real X1, const Real X2, const Real X3,
                               const Pack &p, int v) {
  if (p.GetDim(3) > 1) {
    return Do3D(b, X1, X2, X3, p, v);
  } else if (p.GetDim(2) > 1) {
    return Do2D(b, X1, X2, p, v);
  } else { // 1D
    return Do1D(b, X1, p, v);
  }
}

} // namespace Linear
} // namespace Cent
} // namespace interpolation
} // namespace parthenon
#endif // UTILS_INTERPOLATION_HPP_