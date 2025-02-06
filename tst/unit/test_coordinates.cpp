//========================================================================================
// (C) (or copyright) 2024. Triad National Security, LLC. All rights reserved.
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

#include <array>
#include <iostream>

#include "coordinates/uniform_cartesian.hpp"
#include "coordinates/uniform_cylindrical.hpp"
#include "coordinates/uniform_spherical.hpp"

#include "basic_types.hpp"
#include "defs.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
using Real = double;
using parthenon::ParameterInput;
using parthenon::RegionSize;
using parthenon::UniformCartesian;
using parthenon::UniformCylindrical;
using parthenon::UniformSpherical;
using parthenon::X1DIR;
using parthenon::X2DIR;
using parthenon::X3DIR;

#include <catch2/catch.hpp>

int nghost_save = 888;

TEST_CASE("Checking UniformCartesian") {
  std::array<Real, 3> xrat{1.0, 1.0, 1.0};
  ParameterInput pin;
  nghost_save = parthenon::Globals::nghost;
  parthenon::Globals::nghost = 2;
  GIVEN("A coordinate object") {
    std::array<Real, 3> xmin{0.1, -0.2, 0.3};
    std::array<Real, 3> xmax{0.3, 0.1, 0.7};
    std::array<int, 3> nx{10, 12, 14};
    RegionSize rs(xmin, xmax, xrat, nx);
    UniformCartesian c(rs, &pin);
    REQUIRE(c.Dx<X1DIR>() == (xmax[0] - xmin[0]) / nx[0]);
    REQUIRE(c.Dx<X2DIR>() == (xmax[1] - xmin[1]) / nx[1]);
    REQUIRE(c.Dx<X3DIR>() == (xmax[2] - xmin[2]) / nx[2]);
  }
}

TEST_CASE("Checking UniformSpherical") {
  std::array<Real, 3> xrat{1.0, 1.0, 1.0};
  ParameterInput pin;
  parthenon::Globals::nghost = 2;
  GIVEN("A coordinate object") {
    const Real rout = 3.5;
    const Real rin = 0.0;
    std::array<Real, 3> xmin{rin, 0.0, 0.0};
    std::array<Real, 3> xmax{rout, M_PI, 2 * M_PI};
    std::array<int, 3> nx{3, 5, 8};
    RegionSize rs(xmin, xmax, xrat, nx);
    UniformSpherical c(rs, &pin);
    const auto istart = c.GetStartIndex();
    Real dr = (xmax[0] - xmin[0]) / nx[0];
    const auto cxmin = c.GetXmin();
    REQUIRE(std::abs(cxmin[0] - (xmin[0] - parthenon::Globals::nghost * dr)) < 1.e-14);
    int i0 = 6 + istart[0];
    Real r0 = xmin[0] + (i0 - istart[0]) * dr;
    REQUIRE(c.Xf<X1DIR>(i0) == r0);

    Real area = 0.0;
    for (int j = istart[1]; j < istart[1] + nx[1]; j++) {
      for (int k = istart[2]; k < istart[2] + nx[2]; k++) {
        area += c.FaceArea<X1DIR>(k, j, i0);
      }
    }
    REQUIRE(std::abs(area - 4.0 * M_PI * r0 * r0) / area < 1.e-13);

    Real volume = 0.0;
    for (int k = istart[2]; k < istart[2] + nx[2]; k++) {
      for (int j = istart[1]; j < istart[1] + nx[1]; j++) {
        for (int i = istart[0]; i < istart[0] + nx[0]; i++) {
          volume += c.CellVolume(k, j, i);
        }
      }
    }
    REQUIRE(
        std::abs(volume - (4.0 / 3.0 * M_PI * (std::pow(rout, 3) - std::pow(rin, 3)))) /
            volume <
        1.e-13);
  }
  parthenon::Globals::nghost = nghost_save;
}

TEST_CASE("Checking UniformCylindrical") {
  std::array<Real, 3> xrat{1.0, 1.0, 1.0};
  ParameterInput pin;
  parthenon::Globals::nghost = 2;
  GIVEN("A coordinate object") {
    const Real rout = 3.5;
    const Real rin = 0.0;
    const Real zlo = 1.3;
    const Real zhi = 2.78;
    std::array<Real, 3> xmin{rin, zlo, 0.0};
    std::array<Real, 3> xmax{rout, zhi, 2 * M_PI};
    std::array<int, 3> nx{3, 5, 8};
    RegionSize rs(xmin, xmax, xrat, nx);
    UniformCylindrical c(rs, &pin);
    const auto istart = c.GetStartIndex();
    Real dr = (xmax[0] - xmin[0]) / nx[0];
    const auto cxmin = c.GetXmin();
    REQUIRE(std::abs(cxmin[0] - (xmin[0] - parthenon::Globals::nghost * dr)) < 1.e-14);
    int i0 = 6 + istart[0];
    Real r0 = xmin[0] + (i0 - istart[0]) * dr;
    REQUIRE(c.Xf<X1DIR>(i0) == r0);

    Real area = 0.0;
    for (int j = istart[1]; j < istart[1] + nx[1]; j++) {
      for (int k = istart[2]; k < istart[2] + nx[2]; k++) {
        area += c.FaceArea<X1DIR>(k, j, i0);
      }
    }
    REQUIRE(std::abs(area - 2.0 * M_PI * r0 * (zhi - zlo)) / area < 1.e-13);

    Real volume = 0.0;
    for (int k = istart[2]; k < istart[2] + nx[2]; k++) {
      for (int j = istart[1]; j < istart[1] + nx[1]; j++) {
        for (int i = istart[0]; i < istart[0] + nx[0]; i++) {
          volume += c.CellVolume(k, j, i);
        }
      }
    }
    REQUIRE(std::abs(volume - M_PI * (rout * rout - rin * rin) * (zhi - zlo)) / volume <
            1.e-13);
  }
  parthenon::Globals::nghost = nghost_save;
}
