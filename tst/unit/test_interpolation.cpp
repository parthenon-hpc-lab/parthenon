// Path: tst/unit/test_interpolation.cpp
//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2022 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file test_interpolation.cpp
//  \brief Test interpolation using cubic Hermite polynomials.
//
//========================================================================================

#include "utils/interpolate.hpp"
#include <catch2/catch.hpp>
#include <cstdint>

double interpolate_line(double x) {
  // f(x) = x
  std::vector<double> xv = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
  std::vector<double> fv = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
  parthenon::MonotoneInterpolator<std::vector<double>> spline(xv, fv);
  return spline(x);
}

double interpolate_parabola(double x) {
  // f(x) = x^2
  std::vector<double> xv = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
  std::vector<double> fv = {0.0, 1.0, 4.0, 9.0, 16.0, 25.0};
  parthenon::MonotoneInterpolator<std::vector<double>> spline(xv, fv);
  return spline(x);
}

double interpolate_cubic(double x) {
  // f(x) = x^3
  std::vector<double> xv = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
  std::vector<double> fv = {0.0, 1.0, 8.0, 27.0, 64.0, 125.0};
  parthenon::MonotoneInterpolator<std::vector<double>> spline(xv, fv);
  return spline(x);
}

TEST_CASE("interpolation of lines", "[interpolate_line]") {
  REQUIRE(interpolate_line(1.0) == 1.0);
  REQUIRE(interpolate_line(2.5) == 2.5);
}

TEST_CASE("interpolation of parabolas", "[interpolate_parabola]") {
  REQUIRE(interpolate_parabola(1.0) == 1.0);
  REQUIRE(interpolate_parabola(2.5) == 2.5*2.5);
  REQUIRE(interpolate_parabola(4.0) == 16.0);
}

TEST_CASE("interpolation of cubic functions", "[interpolate_cubic]") {
  REQUIRE(interpolate_cubic(1.0) == 1.0);
  REQUIRE(interpolate_cubic(2.5) == 2.5*2.5*2.5);
  REQUIRE(interpolate_cubic(4.0) == 64.0);
}

double interpolate_broken_line(double x) {
  // f(x) = x
  std::vector<double> xv = {-2.0, -1.0, 0.0, 1.0, 2.0};
  std::vector<double> fv = {2.0, 1.0, 0.0, 1.0, 2.0};
  parthenon::MonotoneInterpolator<std::vector<double>> spline(xv, fv);
  return spline(x);
}

TEST_CASE("interpolation of broken lines", "[interpolate_line_nonmonotone]") {
  REQUIRE(interpolate_broken_line(-2.0) == 2.0);
  REQUIRE(interpolate_broken_line(2.0) == 2.0);
  REQUIRE(interpolate_broken_line(-1.0) == 1.0);
  REQUIRE(interpolate_broken_line(1.0) == 1.0);
  REQUIRE(interpolate_broken_line(0.0) == 0.0);
  REQUIRE(interpolate_broken_line(-0.5) == 0.5);
  REQUIRE(interpolate_broken_line(0.5) == 0.5);
}
