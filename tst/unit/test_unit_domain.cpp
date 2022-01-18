//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
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

#include <iostream>
#include <string>

#include "mesh/domain.hpp"

#include <catch2/catch.hpp>

TEST_CASE("Checking IndexShape indices", "[IndexShape]") {
  const auto interior = parthenon::IndexDomain::interior;
  const auto entire = parthenon::IndexDomain::entire;
  const auto inner_x1 = parthenon::IndexDomain::inner_x1;
  const auto outer_x1 = parthenon::IndexDomain::outer_x1;
  const auto inner_x2 = parthenon::IndexDomain::inner_x2;
  const auto outer_x2 = parthenon::IndexDomain::outer_x2;
  const auto inner_x3 = parthenon::IndexDomain::inner_x3;
  const auto outer_x3 = parthenon::IndexDomain::outer_x3;
  GIVEN("A 1D Index Shape") {
    int nx1 = 6;
    int num_ghost = 1;

    parthenon::IndexShape shape(nx1, num_ghost);
    REQUIRE(shape.is(interior) == 1);
    REQUIRE(shape.ie(interior) == 6);
    REQUIRE(shape.js(interior) == 0);
    REQUIRE(shape.je(interior) == 0);
    REQUIRE(shape.ks(interior) == 0);
    REQUIRE(shape.ke(interior) == 0);
    REQUIRE(shape.GetTotal(interior) == 6);

    REQUIRE(shape.is(entire) == 0);
    REQUIRE(shape.ie(entire) == 7);
    REQUIRE(shape.js(entire) == 0);
    REQUIRE(shape.je(entire) == 0);
    REQUIRE(shape.ks(entire) == 0);
    REQUIRE(shape.ke(entire) == 0);
    REQUIRE(shape.GetTotal(entire) == 8);

    REQUIRE(shape.is(inner_x1) == 0);
    REQUIRE(shape.ie(inner_x1) == 0);
    REQUIRE(shape.js(inner_x1) == 0);
    REQUIRE(shape.je(inner_x1) == 0);
    REQUIRE(shape.ks(inner_x1) == 0);
    REQUIRE(shape.ke(inner_x1) == 0);
    REQUIRE(shape.GetTotal(inner_x1) == 1);

    REQUIRE(shape.is(outer_x1) == 7);
    REQUIRE(shape.ie(outer_x1) == 7);
    REQUIRE(shape.js(outer_x1) == 0);
    REQUIRE(shape.je(outer_x1) == 0);
    REQUIRE(shape.ks(outer_x1) == 0);
    REQUIRE(shape.ke(outer_x1) == 0);
    REQUIRE(shape.GetTotal(outer_x1) == 1);

    REQUIRE(shape.is(inner_x2) == 0);
    REQUIRE(shape.ie(inner_x2) == 7);
    REQUIRE(shape.js(inner_x2) == 0);
    REQUIRE(shape.je(inner_x2) == 0);
    REQUIRE(shape.ks(inner_x2) == 0);
    REQUIRE(shape.ke(inner_x2) == 0);
    REQUIRE(shape.GetTotal(inner_x2) == 0);

    REQUIRE(shape.is(outer_x2) == 0);
    REQUIRE(shape.ie(outer_x2) == 7);
    REQUIRE(shape.js(outer_x2) == 0);
    REQUIRE(shape.je(outer_x2) == 0);
    REQUIRE(shape.ks(outer_x2) == 0);
    REQUIRE(shape.ke(outer_x2) == 0);
    REQUIRE(shape.GetTotal(outer_x2) == 0);

    REQUIRE(shape.is(inner_x3) == 0);
    REQUIRE(shape.ie(inner_x3) == 7);
    REQUIRE(shape.js(inner_x3) == 0);
    REQUIRE(shape.je(inner_x3) == 0);
    REQUIRE(shape.ks(inner_x3) == 0);
    REQUIRE(shape.ke(inner_x3) == 0);
    REQUIRE(shape.GetTotal(inner_x3) == 0);

    REQUIRE(shape.is(outer_x3) == 0);
    REQUIRE(shape.ie(outer_x3) == 7);
    REQUIRE(shape.js(outer_x3) == 0);
    REQUIRE(shape.je(outer_x3) == 0);
    REQUIRE(shape.ks(outer_x3) == 0);
    REQUIRE(shape.ke(outer_x3) == 0);
    REQUIRE(shape.GetTotal(outer_x3) == 0);
  }

  GIVEN("A 2D Index Shape") {
    int nx1 = 6;
    int nx2 = 1;
    int num_ghost = 1;

    parthenon::IndexShape shape(nx2, nx1, num_ghost);
    REQUIRE(shape.is(interior) == 1);
    REQUIRE(shape.ie(interior) == 6);
    REQUIRE(shape.js(interior) == 1);
    REQUIRE(shape.je(interior) == 1);
    REQUIRE(shape.ks(interior) == 0);
    REQUIRE(shape.ke(interior) == 0);
    REQUIRE(shape.GetTotal(interior) == 6);

    REQUIRE(shape.is(entire) == 0);
    REQUIRE(shape.ie(entire) == 7);
    REQUIRE(shape.js(entire) == 0);
    REQUIRE(shape.je(entire) == 2);
    REQUIRE(shape.ks(entire) == 0);
    REQUIRE(shape.ke(entire) == 0);
    REQUIRE(shape.GetTotal(entire) == 24);

    REQUIRE(shape.is(inner_x1) == 0);
    REQUIRE(shape.ie(inner_x1) == 0);
    REQUIRE(shape.js(inner_x1) == 0);
    REQUIRE(shape.je(inner_x1) == 2);
    REQUIRE(shape.ks(inner_x1) == 0);
    REQUIRE(shape.ke(inner_x1) == 0);
    REQUIRE(shape.GetTotal(inner_x1) == 3);

    REQUIRE(shape.is(outer_x1) == 7);
    REQUIRE(shape.ie(outer_x1) == 7);
    REQUIRE(shape.js(outer_x1) == 0);
    REQUIRE(shape.je(outer_x1) == 2);
    REQUIRE(shape.ks(outer_x1) == 0);
    REQUIRE(shape.ke(outer_x1) == 0);
    REQUIRE(shape.GetTotal(outer_x1) == 3);

    REQUIRE(shape.is(inner_x2) == 0);
    REQUIRE(shape.ie(inner_x2) == 7);
    REQUIRE(shape.js(inner_x2) == 0);
    REQUIRE(shape.je(inner_x2) == 0);
    REQUIRE(shape.ks(inner_x2) == 0);
    REQUIRE(shape.ke(inner_x2) == 0);
    REQUIRE(shape.GetTotal(inner_x2) == 8);

    REQUIRE(shape.is(outer_x2) == 0);
    REQUIRE(shape.ie(outer_x2) == 7);
    REQUIRE(shape.js(outer_x2) == 2);
    REQUIRE(shape.je(outer_x2) == 2);
    REQUIRE(shape.ks(outer_x2) == 0);
    REQUIRE(shape.ke(outer_x2) == 0);
    REQUIRE(shape.GetTotal(outer_x2) == 8);

    REQUIRE(shape.is(inner_x3) == 0);
    REQUIRE(shape.ie(inner_x3) == 7);
    REQUIRE(shape.js(inner_x3) == 0);
    REQUIRE(shape.je(inner_x3) == 2);
    REQUIRE(shape.ks(inner_x3) == 0);
    REQUIRE(shape.ke(inner_x3) == 0);
    REQUIRE(shape.GetTotal(inner_x3) == 0);

    REQUIRE(shape.is(outer_x3) == 0);
    REQUIRE(shape.ie(outer_x3) == 7);
    REQUIRE(shape.js(outer_x3) == 0);
    REQUIRE(shape.je(outer_x3) == 2);
    REQUIRE(shape.ks(outer_x3) == 0);
    REQUIRE(shape.ke(outer_x3) == 0);
    REQUIRE(shape.GetTotal(outer_x3) == 0);
  }

  GIVEN("A 3D Index Shape") {
    int nx1 = 6;
    int nx2 = 1;
    int nx3 = 4;
    int num_ghost = 1;

    parthenon::IndexShape shape(nx3, nx2, nx1, num_ghost);
    REQUIRE(shape.is(interior) == 1);
    REQUIRE(shape.ie(interior) == 6);
    REQUIRE(shape.js(interior) == 1);
    REQUIRE(shape.je(interior) == 1);
    REQUIRE(shape.ks(interior) == 1);
    REQUIRE(shape.ke(interior) == 4);
    REQUIRE(shape.GetTotal(interior) == 24);

    REQUIRE(shape.is(entire) == 0);
    REQUIRE(shape.ie(entire) == 7);
    REQUIRE(shape.js(entire) == 0);
    REQUIRE(shape.je(entire) == 2);
    REQUIRE(shape.ks(entire) == 0);
    REQUIRE(shape.ke(entire) == 5);
    REQUIRE(shape.GetTotal(entire) == 144);

    REQUIRE(shape.is(inner_x1) == 0);
    REQUIRE(shape.ie(inner_x1) == 0);
    REQUIRE(shape.js(inner_x1) == 0);
    REQUIRE(shape.je(inner_x1) == 2);
    REQUIRE(shape.ks(inner_x1) == 0);
    REQUIRE(shape.ke(inner_x1) == 5);
    REQUIRE(shape.GetTotal(inner_x1) == 18);

    REQUIRE(shape.is(outer_x1) == 7);
    REQUIRE(shape.ie(outer_x1) == 7);
    REQUIRE(shape.js(outer_x1) == 0);
    REQUIRE(shape.je(outer_x1) == 2);
    REQUIRE(shape.ks(outer_x1) == 0);
    REQUIRE(shape.ke(outer_x1) == 5);
    REQUIRE(shape.GetTotal(outer_x1) == 18);

    REQUIRE(shape.is(inner_x2) == 0);
    REQUIRE(shape.ie(inner_x2) == 7);
    REQUIRE(shape.js(inner_x2) == 0);
    REQUIRE(shape.je(inner_x2) == 0);
    REQUIRE(shape.ks(inner_x2) == 0);
    REQUIRE(shape.ke(inner_x2) == 5);
    REQUIRE(shape.GetTotal(inner_x2) == 48);

    REQUIRE(shape.is(outer_x2) == 0);
    REQUIRE(shape.ie(outer_x2) == 7);
    REQUIRE(shape.js(outer_x2) == 2);
    REQUIRE(shape.je(outer_x2) == 2);
    REQUIRE(shape.ks(outer_x2) == 0);
    REQUIRE(shape.ke(outer_x2) == 5);
    REQUIRE(shape.GetTotal(outer_x2) == 48);

    REQUIRE(shape.is(inner_x3) == 0);
    REQUIRE(shape.ie(inner_x3) == 7);
    REQUIRE(shape.js(inner_x3) == 0);
    REQUIRE(shape.je(inner_x3) == 2);
    REQUIRE(shape.ks(inner_x3) == 0);
    REQUIRE(shape.ke(inner_x3) == 0);
    REQUIRE(shape.GetTotal(inner_x3) == 24);

    REQUIRE(shape.is(outer_x3) == 0);
    REQUIRE(shape.ie(outer_x3) == 7);
    REQUIRE(shape.js(outer_x3) == 0);
    REQUIRE(shape.je(outer_x3) == 2);
    REQUIRE(shape.ks(outer_x3) == 5);
    REQUIRE(shape.ke(outer_x3) == 5);
    REQUIRE(shape.GetTotal(outer_x3) == 24);
  }

  GIVEN("A 3D Index Shape initialize with vector") {
    int nx1 = 6;
    int nx2 = 1;
    int nx3 = 4;
    std::vector<int> nxs = {nx3, nx2, nx1};
    int num_ghost = 1;

    parthenon::IndexShape shape(nxs, num_ghost);
    REQUIRE(shape.is(interior) == 1);
    REQUIRE(shape.ie(interior) == 6);
    REQUIRE(shape.js(interior) == 1);
    REQUIRE(shape.je(interior) == 1);
    REQUIRE(shape.ks(interior) == 1);
    REQUIRE(shape.ke(interior) == 4);
    REQUIRE(shape.GetTotal(interior) == 24);

    REQUIRE(shape.is(entire) == 0);
    REQUIRE(shape.ie(entire) == 7);
    REQUIRE(shape.js(entire) == 0);
    REQUIRE(shape.je(entire) == 2);
    REQUIRE(shape.ks(entire) == 0);
    REQUIRE(shape.ke(entire) == 5);
    REQUIRE(shape.GetTotal(entire) == 144);
  }

  GIVEN("A 3D Index Shape 0 dim in nx3") {
    int nx1 = 6;
    int nx2 = 1;
    int nx3 = 0;
    std::vector<int> nxs = {nx3, nx2, nx1};
    int num_ghost = 1;

    parthenon::IndexShape shape(nxs, num_ghost);
    REQUIRE(shape.is(interior) == 1);
    REQUIRE(shape.ie(interior) == 6);
    REQUIRE(shape.js(interior) == 1);
    REQUIRE(shape.je(interior) == 1);
    REQUIRE(shape.ks(interior) == 0);
    REQUIRE(shape.ke(interior) == 0);
    REQUIRE(shape.GetTotal(interior) == 6);

    REQUIRE(shape.is(entire) == 0);
    REQUIRE(shape.ie(entire) == 7);
    REQUIRE(shape.js(entire) == 0);
    REQUIRE(shape.je(entire) == 2);
    REQUIRE(shape.ks(entire) == 0);
    REQUIRE(shape.ke(entire) == 0);
    REQUIRE(shape.GetTotal(entire) == 24);
  }
}

TEST_CASE("Checking IndexShape cell counts", "[IndexShape]") {
  const auto interior = parthenon::IndexDomain::interior;
  const auto entire = parthenon::IndexDomain::entire;
  const auto inner_x1 = parthenon::IndexDomain::inner_x1;
  const auto outer_x1 = parthenon::IndexDomain::outer_x1;
  const auto inner_x2 = parthenon::IndexDomain::inner_x2;
  const auto outer_x2 = parthenon::IndexDomain::outer_x2;
  const auto inner_x3 = parthenon::IndexDomain::inner_x3;
  const auto outer_x3 = parthenon::IndexDomain::outer_x3;
  GIVEN("A 1D Index Shape, check the numbers of cells") {
    int nx1 = 6;
    int num_ghost = 1;

    parthenon::IndexShape shape(nx1, num_ghost);
    REQUIRE(shape.ncellsi(interior) == 6);
    REQUIRE(shape.ncellsj(interior) == 1);
    REQUIRE(shape.ncellsk(interior) == 1);

    REQUIRE(shape.ncellsi(entire) == 8);
    REQUIRE(shape.ncellsj(entire) == 1);
    REQUIRE(shape.ncellsk(entire) == 1);

    REQUIRE(shape.ncellsi(inner_x1) == 1);
    REQUIRE(shape.ncellsj(inner_x1) == 1);
    REQUIRE(shape.ncellsk(inner_x1) == 1);

    REQUIRE(shape.ncellsi(outer_x1) == 1);
    REQUIRE(shape.ncellsj(outer_x1) == 1);
    REQUIRE(shape.ncellsk(outer_x1) == 1);

    REQUIRE(shape.ncellsi(inner_x2) == 8);
    REQUIRE(shape.ncellsj(inner_x2) == 0);
    REQUIRE(shape.ncellsk(inner_x2) == 1);

    REQUIRE(shape.ncellsi(outer_x2) == 8);
    REQUIRE(shape.ncellsj(outer_x2) == 0);
    REQUIRE(shape.ncellsk(outer_x2) == 1);

    REQUIRE(shape.ncellsi(inner_x3) == 8);
    REQUIRE(shape.ncellsj(inner_x3) == 1);
    REQUIRE(shape.ncellsk(inner_x3) == 0);

    REQUIRE(shape.ncellsi(outer_x3) == 8);
    REQUIRE(shape.ncellsj(outer_x3) == 1);
    REQUIRE(shape.ncellsk(outer_x3) == 0);
  }

  GIVEN("A 2D Index Shape, check the numbers of cells") {
    int nx1 = 6;
    int nx2 = 1;
    int num_ghost = 1;

    parthenon::IndexShape shape(nx2, nx1, num_ghost);
    REQUIRE(shape.ncellsi(interior) == 6);
    REQUIRE(shape.ncellsj(interior) == 1);
    REQUIRE(shape.ncellsk(interior) == 1);

    REQUIRE(shape.ncellsi(entire) == 8);
    REQUIRE(shape.ncellsj(entire) == 3);
    REQUIRE(shape.ncellsk(entire) == 1);

    REQUIRE(shape.ncellsi(inner_x1) == 1);
    REQUIRE(shape.ncellsj(inner_x1) == 3);
    REQUIRE(shape.ncellsk(inner_x1) == 1);

    REQUIRE(shape.ncellsi(outer_x1) == 1);
    REQUIRE(shape.ncellsj(outer_x1) == 3);
    REQUIRE(shape.ncellsk(outer_x1) == 1);

    REQUIRE(shape.ncellsi(inner_x2) == 8);
    REQUIRE(shape.ncellsj(inner_x2) == 1);
    REQUIRE(shape.ncellsk(inner_x2) == 1);

    REQUIRE(shape.ncellsi(outer_x2) == 8);
    REQUIRE(shape.ncellsj(outer_x2) == 1);
    REQUIRE(shape.ncellsk(outer_x2) == 1);

    REQUIRE(shape.ncellsi(inner_x3) == 8);
    REQUIRE(shape.ncellsj(inner_x3) == 3);
    REQUIRE(shape.ncellsk(inner_x3) == 0);

    REQUIRE(shape.ncellsi(outer_x3) == 8);
    REQUIRE(shape.ncellsj(outer_x3) == 3);
    REQUIRE(shape.ncellsk(outer_x3) == 0);
  }

  GIVEN("A 3D Index Shape, check the numbers of cells") {
    int nx1 = 6;
    int nx2 = 1;
    int nx3 = 4;
    int num_ghost = 1;

    parthenon::IndexShape shape(nx3, nx2, nx1, num_ghost);
    REQUIRE(shape.ncellsi(interior) == 6);
    REQUIRE(shape.ncellsj(interior) == 1);
    REQUIRE(shape.ncellsk(interior) == 4);

    REQUIRE(shape.ncellsi(entire) == 8);
    REQUIRE(shape.ncellsj(entire) == 3);
    REQUIRE(shape.ncellsk(entire) == 6);

    REQUIRE(shape.ncellsi(inner_x1) == 1);
    REQUIRE(shape.ncellsj(inner_x1) == 3);
    REQUIRE(shape.ncellsk(inner_x1) == 6);

    REQUIRE(shape.ncellsi(outer_x1) == 1);
    REQUIRE(shape.ncellsj(outer_x1) == 3);
    REQUIRE(shape.ncellsk(outer_x1) == 6);

    REQUIRE(shape.ncellsi(inner_x2) == 8);
    REQUIRE(shape.ncellsj(inner_x2) == 1);
    REQUIRE(shape.ncellsk(inner_x2) == 6);

    REQUIRE(shape.ncellsi(outer_x2) == 8);
    REQUIRE(shape.ncellsj(outer_x2) == 1);
    REQUIRE(shape.ncellsk(outer_x2) == 6);

    REQUIRE(shape.ncellsi(inner_x3) == 8);
    REQUIRE(shape.ncellsj(inner_x3) == 3);
    REQUIRE(shape.ncellsk(inner_x3) == 1);

    REQUIRE(shape.ncellsi(outer_x3) == 8);
    REQUIRE(shape.ncellsj(outer_x3) == 3);
    REQUIRE(shape.ncellsk(outer_x3) == 1);
  }

  GIVEN("A 3D Index Shape, check the numbers of cells after initializing with a vector") {
    int nx1 = 6;
    int nx2 = 1;
    int nx3 = 4;
    int num_ghost = 1;
    std::vector<int> nxs = {nx3, nx2, nx1};

    parthenon::IndexShape shape(nxs, num_ghost);
    REQUIRE(shape.ncellsi(interior) == 6);
    REQUIRE(shape.ncellsj(interior) == 1);
    REQUIRE(shape.ncellsk(interior) == 4);

    REQUIRE(shape.ncellsi(entire) == 8);
    REQUIRE(shape.ncellsj(entire) == 3);
    REQUIRE(shape.ncellsk(entire) == 6);
  }

  GIVEN("A 3D Index Shape, check the numbers of cells with 0 dim") {
    int nx1 = 6;
    int nx2 = 1;
    int nx3 = 0;
    int num_ghost = 1;
    std::vector<int> nxs = {nx3, nx2, nx1};

    parthenon::IndexShape shape(nxs, num_ghost);
    REQUIRE(shape.ncellsi(interior) == 6);
    REQUIRE(shape.ncellsj(interior) == 1);
    REQUIRE(shape.ncellsk(interior) == 1);

    REQUIRE(shape.ncellsi(entire) == 8);
    REQUIRE(shape.ncellsj(entire) == 3);
    REQUIRE(shape.ncellsk(entire) == 1);
  }
}
