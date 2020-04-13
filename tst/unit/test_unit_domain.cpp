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
#include <string>
#include <catch2/catch.hpp>
#include "mesh/domain.hpp"

TEST_CASE("Checking interior and entire indices with different initializations", "[x1s,x1e,x2s,x2e,x3s,x3e]"){

  const parthenon::IndexDomain interior = parthenon::IndexDomain::interior;
  const parthenon::IndexDomain entire = parthenon::IndexDomain::entire;
  GIVEN( "A 1D Index Shape"){
    int nx1 = 6;
    int num_ghost = 1;

    parthenon::IndexShape shape(nx1, num_ghost);
    REQUIRE(shape.is(interior)==num_ghost);
    REQUIRE(shape.ie(interior)==num_ghost+nx1-1);
    REQUIRE(shape.js(interior)==0);
    REQUIRE(shape.je(interior)==0);
    REQUIRE(shape.ks(interior)==0);
    REQUIRE(shape.ke(interior)==0);

    REQUIRE(shape.is(entire)==0);
    REQUIRE(shape.ie(entire)==2*num_ghost+nx1-1);
    REQUIRE(shape.js(entire)==0);
    REQUIRE(shape.je(entire)==0);
    REQUIRE(shape.ks(entire)==0);
    REQUIRE(shape.ke(entire)==0);
  }
 
  GIVEN( "A 2D Index Shape"){
    int nx1 = 6;
    int nx2 = 1;
    int num_ghost = 1;

    parthenon::IndexShape shape(nx1,nx2, num_ghost);
    REQUIRE(shape.is(interior)==num_ghost);
    REQUIRE(shape.ie(interior)==num_ghost+nx1-1);
    REQUIRE(shape.js(interior)==num_ghost);
    REQUIRE(shape.je(interior)==num_ghost+nx2-1);
    REQUIRE(shape.ks(interior)==0);
    REQUIRE(shape.ke(interior)==0);

    REQUIRE(shape.is(entire)==0);
    REQUIRE(shape.ie(entire)==2*num_ghost+nx1-1);
    REQUIRE(shape.js(entire)==0);
    REQUIRE(shape.je(entire)==2*num_ghost+nx2-1);
    REQUIRE(shape.ks(entire)==0);
    REQUIRE(shape.ke(entire)==0);
  }

  GIVEN( "A 3D Index Shape"){
    int nx1 = 6;
    int nx2 = 1;
    int nx3 = 4;
    int num_ghost = 1;

    parthenon::IndexShape shape(nx1,nx2,nx3,num_ghost);
    REQUIRE(shape.is(interior)==num_ghost);
    REQUIRE(shape.ie(interior)==num_ghost+nx1-1);
    REQUIRE(shape.js(interior)==num_ghost);
    REQUIRE(shape.je(interior)==num_ghost+nx2-1);
    REQUIRE(shape.ks(interior)==num_ghost);
    REQUIRE(shape.ke(interior)==num_ghost+nx3-1);

    REQUIRE(shape.is(entire)==0);
    REQUIRE(shape.ie(entire)==2*num_ghost+nx1-1);
    REQUIRE(shape.js(entire)==0);
    REQUIRE(shape.je(entire)==2*num_ghost+nx2-1);
    REQUIRE(shape.ks(entire)==0);
    REQUIRE(shape.ke(entire)==2*num_ghost+nx3-1);
  }

}

TEST_CASE("Checking interior and entire sizes with different initializations", "[nx1,nx2,nx3]"){

  const parthenon::IndexDomain interior = parthenon::IndexDomain::interior;
  const parthenon::IndexDomain entire = parthenon::IndexDomain::entire;
  GIVEN( "A 1D Index Shape, check the numbers of cells"){
    int nx1 = 6;
    int num_ghost = 1;

    parthenon::IndexShape shape(nx1,num_ghost);
    REQUIRE(shape.ncellsi(interior)==nx1);
    REQUIRE(shape.ncellsj(interior)==1);
    REQUIRE(shape.ncellsk(interior)==1);

    REQUIRE(shape.ncellsi(entire)==nx1+2*num_ghost);
    REQUIRE(shape.ncellsj(entire)==1);
    REQUIRE(shape.ncellsk(entire)==1);
  }
 
  GIVEN( "A 2D Index Shape, check the numbers of cells"){
    int nx1 = 6;
    int nx2 = 1;
    int num_ghost = 1;

    parthenon::IndexShape shape(nx1,nx2,num_ghost);
    REQUIRE(shape.ncellsi(interior)==nx1);
    REQUIRE(shape.ncellsj(interior)==nx2);
    REQUIRE(shape.ncellsk(interior)==1);

    REQUIRE(shape.ncellsi(entire)==nx1+2*num_ghost);
    REQUIRE(shape.ncellsj(entire)==nx2+2*num_ghost);
    REQUIRE(shape.ncellsk(entire)==1);
  }

  GIVEN( "A 3D Index Shape, check the numbers of cells"){
    int nx1 = 6;
    int nx2 = 1;
    int nx3 = 4;
    int num_ghost = 1;

    parthenon::IndexShape shape(nx1,nx2,nx3,num_ghost);
    REQUIRE(shape.ncellsi(interior)==nx1);
    REQUIRE(shape.ncellsj(interior)==nx2);
    REQUIRE(shape.ncellsk(interior)==nx3);

    REQUIRE(shape.ncellsi(entire)==nx1+2*num_ghost);
    REQUIRE(shape.ncellsj(entire)==nx2+2*num_ghost);
    REQUIRE(shape.ncellsk(entire)==nx3+2*num_ghost);
  }

}
