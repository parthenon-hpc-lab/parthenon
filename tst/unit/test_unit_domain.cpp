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

using parthenon::IndexShape;

TEST_CASE("Checking interior and entire indices with different initializations", "[x1s,x1e,x2s,x2e,x3s,x3e]"){

  GIVEN( "A 1D Index Shape"){
    int nx1 = 6;
    int num_ghost = 1;

    IndexShape shape(nx1, num_ghost);
    REQUIRE(shape.is(parthenon::interior)==num_ghost);
    REQUIRE(shape.ie(parthenon::interior)==num_ghost+nx1-1);
    REQUIRE(shape.js(parthenon::interior)==0);
    REQUIRE(shape.je(parthenon::interior)==0);
    REQUIRE(shape.ks(parthenon::interior)==0);
    REQUIRE(shape.ke(parthenon::interior)==0);

    REQUIRE(shape.is(parthenon::entire)==0);
    REQUIRE(shape.ie(parthenon::entire)==2*num_ghost+nx1-1);
    REQUIRE(shape.js(parthenon::entire)==0);
    REQUIRE(shape.je(parthenon::entire)==0);
    REQUIRE(shape.ks(parthenon::entire)==0);
    REQUIRE(shape.ke(parthenon::entire)==0);
  }
 
  GIVEN( "A 2D Index Shape"){
    int nx1 = 6;
    int nx2 = 1;
    int num_ghost = 1;

    IndexShape shape(nx1,nx2, num_ghost);
    REQUIRE(shape.is(parthenon::interior)==num_ghost);
    REQUIRE(shape.ie(parthenon::interior)==num_ghost+nx1-1);
    REQUIRE(shape.js(parthenon::interior)==num_ghost);
    REQUIRE(shape.je(parthenon::interior)==num_ghost+nx2-1);
    REQUIRE(shape.ks(parthenon::interior)==0);
    REQUIRE(shape.ke(parthenon::interior)==0);

    REQUIRE(shape.is(parthenon::entire)==0);
    REQUIRE(shape.ie(parthenon::entire)==2*num_ghost+nx1-1);
    REQUIRE(shape.js(parthenon::entire)==0);
    REQUIRE(shape.je(parthenon::entire)==2*num_ghost+nx2-1);
    REQUIRE(shape.ks(parthenon::entire)==0);
    REQUIRE(shape.ke(parthenon::entire)==0);
  }

  GIVEN( "A 3D Index Shape"){
    int nx1 = 6;
    int nx2 = 1;
    int nx3 = 4;
    int num_ghost = 1;

    IndexShape shape(nx1,nx2,nx3,num_ghost);
    REQUIRE(shape.is(parthenon::interior)==num_ghost);
    REQUIRE(shape.ie(parthenon::interior)==num_ghost+nx1-1);
    REQUIRE(shape.js(parthenon::interior)==num_ghost);
    REQUIRE(shape.je(parthenon::interior)==num_ghost+nx2-1);
    REQUIRE(shape.ks(parthenon::interior)==num_ghost);
    REQUIRE(shape.ke(parthenon::interior)==num_ghost+nx3-1);

    REQUIRE(shape.is(parthenon::entire)==0);
    REQUIRE(shape.ie(parthenon::entire)==2*num_ghost+nx1-1);
    REQUIRE(shape.js(parthenon::entire)==0);
    REQUIRE(shape.je(parthenon::entire)==2*num_ghost+nx2-1);
    REQUIRE(shape.ks(parthenon::entire)==0);
    REQUIRE(shape.ke(parthenon::entire)==2*num_ghost+nx3-1);
  }

}

TEST_CASE("Checking interior and parthenon::entire sizes with different initializations", "[nx1,nx2,nx3]"){

  GIVEN( "A 1D Index Shape, check the numbers of cells"){
    int nx1 = 6;
    int num_ghost = 1;

    IndexShape shape(nx1,num_ghost);
    REQUIRE(shape.nx1(parthenon::interior)==nx1);
    REQUIRE(shape.nx2(parthenon::interior)==1);
    REQUIRE(shape.nx3(parthenon::interior)==1);

    REQUIRE(shape.nx1(parthenon::entire)==nx1+2*num_ghost);
    REQUIRE(shape.nx2(parthenon::entire)==1);
    REQUIRE(shape.nx3(parthenon::entire)==1);
  }
 
  GIVEN( "A 2D Index Shape, check the numbers of cells"){
    int nx1 = 6;
    int nx2 = 1;
    int num_ghost = 1;

    IndexShape shape(nx1,nx2,num_ghost);
    REQUIRE(shape.nx1(parthenon::interior)==nx1);
    REQUIRE(shape.nx2(parthenon::interior)==nx2);
    REQUIRE(shape.nx3(parthenon::interior)==1);

    REQUIRE(shape.nx1(parthenon::entire)==nx1+2*num_ghost);
    REQUIRE(shape.nx2(parthenon::entire)==nx2+2*num_ghost);
    REQUIRE(shape.nx3(parthenon::entire)==1);
  }

  GIVEN( "A 3D Index Shape, check the numbers of cells"){
    int nx1 = 6;
    int nx2 = 1;
    int nx3 = 4;
    int num_ghost = 1;

    IndexShape shape(nx1,nx2,nx3,num_ghost);
    REQUIRE(shape.nx1(parthenon::interior)==nx1);
    REQUIRE(shape.nx2(parthenon::interior)==nx2);
    REQUIRE(shape.nx3(parthenon::interior)==nx3);

    REQUIRE(shape.nx1(parthenon::entire)==nx1+2*num_ghost);
    REQUIRE(shape.nx2(parthenon::entire)==nx2+2*num_ghost);
    REQUIRE(shape.nx3(parthenon::entire)==nx3+2*num_ghost);
  }

}
