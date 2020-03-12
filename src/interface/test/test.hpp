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
#ifndef TEST_PK_H
#define TEST_PK_H

namespace parthenon {
///< a class that includes bare minimum to help compile test
class mesh {
public:
  bool f2;
  bool f3;
  bool multilevel;
  mesh(bool a,bool b,bool c):f2(a),f3(b),multilevel(c)
  {}
};
class MeshBlock {
public:
  int ncells1;
  int ncells2;
  int ncells3;
  int ncc1;
  int ncc2;
  int ncc3;
  struct mesh* pmy_mesh;
  MeshBlock(int nc1, int nc2, int nc3):ncells1(nc1),ncells2(nc2),ncells3(nc3),
				       ncc1(nc1),ncc2(nc2),ncc3(nc3)
  {pmy_mesh = new mesh(true,true,true);};
    
};

class CellCenteredBoundaryVariable {
public:
  int a;
};

#include <array>
#include <iostream>
#include "Container.hpp"
#include "ContainerIterator.hpp"
extern void testMetadata();
extern void testVariable();
extern void testContainer();
  
}
#endif
