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

#ifndef MESH_DOMAIN_HPP_
#define MESH_DOMAIN_HPP_

namespace parthenon {

  struct IndexRange {
    int start;
    int stop;
    int n;
  };

  enum IndexShapRegion {
    unassigned, 
    ghost,
    all,
    base
  };

  //! \class IndexVolume
  //  \brief Defines the dimensions of a shape of indices
  //
  //  Defines the range of each dimension of the indices by defining a starting and stopping index
  //  also contains a label for defining which region the index shape is assigned too 
  class IndexShape {
    public:
      IndexVolume() {};
      IndexVolume(int nx1,int nx2,int nx3) : x1.n(d1), x2.n(d2), x3.n(d3) {};
      IndexRange x1;
      IndexRange x2;
      IndexRange x3;

      IndexShapeRegion region = IndexShapeLabel::unassigned;
      int GetTotal() const noexcept { return x1.n*x2.n*x3.n; } 
  };

}

#endif // MESH_DOMAIN_HPP_
