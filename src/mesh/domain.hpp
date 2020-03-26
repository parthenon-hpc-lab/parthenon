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

#include <numeric>

namespace parthenon {

  struct IndexRange {
    int s; /// Starting Index (inclusive)
    int e; /// Ending Index (inclusive)
    int n() const { return e-s+1; }
  };

  //! \enum Defines what regions the shape encompasses
  //
  // Assuming we have a block
  //
  //  - - - - - - - - -   ^
  //  |  |  ghost  |  |   | 
  //  - - - - - - - - -   | 
  //  |  |         |  |   
  //  |  |  Base   |  |   all
  //  |  |         |  |    
  //  |  |         |  |   | 
  //  - - - - - - - - -   |
  //  |  |         |  |   | 
  //  - - - - - - - - -   v 
  //
  // The regions are separated as shown
  // 
/*  enum IndexShapeType {
    unassigned, 
    ghost,
    all,
    active
  };*/

  //! \class IndexVolume
  //  \brief Defines the dimensions of a shape of indices
  //
  //  Defines the range of each dimension of the indices by defining a starting and stopping index
  //  also contains a label for defining which region the index shape is assigned too 
  class IndexShape {
    public:
      std::vector<IndexRange> x;

      IndexShape() {};
      IndexShape(int nx1,int nx2,int nx3) : 
        x{IndexRange{0,nx1-1},IndexRange{0,nx2-1},IndexRange{0,nx3-1}} {};

      IndexShape(int is,int ie, int js, int je, int ks, int ke) :
        x{IndexRange{is,ie},IndexRange{js,je},IndexRange{ks,ke}} {};

      int GetTotal() const noexcept { 
        return std::accumulate(std::begin(x), std::end(x), 1,
            [](int x,const IndexRange & y){ return x*y.n();} );
      }
          
  };

}

#endif // MESH_DOMAIN_HPP_
