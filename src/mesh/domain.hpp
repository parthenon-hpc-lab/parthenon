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

#include <array>

#include "athena.hpp" 
#include "athena_arrays.hpp"

namespace parthenon {

  const int NDIM = 3;

  struct IndexRange {
    int start = 0; /// Starting Index (inclusive)
    int end = 0; /// Ending Index (inclusive)
    int ncells() const noexcept { return end-start+1; }
  };
  
  // Assuming we have a block
  //
  //  - - - - - - - - - -   ^
  //  |  |  ghost    |  |   | 
  //  - - - - - - - - - -   | 
  //  |  |     ^     |  |   | 
  //  |  |     |     |  |   
  //  |  | interior  |  |   entire
  //  |  |     |     |  |    
  //  |  |     v     |  |   | 
  //  - - - - - - - - - -   |
  //  |  |           |  |   | 
  //  - - - - - - - - - -   v 
  //
  enum IndexShapeType {
    entire,
    interior
  };

  //! \class IndexVolume
  //  \brief Defines the dimensions of a shape of indices
  //
  //  Defines the range of each dimension of the indices by defining a starting and stopping index
  //  also contains a label for defining which region the index shape is assigned too 
  class IndexShape {
    private:
      std::array<IndexRange,NDIM> x_;
      std::array<int,NDIM> entire_ncells_;

    public:
  
      IndexShape() {};

      IndexShape(const int & nx1, const int & nx2, const int & nx3, const int & ndim,const int & ng) 
      : IndexShape( std::vector<int> {nx1,nx2,nx3}, ndim, ng) {};

      IndexShape(const std::vector<int> & interior_dims, const int & ndim,const int & ng) { 
        assert(ndim<=NDIM && "IndexShape cannot be initialized, the number of dimensions exceeds the statically set dimensions, you will need to change the NDIM constant.");
        for( int dim=1, index=0; dim<=NDIM; ++dim, ++index){
          if (dim <= ndim) {
            x_[index].start = ng;
            x_[index].end = x_[index].start + interior_dims.at(index) - 1;
            entire_ncells_[index] = interior_dims.at(index) + 2*ng;
          } else {
            entire_ncells_[index] = 1;
          }
        }
      };

      inline int x1s(const IndexShapeType & type) const 
      { return (type==IndexShapeType::entire) ? 0 : x_[0].start; }
      
      inline int x2s(const IndexShapeType & type) const
      { return (type==IndexShapeType::entire) ? 0 : x_[1].start; }
      
      inline int x3s(const IndexShapeType & type) const
      { return (type==IndexShapeType::entire) ? 0 : x_[2].start; }
      
      inline int x1e(const IndexShapeType & type) const 
      { return (type==IndexShapeType::entire) ? entire_ncells_[0]-1 : x_[0].end; }
      
      inline int x2e(const IndexShapeType & type) const 
      { return (type==IndexShapeType::entire) ? entire_ncells_[1]-1 : x_[1].end; }
      
      inline int x3e(const IndexShapeType & type) const 
      { return (type==IndexShapeType::entire) ? entire_ncells_[2]-1 : x_[2].end; }

      inline int nx1(const IndexShapeType & type) const 
      { return (type==IndexShapeType::entire) ? entire_ncells_[0] : x_[0].ncells(); }
      
      inline int nx2(const IndexShapeType & type) const 
      { return (type==IndexShapeType::entire) ? entire_ncells_[1] : x_[1].ncells(); }
      
      inline int nx3(const IndexShapeType & type) const 
      { return (type==IndexShapeType::entire) ? entire_ncells_[2] : x_[2].ncells(); }


      void GetIndices(const IndexShapeType & type, 
          int & is, int & ie, int & js, int & je, int & ks, int & ke) const {

        if(type==interior){
          is = x_[0].start;
          js = x_[1].start;
          ks = x_[2].start;
          ie = x_[0].end;
          je = x_[1].end;
          ke = x_[2].end;
        }else{
          is = js = ks = 0;
          ie = entire_ncells_[0];
          je = entire_ncells_[1];
          ke = entire_ncells_[2];
        }
      }

      void GetNx(const IndexShapeType & type, int & nx1, int & nx2, int & nx3) const {
        if(type == interior){
          nx1 = x_[0].ncells();
          nx2 = x_[1].ncells();
          nx3 = x_[2].ncells();
        }else{
          nx1 = entire_ncells_[0];
          nx2 = entire_ncells_[1];
          nx3 = entire_ncells_[2];
        }
      }

      // Kept basic for kokkos
      int GetTotal(const IndexShapeType & type) const noexcept { 
        if(x_.size() == 0) return 0;
        int total = 1;
        if(type==entire){
          for( int i = 0; i<NDIM; ++i) total*= x_[i].ncells();
        }else{
          for( int i = 0; i<NDIM; ++i) total*= entire_ncells_[i];
        }
        return total;
      }
          
  };

}

#endif // MESH_DOMAIN_HPP_
