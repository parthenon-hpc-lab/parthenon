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

  //! \class IndexVolume
  //  \brief Defines the dimensions of a volume of indices, will also return the volume by 
  //  multiplying the dimensions together
  class IndexVolume {
    public:
      IndexVolume() {};
      IndexVolume(int d1,int d2,int d3) : dim1(d1), dim2(d2), dim3(d3) {};
      int dim1, dim2, dim3;
      int GetVolume() const noexcept { return dim1*dim2*dim3; } 
  };

}

#endif // MESH_DOMAIN_HPP_
