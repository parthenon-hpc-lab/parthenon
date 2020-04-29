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
//! \file cartesian.cpp
//  \brief implements functions for Cartesian (x-y-z) coordinates in a derived class of
//  the Coordinates abstract base class.

#include "coordinates/coordinates.hpp"

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"

namespace parthenon {

//----------------------------------------------------------------------------------------
// Cartesian coordinates constructor

Cartesian::Cartesian(MeshBlock *pmb, ParameterInput *pin, bool flag)
    : Coordinates(pmb, pin, flag) {

  // get host mirrors to initialize on host
  auto x1v_h = x1v.GetHostMirror();
  auto dx1v_h = dx1v.GetHostMirror();
  auto x1f_h = x1f.GetHostMirrorAndCopy();   // initialized in the Coordinates
  auto dx1f_h = dx1f.GetHostMirrorAndCopy(); // initialized in the Coordinates
  auto x2v_h = x2v.GetHostMirror();
  auto dx2v_h = dx2v.GetHostMirror();
  auto x2f_h = x2f.GetHostMirrorAndCopy();   // initialized in the Coordinates
  auto dx2f_h = dx2f.GetHostMirrorAndCopy(); // initialized in the Coordinates
  auto x3v_h = x3v.GetHostMirror();
  auto dx3v_h = dx3v.GetHostMirror();
  auto x3f_h = x3f.GetHostMirrorAndCopy();   // initialized in the Coordinates
  auto dx3f_h = dx3f.GetHostMirrorAndCopy(); // initialized in the Coordinates

  // initialize volume-averaged coordinates and spacing
  // x1-direction: x1v = dx/2
  for (int i = il - ng; i <= iu + ng; ++i) {
    x1v_h(i) = 0.5 * (x1f_h(i + 1) + x1f_h(i));
  }
  for (int i = il - ng; i <= iu + ng - 1; ++i) {
    if (pmb->block_size.x1rat != 1.0) {
      dx1v_h(i) = x1v_h(i + 1) - x1v_h(i);
    } else {
      // dx1v = dx1f constant for uniform mesh; may disagree with x1v_h(i+1) - x1v_h(i)
      dx1v_h(i) = dx1f_h(i);
    }
  }

  // x2-direction: x2v = dy/2
  if (pmb->block_size.nx2 == 1) {
    x2v_h(jl) = 0.5 * (x2f_h(jl + 1) + x2f_h(jl));
    dx2v_h(jl) = dx2f_h(jl);
  } else {
    for (int j = jl - ng; j <= ju + ng; ++j) {
      x2v_h(j) = 0.5 * (x2f_h(j + 1) + x2f_h(j));
    }
    for (int j = jl - ng; j <= ju + ng - 1; ++j) {
      if (pmb->block_size.x2rat != 1.0) {
        dx2v_h(j) = x2v_h(j + 1) - x2v_h(j);
      } else {
        // dx2v = dx2f constant for uniform mesh; may disagree with x2v_h(j+1) - x2v_h(j)
        dx2v_h(j) = dx2f_h(j);
      }
    }
  }

  // x3-direction: x3v = dz/2
  if (pmb->block_size.nx3 == 1) {
    x3v_h(kl) = 0.5 * (x3f_h(kl + 1) + x3f_h(kl));
    dx3v_h(kl) = dx3f_h(kl);
  } else {
    for (int k = kl - ng; k <= ku + ng; ++k) {
      x3v_h(k) = 0.5 * (x3f_h(k + 1) + x3f_h(k));
    }
    for (int k = kl - ng; k <= ku + ng - 1; ++k) {
      if (pmb->block_size.x3rat != 1.0) {
        dx3v_h(k) = x3v_h(k + 1) - x3v_h(k);
      } else {
        // dxkv = dx3f constant for uniform mesh; may disagree with x3v_h(k+1) - x3v_h(k)
        dx3v_h(k) = dx3f_h(k);
      }
    }
  }

  // helper function to init on host
  auto InitOnHost = [](ParArrayND<Real> dev_arr, int low, int up, Real init_val) {
    auto host_arr = dev_arr.GetHostMirror();
    for (int i = low; i <= up; ++i) {
      host_arr(i) = init_val;
    }
    dev_arr.DeepCopy(host_arr);
  };

  // initialize geometry coefficients
  // x1-direction
  InitOnHost(h2v, il - ng, iu + ng, 1.0);
  InitOnHost(h2f, il - ng, iu + ng, 1.0);
  InitOnHost(h31v, il - ng, iu + ng, 1.0);
  InitOnHost(h31f, il - ng, iu + ng, 1.0);
  InitOnHost(dh2vd1, il - ng, iu + ng, 0.0);
  InitOnHost(dh2fd1, il - ng, iu + ng, 0.0);
  InitOnHost(dh31vd1, il - ng, iu + ng, 0.0);
  InitOnHost(dh31fd1, il - ng, iu + ng, 0.0);

  // x2-direction
  if (pmb->block_size.nx2 == 1) {
    InitOnHost(h32v, jl, jl, 1.0);
    InitOnHost(h32f, jl, jl, 1.0);
    InitOnHost(dh32vd2, jl, jl, 0.0);
    InitOnHost(dh32fd2, jl, jl, 0.0);
  } else {
    InitOnHost(h32v, jl - ng, ju + ng, 1.0);
    InitOnHost(h32f, jl - ng, ju + ng, 1.0);
    InitOnHost(dh32vd2, jl - ng, ju + ng, 0.0);
    InitOnHost(dh32fd2, jl - ng, ju + ng, 0.0);
  }

  // copy data back to device
  x1v.DeepCopy(x1v_h);
  x2v.DeepCopy(x2v_h);
  x3v.DeepCopy(x3v_h);
  dx1v.DeepCopy(dx1v_h);
  dx2v.DeepCopy(dx2v_h);
  dx3v.DeepCopy(dx3v_h);
}

} // namespace parthenon
