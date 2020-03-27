//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001
// for Los Alamos National Laboratory (LANL), which is operated by Triad
// National Security, LLC for the U.S. Department of Energy/National Nuclear
// Security Administration. All rights in the program are reserved by Triad
// National Security, LLC, and the U.S. Department of Energy/National Nuclear
// Security Administration. The Government is granted for itself and others
// acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license
// in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do
// so.
//========================================================================================

#include <iostream>
#include <math.h>

#include "kokkos_abstraction.hpp"
#include "parthenon_arrays.hpp"

#include <catch2/catch.hpp>

using parthenon::DevSpace;
using parthenon::ParArrayND;
using parthenon::ParArray3D;
using Real = double;

constexpr int N = 32 + 2;
constexpr int NT = 100;

KOKKOS_INLINE_FUNCTION Real coord(const int i, const int n) {
  const Real dx = 2.0/(n+1.0);
  return -1.0 + dx*i;
}

KOKKOS_INLINE_FUNCTION Real gaussian(const int iz, const int iy, const int ix,
                                     const int nz, const int ny, const int nx) {
  Real x = coord(ix,nx);
  Real y = coord(iy,ny);
  Real z = coord(iz,nz);
  Real r2 = x*x + y*y + z*z;
  return exp(-r2);
}

KOKKOS_INLINE_FUNCTION Real gaussian(const int iz, const int iy, const int ix) {
  return gaussian(iz,iy,ix,N,N,N);
}

#define stencil(l,r,k,j,i) l(k,j,i) = (1./6.)*(r(k-1,j,i)+r(k+1,j,i)+r(k,j-1,i)+r(k,j+1,i)+r(k,j,i-1)+r(k,j,i+1))

template <class T> void profile_wrapper_3d(T loop_pattern) {
  auto exec_space = DevSpace();
  Kokkos::Timer timer;

  ParArray3D<Real> raw0("raw",N,N,N);
  ParArrayND<Real> nda0("ND",N,N,N);
  auto xtra0 = nda0.Get<3>();

  ParArray3D<Real> raw1("raw",N,N,N);
  ParArrayND<Real> nda1("ND",N,N,N); 
  auto xtra1 = nda1.Get<3>();

  parthenon::par_for(loop_pattern,
          "initial data", exec_space,
          0,N-1,0,N-1,0,N-1,
          KOKKOS_LAMBDA(const int k, const int j, const int i) {
            Real f = gaussian(k,j,i);
            raw0(k,j,i) = f;
            nda0(k,j,i) = f;
          });
  Kokkos::fence();
  timer.reset();
  for (int it = 0; it < NT; it++) {
    parthenon::par_for(loop_pattern,
                       "main loop", exec_space,
                       1,N-2,1,N-2,1,N-2,
                       KOKKOS_LAMBDA(const int k, const int j, const int i) {
                         stencil(raw1,raw0,k,j,i);
                       });
    parthenon::par_for(loop_pattern,
                       "main loop", exec_space,
                       1,N-2,1,N-2,1,N-2,
                       KOKKOS_LAMBDA(const int k, const int j, const int i) {
                         stencil(raw0,raw1,k,j,i);
                       });
  }
  Kokkos::fence();
  auto time_raw = timer.seconds();
  timer.reset();
  for (int it = 0; it < NT; it++) {
    parthenon::par_for(loop_pattern,
                       "main loop", exec_space,
                       1,N-2,1,N-2,1,N-2,
                       KOKKOS_LAMBDA(const int k, const int j, const int i) {
                         stencil(nda1,nda0,k,j,i);
                       });
    parthenon::par_for(loop_pattern,
                       "main loop", exec_space,
                       1,N-2,1,N-2,1,N-2,
                       KOKKOS_LAMBDA(const int k, const int j, const int i) {
                         stencil(nda0,nda1,k,j,i);
                       });
  }
  Kokkos::fence();
  auto time_ND_arrays = timer.seconds();

  timer.reset();
  for (int it = 0; it < NT; it++) {
    parthenon::par_for(loop_pattern,
                       "main loop", exec_space,
                       1,N-2,1,N-2,1,N-2,
                       KOKKOS_LAMBDA(const int k, const int j, const int i) {
                         stencil(xtra1,xtra0,k,j,i);
                       });
    parthenon::par_for(loop_pattern,
                       "main loop", exec_space,
                       1,N-2,1,N-2,1,N-2,
                       KOKKOS_LAMBDA(const int k, const int j, const int i) {
                         stencil(xtra0,xtra1,k,j,i);
                       });
  }
  Kokkos::fence();
  auto time_extracted = timer.seconds();

  std::cout << "Times:\n"
            << "\traw views   = " << time_raw << " s\n"
            << "\tND arrays   = " << time_ND_arrays << " s\n"
            << "\textracted   = " << time_extracted << " s\n"
            << std::endl;
}

TEST_CASE("Time simple stencil operations") {
  SECTION("1d range") {
    std::cout << "1d range:" << std::endl;
    profile_wrapper_3d(parthenon::loop_pattern_flatrange_tag);
  }
  SECTION("md range") {
    std::cout << "md range:" << std::endl;
    profile_wrapper_3d(parthenon::loop_pattern_mdrange_tag);
  }
  SECTION("tpttr") {
    std::cout << "tpttr range:" << std::endl;
    profile_wrapper_3d(parthenon::loop_pattern_tpttr_tag);
  }
  SECTION("tpttrtvr") {
    std::cout << "tpttrvr range:" << std::endl;
    profile_wrapper_3d(parthenon::loop_pattern_tpttrtvr_tag);
  }
#ifndef KOKKOS_ENABLE_CUDA
  SECTION("tptvr") {
    std::cout << "tptvr range:" << std::endl;
    profile_wrapper_3d(parthenon::loop_pattern_tptvr_tag);
  }
  SECTION("simdfor") {
    std::cout << "simd range:" << std::endl;
    profile_wrapper_3d(parthenon::loop_pattern_simdfor_tag);
  }
#endif
}
