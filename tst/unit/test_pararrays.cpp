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

#include <catch2/catch.hpp>

#include "kokkos_abstraction.hpp"
#include "parthenon_arrays.hpp"

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

TEST_CASE("ParArrayND","[ParArrayND],[Kokkos]") {

  GIVEN("A ParArrayND allocated with no label") {
    ParArrayND<Real> a(PARARRAY_TEMP,5,4,3,2);
    THEN("The label is the correct default") {
      REQUIRE( a.label().find("ParArrayND") != std::string::npos );
    }
  }

  GIVEN("A ParArrayND with some dimensions") {
    constexpr int N1 = 2;
    constexpr int N2 = 3;
    constexpr int N3 = 4;
    ParArrayND<Real> a("test",N3,N2,N1);
    WHEN("The ParArray legacy NewParArray method is used") {
      ParArrayND<Real> b;
      b.NewParArrayND(N3,N2,N1);
      THEN("The dimensions are correct") {
        REQUIRE( b.GetDim(3) == N3 );
        REQUIRE( b.GetDim(2) == N2 );
        REQUIRE( b.GetDim(1) == N1 );
        for (int d = 4; d <= 6; d++) {
          REQUIRE( b.GetDim(d) == 1 );
        }
      }
    }
    WHEN("We fill it with increasing integers") {
      // auto view = a.Get<3>();
      // auto mirror = Kokkos::create_mirror(view);
      auto mirror = a.GetHostMirror();
      int n = 0;
      int sum_host = 0;
      for (int k = 0; k < N3; k++) {
        for (int j = 0; j < N2; j++) {
          for (int i = 0; i < N1; i++) {
            mirror(k,j,i) = n;
            sum_host += n;
            n++;
          }
        }
      }
      // Kokkos::deep_copy(view,mirror);
      a.DeepCopy(mirror);
      THEN("the sum of the lower three indices is correct") {
        int sum_device = 0;
        using policy = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
        Kokkos::parallel_reduce(policy({0,0,0}, {N3,N2,N1}),
                                KOKKOS_LAMBDA(const int k, const int j,
                                              const int i,
                                              int& update) {
                                  update += a(k,j,i);
                                },
                                sum_device);
        REQUIRE( sum_host == sum_device );
      }
      THEN("the sum of the lower TWO indices is correct") {
        int sum_host = 0;
        int n = 0;
        for (int j = 0; j < N2; j++) {
          for (int i = 0; i < N1; i++) {
            sum_host += n;
            n++;
          }
        }
        int sum_device = 0;
        using policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
        Kokkos::parallel_reduce(policy({0,0}, {N2,N1}),
                                KOKKOS_LAMBDA(const int j, const int i,
                                              int& update) {
                                  update += a(j,i);
                                },
                                sum_device);
        REQUIRE( sum_host == sum_device );
        AND_THEN("We can get a raw 2d subview and it works the same way.") {
          auto v2d = a.Get<2>();
          int sum_device = 0;
          using policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
          Kokkos::parallel_reduce(policy({0,0}, {N2,N1}),
                                  KOKKOS_LAMBDA(const int j, const int i,
                                                int& update) {
                                    update += v2d(j,i);
                                  },
                                  sum_device);
          REQUIRE( sum_host == sum_device );
        }
      }
      THEN("slicing is possible") {
        // auto b = a.SliceD(std::make_pair(1,3),3);
        // auto b = a.SliceD<3>(std::make_pair(1,3));
        auto b = a.SliceD<3>(1,2); // indx,nvar
        AND_THEN("slices have correct values.") {
          int total_errors = 1; // != 0
          using policy = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;
          Kokkos::parallel_reduce(policy({0,0,0}, {2,N2,N1}),
                                  KOKKOS_LAMBDA(const int k,
                                                const int j,
                                                const int i,
                                                int& update) {
                                    update += (b(k,j,i) == a(k+1,j,i)) ? 0 : 1;
                                  },
                                  total_errors);
          REQUIRE( total_errors == 0 );
        }
      }
    }
  }
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
