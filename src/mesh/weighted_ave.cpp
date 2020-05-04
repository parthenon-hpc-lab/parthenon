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
//! \file weighted_ave.cpp
//  \brief

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "parthenon_arrays.hpp"

namespace parthenon {

//----------------------------------------------------------------------------------------
//! \fn  void WeightedAve::WeightedAve
//  \brief Compute weighted average of ParArrayNDs (including cell-averaged U in time
//         integrator step)

void MeshBlock::WeightedAve(ParArrayND<Real> &u_out, ParArrayND<Real> &u_in1,
                            ParArrayND<Real> &u_in2, const Real wght[3]) {
  // consider every possible simplified form of weighted sum operator:
  // U = a*U + b*U1 + c*U2

  // assuming all 3x arrays are of the same size (or at least u_out is equal or larger
  // than each input array) in each array dimension, and full range is desired:
  // nx4*(3D real MeshBlock cells)
  const int nu = u_out.GetDim(4) - 1;

  const IndexDomain interior = IndexDomain::interior;
  IndexRange ib = cellbounds.GetBoundsI(interior);
  IndexRange jb = cellbounds.GetBoundsJ(interior);
  IndexRange kb = cellbounds.GetBoundsK(interior);
  // u_in2 may be an unallocated AthenaArray if using a 2S time integrator
  if (wght[0] == 1.0) {
    if (wght[2] != 0.0) {
      for (int n = 0; n <= nu; ++n) {
        for (int k = kb.s; k <= kb.e; ++k) {
          for (int j = jb.s; j <= jb.e; ++j) {
#pragma omp simd
            for (int i = ib.s; i <= ib.e; ++i) {
              u_out(n, k, j, i) +=
                  wght[1] * u_in1(n, k, j, i) + wght[2] * u_in2(n, k, j, i);
            }
          }
        }
      }
    } else { // do not dereference u_in2
      if (wght[1] != 0.0) {
        for (int n = 0; n <= nu; ++n) {
          for (int k = kb.s; k <= kb.e; ++k) {
            for (int j = jb.s; j <= jb.e; ++j) {
#pragma omp simd
              for (int i = ib.s; i <= ib.e; ++i) {
                u_out(n, k, j, i) += wght[1] * u_in1(n, k, j, i);
              }
            }
          }
        }
      }
    }
  } else if (wght[0] == 0.0) {
    if (wght[2] != 0.0) {
      for (int n = 0; n <= nu; ++n) {
        for (int k = kb.s; k <= kb.e; ++k) {
          for (int j = jb.s; j <= jb.e; ++j) {
#pragma omp simd
            for (int i = ib.s; i <= ib.e; ++i) {
              u_out(n, k, j, i) =
                  wght[1] * u_in1(n, k, j, i) + wght[2] * u_in2(n, k, j, i);
            }
          }
        }
      }
    } else if (wght[1] == 1.0) {
      // just deep copy
      for (int n = 0; n <= nu; ++n) {
        for (int k = kb.s; k <= kb.e; ++k) {
          for (int j = jb.s; j <= jb.e; ++j) {
#pragma omp simd
            for (int i = ib.s; i <= ib.e; ++i) {
              u_out(n, k, j, i) = u_in1(n, k, j, i);
            }
          }
        }
      }
    } else {
      for (int n = 0; n <= nu; ++n) {
        for (int k = kb.s; k <= kb.e; ++k) {
          for (int j = jb.s; j <= jb.e; ++j) {
#pragma omp simd
            for (int i = ib.s; i <= ib.e; ++i) {
              u_out(n, k, j, i) = wght[1] * u_in1(n, k, j, i);
            }
          }
        }
      }
    }
  } else {
    if (wght[2] != 0.0) {
      for (int n = 0; n <= nu; ++n) {
        for (int k = kb.s; k <= kb.e; ++k) {
          for (int j = jb.s; j <= jb.e; ++j) {
#pragma omp simd
            for (int i = ib.s; i <= ib.e; ++i) {
              u_out(n, k, j, i) = wght[0] * u_out(n, k, j, i) +
                                  wght[1] * u_in1(n, k, j, i) +
                                  wght[2] * u_in2(n, k, j, i);
            }
          }
        }
      }
    } else { // do not dereference u_in2
      if (wght[1] != 0.0) {
        for (int n = 0; n <= nu; ++n) {
          for (int k = kb.s; k <= kb.e; ++k) {
            for (int j = jb.s; j <= jb.e; ++j) {
#pragma omp simd
              for (int i = ib.s; i <= ib.e; ++i) {
                u_out(n, k, j, i) =
                    wght[0] * u_out(n, k, j, i) + wght[1] * u_in1(n, k, j, i);
              }
            }
          }
        }
      } else { // do not dereference u_in1
        for (int n = 0; n <= nu; ++n) {
          for (int k = kb.s; k <= kb.e; ++k) {
            for (int j = jb.s; j <= jb.e; ++j) {
#pragma omp simd
              for (int i = ib.s; i <= ib.e; ++i) {
                u_out(n, k, j, i) *= wght[0];
              }
            }
          }
        }
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void MeshBlock::WeightedAve
//  \brief Compute weighted average of face-averaged B in time integrator step

void MeshBlock::WeightedAve(FaceField &b_out, FaceField &b_in1, FaceField &b_in2,
                            const Real wght[3]) {

  IndexRange ib = cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = cellbounds.GetBoundsK(IndexDomain::interior);

  int jl = jb.s;
  int ju = jb.e + 1;
  // Note: these loops can be combined now that they avoid curl terms
  // Only need to separately account for the final longitudinal face in each loop limit
  if (wght[0] == 1.0) {
    if (wght[2] != 0.0) {
      //---- B1
      for (int k = kb.s; k <= kb.e; ++k) {
        for (int j = jb.s; j <= jb.e; ++j) {
#pragma omp simd
          for (int i = ib.s; i <= ib.e + 1; ++i) {
            b_out.x1f(k, j, i) +=
                wght[1] * b_in1.x1f(k, j, i) + wght[2] * b_in2.x1f(k, j, i);
          }
        }
      }
      //---- B2
      for (int k = kb.s; k <= kb.e; ++k) {
        for (int j = jl; j <= ju; ++j) {
#pragma omp simd
          for (int i = ib.s; i <= ib.e; ++i) {
            b_out.x2f(k, j, i) +=
                wght[1] * b_in1.x2f(k, j, i) + wght[2] * b_in2.x2f(k, j, i);
          }
        }
      }
      //---- B3
      for (int k = kb.s; k <= kb.e + 1; ++k) {
        for (int j = jb.s; j <= jb.e; ++j) {
#pragma omp simd
          for (int i = ib.s; i <= ib.e; ++i) {
            b_out.x3f(k, j, i) +=
                wght[1] * b_in1.x3f(k, j, i) + wght[2] * b_in2.x3f(k, j, i);
          }
        }
      }
    } else { // do not dereference u_in2
      if (wght[1] != 0.0) {
        //---- B1
        for (int k = kb.s; k <= kb.e; ++k) {
          for (int j = jb.s; j <= jb.e; ++j) {
#pragma omp simd
            for (int i = ib.s; i <= ib.e + 1; ++i) {
              b_out.x1f(k, j, i) += wght[1] * b_in1.x1f(k, j, i);
            }
          }
        }
        //---- B2
        for (int k = kb.s; k <= kb.e; ++k) {
          for (int j = jl; j <= ju; ++j) {
#pragma omp simd
            for (int i = ib.s; i <= ib.e; ++i) {
              b_out.x2f(k, j, i) += wght[1] * b_in1.x2f(k, j, i);
            }
          }
        }
        //---- B3
        for (int k = kb.s; k <= kb.e + 1; ++k) {
          for (int j = jb.s; j <= jb.e; ++j) {
#pragma omp simd
            for (int i = ib.s; i <= ib.e; ++i) {
              b_out.x3f(k, j, i) += wght[1] * b_in1.x3f(k, j, i);
            }
          }
        }
      }
    }
  } else if (wght[0] == 0.0) {
    if (wght[2] != 0.0) {
      //---- B1
      for (int k = kb.s; k <= kb.e; ++k) {
        for (int j = jb.s; j <= jb.e; ++j) {
#pragma omp simd
          for (int i = ib.s; i <= ib.e + 1; ++i) {
            b_out.x1f(k, j, i) =
                wght[1] * b_in1.x1f(k, j, i) + wght[2] * b_in2.x1f(k, j, i);
          }
        }
      }
      //---- B2
      for (int k = kb.s; k <= kb.e; ++k) {
        for (int j = jl; j <= ju; ++j) {
#pragma omp simd
          for (int i = ib.s; i <= ib.e; ++i) {
            b_out.x2f(k, j, i) =
                wght[1] * b_in1.x2f(k, j, i) + wght[2] * b_in2.x2f(k, j, i);
          }
        }
      }
      //---- B3
      for (int k = kb.s; k <= kb.e + 1; ++k) {
        for (int j = jb.s; j <= jb.e; ++j) {
#pragma omp simd
          for (int i = ib.s; i <= ib.e; ++i) {
            b_out.x3f(k, j, i) =
                wght[1] * b_in1.x3f(k, j, i) + wght[2] * b_in2.x3f(k, j, i);
          }
        }
      }
    } else if (wght[1] == 1.0) {
      // jb.est deep copy
      //---- B1
      for (int k = kb.s; k <= kb.e; ++k) {
        for (int j = jl; j <= ju; ++j) {
#pragma omp simd
          for (int i = ib.s; i <= ib.e + 1; ++i) {
            b_out.x1f(k, j, i) = b_in1.x1f(k, j, i);
          }
        }
      }
      //---- B2
      for (int k = kb.s; k <= kb.e; ++k) {
        for (int j = jb.s; j <= jb.e; ++j) {
#pragma omp simd
          for (int i = ib.s; i <= ib.e; ++i) {
            b_out.x2f(k, j, i) = b_in1.x2f(k, j, i);
          }
        }
      }
      //---- B3
      for (int k = kb.s; k <= kb.e + 1; ++k) {
        for (int j = jb.s; j <= jb.e; ++j) {
#pragma omp simd
          for (int i = ib.s; i <= ib.e; ++i) {
            b_out.x3f(k, j, i) = b_in1.x3f(k, j, i);
          }
        }
      }
    } else {
      //---- B1
      for (int k = kb.s; k <= kb.e; ++k) {
        for (int j = jb.s; j <= jb.e; ++j) {
#pragma omp simd
          for (int i = ib.s; i <= ib.e + 1; ++i) {
            b_out.x1f(k, j, i) = wght[1] * b_in1.x1f(k, j, i);
          }
        }
      }
      //---- B2
      for (int k = kb.s; k <= kb.e; ++k) {
        for (int j = jl; j <= ju; ++j) {
#pragma omp simd
          for (int i = ib.s; i <= ib.e; ++i) {
            b_out.x2f(k, j, i) = wght[1] * b_in1.x2f(k, j, i);
          }
        }
      }
      //---- B3
      for (int k = kb.s; k <= kb.e + 1; ++k) {
        for (int j = jb.s; j <= jb.e; ++j) {
#pragma omp simd
          for (int i = ib.s; i <= ib.e; ++i) {
            b_out.x3f(k, j, i) = wght[1] * b_in1.x3f(k, j, i);
          }
        }
      }
    }
  } else {
    if (wght[2] != 0.0) {
      //---- B1
      for (int k = kb.s; k <= kb.e; ++k) {
        for (int j = jb.s; j <= jb.e; ++j) {
#pragma omp simd
          for (int i = ib.s; i <= ib.e + 1; ++i) {
            b_out.x1f(k, j, i) = wght[0] * b_out.x1f(k, j, i) +
                                 wght[1] * b_in1.x1f(k, j, i) +
                                 wght[2] * b_in2.x1f(k, j, i);
          }
        }
      }
      //---- B2
      for (int k = kb.s; k <= kb.e; ++k) {
        for (int j = jl; j <= ju; ++j) {
#pragma omp simd
          for (int i = ib.s; i <= ib.e; ++i) {
            b_out.x2f(k, j, i) = wght[0] * b_out.x2f(k, j, i) +
                                 wght[1] * b_in1.x2f(k, j, i) +
                                 wght[2] * b_in2.x2f(k, j, i);
          }
        }
      }
      //---- B3
      for (int k = kb.s; k <= kb.e + 1; ++k) {
        for (int j = jb.s; j <= jb.e; ++j) {
#pragma omp simd
          for (int i = ib.s; i <= ib.e; ++i) {
            b_out.x3f(k, j, i) = wght[0] * b_out.x3f(k, j, i) +
                                 wght[1] * b_in1.x3f(k, j, i) +
                                 wght[2] * b_in2.x3f(k, j, i);
          }
        }
      }
    } else { // do not dereference u_in2
      if (wght[1] != 0.0) {
        //---- B1
        for (int k = kb.s; k <= kb.e; ++k) {
          for (int j = jb.s; j <= jb.e; ++j) {
#pragma omp simd
            for (int i = ib.s; i <= ib.e + 1; ++i) {
              b_out.x1f(k, j, i) =
                  wght[0] * b_out.x1f(k, j, i) + wght[1] * b_in1.x1f(k, j, i);
            }
          }
        }
        //---- B2
        for (int k = kb.s; k <= kb.e; ++k) {
          for (int j = jl; j <= ju; ++j) {
#pragma omp simd
            for (int i = ib.s; i <= ib.e; ++i) {
              b_out.x2f(k, j, i) =
                  wght[0] * b_out.x2f(k, j, i) + wght[1] * b_in1.x2f(k, j, i);
            }
          }
        }
        //---- B3
        for (int k = kb.s; k <= kb.e + 1; ++k) {
          for (int j = jb.s; j <= jb.e; ++j) {
#pragma omp simd
            for (int i = ib.s; i <= ib.e; ++i) {
              b_out.x3f(k, j, i) =
                  wght[0] * b_out.x3f(k, j, i) + wght[1] * b_in1.x3f(k, j, i);
            }
          }
        }
      } else { // do not dereference u_in1
        //---- B1
        for (int k = kb.s; k <= kb.e; ++k) {
          for (int j = jb.s; j <= jb.e; ++j) {
#pragma omp simd
            for (int i = ib.s; i <= ib.e + 1; ++i) {
              b_out.x1f(k, j, i) *= wght[0];
            }
          }
        }
        //---- B2
        for (int k = kb.s; k <= kb.e; ++k) {
          for (int j = jl; j <= ju; ++j) {
#pragma omp simd
            for (int i = ib.s; i <= ib.e; ++i) {
              b_out.x2f(k, j, i) *= wght[0];
            }
          }
        }
        //---- B3
        for (int k = kb.s; k <= kb.e + 1; ++k) {
          for (int j = jb.s; j <= jb.e; ++j) {
#pragma omp simd
            for (int i = ib.s; i <= ib.e; ++i) {
              b_out.x3f(k, j, i) *= wght[0];
            }
          }
        }
      }
    }
  }
  return;
}

} // namespace parthenon
