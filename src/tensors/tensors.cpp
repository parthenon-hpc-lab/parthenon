//========================================================================================
// (C) (or copyright) 2025. Triad National Security, LLC. All rights reserved.
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

#include <vector>

#include "tensors/tensors.hpp"

namespace parthenon {
namespace tensors {

TensorTrain aXPlusY(pool_map_t &pool_map, const Real a, const TensorTrain &X,
                    const TensorTrain &Y) {
  PARTHENON_REQUIRE_THROWS(X.GetNumCores() == Y.GetNumCores(),
                           "Ensure tensor"
                           " trains being added have same number of cores");

  // declare and construct the tensor cores for the resulting train
  std::vector<TensorCoreHost> cores;
  for (int i = 0; i < X.GetNumCores(); i++) {
    PARTHENON_REQUIRE_THROWS(X.GetPhysicalIndexSize(i) == Y.GetPhysicalIndexSize(i),
                             "Ensure tensor trains being added have"
                             " same physical index size in corresponding cores");

    const std::size_t rL = (i == 0) ? 1 : X.GetLeftRank(i) + Y.GetLeftRank(i);
    const std::size_t nc = X.GetPhysicalIndexSize(i);
    const std::size_t rR =
        (i == X.GetNumCores() - 1) ? 1 : X.GetRightRank(i) + Y.GetRightRank(i);

    cores.emplace_back(pool_map, rL, nc, rR);
  }

  // construct the train that will contain the result with these cores
  TensorTrain Z(std::to_string(a) + "*" + X.label() + " + " + Y.label(), cores);

  auto Xcores = X.cores_device_;
  auto Ycores = Y.cores_device_;
  auto Zcores = Z.cores_device_;
  par_for(
      PARTHENON_AUTO_LABEL, 0, X.GetNumCores() - 1, KOKKOS_LAMBDA(const int i) {
        const std::size_t nXL = Xcores(i).GetLeftRank();
        const std::size_t nXR = Xcores(i).GetRightRank();

        // zero initialize
        for (int iL = 0; iL < Zcores(i).GetLeftRank(); iL++) {
          for (int iR = 0; iR < Zcores(i).GetRightRank(); iR++) {
            for (int ic = 0; ic < Zcores(i).GetPhysicalIndexSize(); ic++) {
              Zcores(i)(iL, ic, iR) = 0.;
            }
          }
        }

        // from X
        for (int iL = 0; iL < nXL; iL++) {
          for (int iR = 0; iR < nXR; iR++) {
            for (int ic = 0; ic < Xcores(i).GetPhysicalIndexSize(); ic++) {
              Zcores(i)(iL, ic, iR) = Xcores(i)(iL, ic, iR);
              if (i == 0) Zcores(i)(iL, ic, iR) *= a;
            }
          }
        }

        // from Y - left (right) offset is zero for first (last) core
        const int oL = (i > 0) * nXL;
        const int oR = (i < X.GetNumCores() - 1) * nXR;
        for (int iL = 0; iL < Ycores(i).GetLeftRank(); iL++) {
          for (int iR = 0; iR < Ycores(i).GetRightRank(); iR++) {
            for (int ic = 0; ic < Ycores(i).GetPhysicalIndexSize(); ic++) {
              Zcores(i)(oL + iL, ic, oR + iR) = Ycores(i)(iL, ic, iR);
            }
          }
        }
      });

  return Z;
} // AXPlusY

// Gram-SVD TT rounding with tolerance eps. Reduces TT ranks while
// preserving the tensor up to Frobenius error eps.
void TensorTrain::GramSVDRound(const Real eps) {
  // get max left and right ranks, which set max gram matrix sizes for malloc
  const std::size_t max_left_rank = GetMaximumLeftRank();
  const std::size_t max_right_rank = GetMaximumRightRank();

  // number of Gram matrices (this many left, this many right)
  const int Ngram = GetNumCores();
  // maximum gram matrix size (this should not be needed, should be the same)
  const int GN = std::max(max_left_rank, max_right_rank);
  const int Gdim = GN * GN;

  // calculate scratch size for maximum storage needed:
  // * one left and one right Gram matrix per tensor core, hence 2
  const int s_RG = Ngram * Gdim; // size of all right gram matrices
  const int s_LG = Ngram * Gdim; // size of all right gram matrices
  const int s_EVec = Gdim; // size of an eigenvector matrix
  const int s_EVal = GN; // size of an eigenvalue diagonal matrix
  // total scratch size:
  // Gram matrices:
  // * Ncores right Gram matrices
  // * Ncores left Gram matrices
  // * 2 eigenvector (left and right) matrices
  // * 2 eigenvalue diagonal matrices (left and right)
  // * 1 input matrx for passing to SVD
  // SVD output:
  // * 1 left singular vector matrix from SVD
  // * 1 right singular vector matrix from SVD
  // * 1 diagonal singular value matrix from SVD
  // Totals:
  // * 2 * Ncores Gram matrices
  // * 4 eigenvector matrices
  // * 2 diagonal matrices
  const int scratch_size = s_RG + s_LG + 4 * s_EVec + 2 * s_EVal;
  const int scratch_level = 0; // ? team or thread?

  par_for_outer(DEFAULT_OUTER_LOOP_PATTERN, "Gram SVD rounding", DevExecSpace(),
      scratch_size,
      scratch_level, 0, 1, KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int dummy) {

      // assign scratch space for right Gram matrices
      ScratchPad3D<Real> GR( member.team_scratch(scratch_level), Ngram, GN, GN);
      ScratchPad3D<Real> GL( member.team_scratch(scratch_level), Ngram, GN, GN);

      // pull out cores object
      auto cores = cores_device_;

      // zero Gram matrices
      for (int k = 0; k < Ngram; ++k) {
        for (int j = 0; j < max_right_rank; ++j) {
          for (int i = 0; i < max_right_rank; ++i) {
            GR(k, j, i) = 0.;
          }
        }
      }

      // compute all the right Gram matrices (recursive sweep from right to
      // left)

      GR(Ngram-1, 0, 0) = 1.;
      for (int n = Ngram-2; n >= 0; --n) {

        // loop over elements of this Gram matrix
        for (int a = 0; a < cores(n).GetRightRank(); a++) {
          for (int ap = 0; ap < cores(n).GetRightRank(); ap++) {

          // perform the contraction; we could do this with par reduce inner
          // only over the physical index (which is the fastest moving) but
          // that would come at the expense of more reductions
          Real accum{0.};
          par_reduce_inner(parthenon::InnerLoopPatternTTR(), member, 0, cores(n+1).GetPhysicalIndexSize(), 
              [&](const int i, Real &tmp) {
            for (int b = 0; b < cores(n+1).GetRightRank(); b++) {
              for (int bp = 0; bp < cores(n+1).GetRightRank(); bp++) {
                    tmp += cores(n+1)(a, i, b) * GR(n+1, b, bp) * 
                      cores(n+1)(ap, i, bp);
                }
              }
            }, 
            Kokkos::Sum<Real, parthenon::DevMemSpace>(accum)); // par_reduce_inner
            GR(n, a, ap) = accum;
          }
        }

      }

      // compute all the left Gram matrices (recursive sweep from left to
      // right)

      GL(0, 0, 0) = 1.;
      for (int n = 1; n < Ngram; ++n) {

        // loop over elements of this Gram matrix
        for (int b = 0; b < cores(n-1).GetLeftRank(); b++) {
          for (int bp = 0; bp < cores(n-1).GetLeftRank(); bp++) {

          // perform the contraction
          Real accum{0.};
          par_reduce_inner(parthenon::InnerLoopPatternTTR(), member, 0, cores(n).GetPhysicalIndexSize(), 
              [&](const int i, Real &tmp) {
            for (int a = 0; a < cores(n-1).GetLeftRank(); a++) {
              for (int ap = 0; ap < cores(n-1).GetLeftRank(); ap++) {
                    tmp += cores(n)(a, i, b) * GL(n-1, a, ap) * 
                      cores(n)(ap, i, bp);
                }
              }
            }, 
            Kokkos::Sum<Real, parthenon::DevMemSpace>(accum)); // par_reduce_inner
            GL(n, b, bp) = accum;
          }
        }

      }

      // sweep left to right again and for each core do the following:
      // * perform eigenvalue decompositions of G_L and G_R
      // * construct the matrix we want to SVD
      // * perform the SVD of that matrix
      // * truncate the SVD result considering the requested error tolerance
      // * update the right index space of the left core and the left index
      //   space of the right core using the truncated SVD result
      for (int n = 0; n < Ngram; ++n) {

      }

    }
  ); // par_for_outer

}

} // namespace tensors
} // namespace parthenon






























