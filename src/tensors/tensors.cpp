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

#include "linear_algebra/square_svd.hpp"
#include "linear_algebra/symmetric_evd.hpp"
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

KOKKOS_INLINE_FUNCTION
void CalculateRightGramMatrices(const TensorTrain &TT, const ScratchPad3D<Real> &GR,
                                const parthenon::team_mbr_t &member) {
  const int Ngram = TT.GetNumCores();
  // pull out cores object
  auto cores = TT.cores_device();

  GR(Ngram - 1, 0, 0) = 1.;
  for (int n = Ngram - 2; n >= 0; --n) {

    // loop over elements of this Gram matrix
    // TODO experiment with patterns for reductions to find what is optimal
    for (int a = 0; a < cores(n).GetRightRank(); a++) {
      for (int ap = 0; ap < cores(n).GetRightRank(); ap++) {

        // perform the contraction; we could do this with par reduce inner
        // only over the physical index (which is the fastest moving) but
        // that would come at the expense of more reductions
        // TODO put the inner loops into the par for inner
        Real accum{0.};
        par_reduce_inner(
            parthenon::InnerLoopPatternTTR(), member, 0,
            cores(n + 1).GetPhysicalIndexSize() - 1, 0, cores(n + 1).GetRightRank() - 1,
            0, cores(n + 1).GetRightRank() - 1,
            [&](const int i, const int b, const int bp, Real &tmp) {
              tmp += cores(n + 1)(a, i, b) * GR(n + 1, b, bp) * cores(n + 1)(ap, i, bp);
            },
            Kokkos::Sum<Real, parthenon::DevMemSpace>(accum)); // par_reduce_inner
        GR(n, a, ap) = accum;
      }
    }
  }
}

KOKKOS_INLINE_FUNCTION
void CalculateLeftGramMatrices(const TensorTrain &TT, const ScratchPad3D<Real> &GL,
                               const parthenon::team_mbr_t &member) {
  const int Ngram = TT.GetNumCores();
  // pull out cores object
  auto cores = TT.cores_device();

  GL(0, 0, 0) = 1.;
  for (int n = 1; n < Ngram; ++n) {

    // loop over elements of this Gram matrix
    // TODO experiment with patterns for reductions to find what is optimal
    for (int b = 0; b < cores(n).GetLeftRank(); b++) {
      for (int bp = 0; bp < cores(n).GetLeftRank(); bp++) {

        // perform the contraction
        Real accum{0.};
        par_reduce_inner(
            parthenon::InnerLoopPatternTTR(), member, 0,
            cores(n).GetPhysicalIndexSize() - 1, 0, cores(n).GetLeftRank() - 1, 0,
            cores(n).GetLeftRank() - 1,
            [&](const int i, const int a, const int ap, Real &tmp) {
              tmp += cores(n)(a, i, b) * GL(n - 1, a, ap) * cores(n)(ap, i, bp);
            },
            Kokkos::Sum<Real, parthenon::DevMemSpace>(accum)); // par_reduce_inner
        GL(n, b, bp) = accum;
      }
    }
  }
}

KOKKOS_INLINE_FUNCTION
void CalculateGramSVDs(const TensorTrain &TT, const ScratchPad3D<Real> &GL,
                       const ScratchPad3D<Real> &GR, const ScratchPad3D<Real> svdU,
                       const ScratchPad3D<Real> svdV, const ScratchPad2D<Real> svdS,
                       const parthenon::team_mbr_t &member,
                       Kokkos::ScratchMemorySpace<parthenon::DevExecSpace> ts) {
  const int Ngram = TT.GetNumCores();
  // pull out cores object
  auto cores = TT.cores_device();

  // sweep left to right again and for each core do the following:
  // * perform eigenvalue decompositions of G_L and G_R
  // * construct the matrix we want to SVD
  // * perform the SVD of that matrix
  for (int n = 0; n < Ngram; ++n) {

    const std::size_t Rn = cores(n).GetRightRank();

    /////////////////////////////////////////////////////////////////////////////////////
    // LEFT GRAM
    /////////////////////////////////////////////////////////////////////////////////////

    // allocate scratch for input matrix (which is destroyed), eigenvalues
    // and output matrix (eigenvectors)
    ScratchPad2D<Real> AL(ts, Rn, Rn);
    ScratchPad2D<Real> QL(ts, Rn, Rn);
    ScratchPad1D<Real> eigsL(ts, Rn);
    // allocate scratch required by SymmetricEVD::execute
    ScratchPad1D<Real> lscratchL(ts, SymmetricEVD::sizet_scratch_size(Rn));
    ScratchPad1D<std::size_t> liscratchL(ts, SymmetricEVD::sizet_scratch_size(Rn));

    // write left Gram matrix to A (so that it is in contiguous memory)
    par_for_inner(member, 0, Rn - 1, 0, Rn - 1,
                  [&](const int b, const int bp) { AL(b, bp) = GL(n, b, bp); });

    // perform the eigenvalue decomposition
    member.team_barrier();
    SymmetricEVD::execute(member, &AL, &QL, eigsL.data(), lscratchL.data(),
                          liscratchL.data());

    /////////////////////////////////////////////////////////////////////////////////////
    // RIGHT GRAM
    /////////////////////////////////////////////////////////////////////////////////////

    // allocate scratch for input matrix (which is destroyed), eigenvalues
    // and output matrix (eigenvectors)
    ScratchPad2D<Real> AR(ts, Rn, Rn);
    ScratchPad2D<Real> QR(ts, Rn, Rn);
    ScratchPad1D<Real> eigsR(ts, Rn);
    // allocate scratch required by SymmetricEVD::execute
    ScratchPad1D<Real> lscratchR(ts, SymmetricEVD::sizet_scratch_size(Rn));
    ScratchPad1D<std::size_t> liscratchR(ts, SymmetricEVD::sizet_scratch_size(Rn));

    // write left Gram matrix to A (so that it is in contiguous memory)
    par_for_inner(member, 0, Rn - 1, 0, Rn - 1,
                  [&](const int a, const int ap) { AR(a, ap) = GR(n, a, ap); });

    // perform the eigenvalue decomposition
    member.team_barrier();
    SymmetricEVD::execute(member, &AR, &QR, eigsR.data(), lscratchR.data(),
                          liscratchR.data());

    //////////////////////////////////////////////////////////////////////////////////
    // Now we have the left gram's eigenvalues and eigenvectors eigsL, QL
    // and the right gram's eigenvalues and eigenvectors eigsR, QR.
    // Construct the matrix that we want to obtain the truncated SVD
    // of.
    ScratchPad2D<Real> M(ts, Rn, Rn);

    // as a par reduce inner
    for (int a = 0; a < Rn; a++) {
      for (int b = 0; b < Rn; b++) {
        Real accum{0.};
        par_reduce_inner(
            parthenon::InnerLoopPatternTTR(), member, 0, Rn - 1,
            [&](const int i, Real &tmp) { tmp += QL(a, i) * QR(i, b); },
            Kokkos::Sum<Real, parthenon::DevMemSpace>(accum)); // par_reduce_inner

        M(a, b) = std::sqrt(eigsL(a)) * accum * std::sqrt(eigsR(b));
      }
    }

    // Now we have the whitened linear map we can perform the SVD
    ScratchPad1D<Real> svdS(ts, Rn);     // singular values
    ScratchPad2D<Real> svdU(ts, Rn, Rn); // left eigenvectors
    ScratchPad2D<Real> svdV(ts, Rn, Rn); // right eigenvectors
    SquareSVD::execute(&M, &svdU, &svdV, svdS.data());
  } // end left to right sweep through cores
}

// Gram-SVD TT rounding with tolerance eps. Reduces TT ranks while
// preserving the tensor up to Frobenius error eps.
void TensorTrain::GramSVDRound(const Real eps) {
  // get max right ranks, which set max gram matrix sizes for malloc
  const std::size_t RMax = GetMaximumRightRank();

  // number of Gram matrices (this many left, this many right)
  const int Ngram = GetNumCores();
  // maximum gram matrix size (this should not be needed, should be the same)
  const int Gdim = RMax * RMax;

  // calculate scratch size for maximum storage needed:
  // * one left and one right Gram matrix per tensor core, hence 2
  const int s_RG = Ngram * Gdim; // size of all right gram matrices
  const int s_LG = Ngram * Gdim; // size of all right gram matrices
  const int s_EVec = Gdim;       // size of an eigenvector matrix
  const int s_EVal = RMax;       // size of an eigenvalue diagonal matrix
  const int s_SVD = 2 * Ngram * Gdim + Ngram * s_EVal; // size of all SVDs
  // total scratch size:
  // Gram matrices:
  // * Ncores right Gram matrices
  // * Ncores left Gram matrices
  // * Ncores SVD (2 Rmax x Rmax matrices and 1 Rmax eigenvalue vector)
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
  const int scratch_size = s_RG + s_LG + 4 * s_EVec + 2 * s_EVal + s_SVD;
  const int scratch_level = 0; // ? team or thread?

  par_for_outer(
      DEFAULT_OUTER_LOOP_PATTERN, "Gram SVD rounding", DevExecSpace(), scratch_size,
      scratch_level, 0, 1, KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int dummy) {
        auto &ts = member.team_scratch(scratch_level);

        // pull out cores object
        auto cores = cores_device_;

        // assign scratch space for right Gram matrices and compute all the right
        // Gram matrices (recursive sweep from right to left)
        // TODO make this 2D GR(ts, Ngram, RMax * RMax) so is contiguous and make
        // kokkos 2d views to handle indexing
        ScratchPad3D<Real> GR(ts, Ngram, RMax, RMax);
        CalculateRightGramMatrices(*this, GR, member);

        // assign scratch space for left Gram matrices and compute all the left
        // Gram matrices (recursive sweep from left to right)
        // TODO make this 2D GR(ts, Ngram, RMax * RMax) so is contiguous and make
        // kokkos 2d views to handle indexing
        ScratchPad3D<Real> GL(ts, Ngram, RMax, RMax);
        CalculateLeftGramMatrices(*this, GL, member);

        // assign scratch space for SVDs and compute them all and store result in
        // scratch
        ScratchPad3D<Real> svdU(ts, Ngram, RMax, RMax);
        ScratchPad3D<Real> svdV(ts, Ngram, RMax, RMax);
        ScratchPad2D<Real> svdS(ts, Ngram, RMax);

        CalculateGramSVDs(*this, GL, GR, svdU, svdV, svdS, member, ts);

        ScratchPad2D<int> keep(ts, Ngram, RMax);
        SelectSingularModes(*this, svdS, keep, eps);

        // Now we have the SVD of M in rank space; truncate by discarding
        // singular vectors associated with singular values below the
        // requested error tolerance (flagged in "keep").

        // sweep left to right again and for each SVD do the following:
        // Update the left index space of core on the right (n+1)
        // Update the right index space of the core (n)
        //
        // The left index space update operates on the original core n+1.
        // The right index space update operates on the modified (left index
        // space already updated) core n
        // for (int n = 0; n < Ngram; ++n) {
        // const std::size_t Rn = cores(n).GetRightRank();
        //}

        // Actually, let's update each core one at a time, writing a temporary
        // core
        //
        // That means the SVDs we use relative to this core will be the n-1th
        // (left index space) and nth (right index space)
        for (int n = 0; n < Ngram; ++n) {
          const std::size_t Rnm1 = cores(n).GetLeftRank();
          const std::size_t Rn = cores(n).GetRightRank();

          // allocate temporary core
          ScratchPad3D<Real> tmp(ts, cores(n).GetLeftRank(),
                                 cores(n).GetPhysicalIndexSize(),
                                 cores(n).GetRightRank());

          // update left index space (unless first core) and write into temporary
          // core; uses n-1th SVD

          // create a map to singular vectors/values we are keeping
          ScratchPad1D<int> gamma_mapL(ts, Rnm1);
          int Rnm1_new = 0;
          for (int gamma = 0; gamma < Rnm1; ++gamma) {
            gamma_mapL(gamma) = Rnm1_new;
            if (keep(n, gamma)) {
              Rnm1_new++;
            }
          }

          for (int gam = 0; gam < Rnm1_new; gam++) {
            int g = gamma_mapL(gam);
            for (int alf = 0; alf < Rn; alf++) {
              for (int i = 0; i < cores(n).GetPhysicalIndexSize(); i++) {

                Real accum{0.};
                for (int bet = 0; bet < Rnm1; bet++) {
                  for (int nu = 0; nu < Rnm1; nu++) {
                    accum += svdS(n - 1, g) * svdV(n - 1, nu, g) /
                             std::sqrt(RIGHT_EIGENVAL) * RIGHT_EIGENVEC(bet, nu) *
                             cores(n)(bet, i, alf);
                  }
                }

                tmp(gam, i, alf) = accum;
              }
            }
          }

          // update right index space (unless last core) and write back into real
          // core; uses nth SVD

          // create a map to singular vectors/values we are keeping
          ScratchPad1D<int> gamma_mapR(ts, Rn);
          int Rn_new = 0;
          for (int gamma = 0; gamma < Rn; ++gamma) {
            gamma_mapR(gamma) = Rn_new;
            if (keep(n, gamma)) {
              Rn_new++;
            }
          }

          for (int alf = 0; alf < Rnm1_new; alf++) {
            for (int gam = 0; gam < Rn_new; gam++) {
              int g = gamma_mapR(gam);
              for (int i = 0; i < cores(n).GetPhysicalIndexSize(); i++) {

                Real accum{0.};
                for (int bet = 0; bet < Rn; bet++) {
                  for (int mu = 0; mu < Rn; mu++) {
                    accum += tmp(alf, i, bet) * LEFT_EIGENVEC(bet, mu) /
                             std::sqrt(LEFT_EIGENVAL(mu)) * svdU(n, mu, g)
                  }
                }

                cores(n)(alf, i, gam) = accum;
              }
            }
          }
        }
      }); // par_for_outer
}

} // namespace tensors
} // namespace parthenon
