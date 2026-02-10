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
void TensorTrain::CalculateRightGramMatrices(const ScratchPad2D<Real> &GR,
                                const parthenon::team_mbr_t &member) {
  const int Ngram = this->GetNumCores();
  // pull out cores object
  auto cores = this->cores_device_;

  // Last right gram matrix is simply a 1x1 with entry 1
  GR(Ngram - 1, 0) = 1.;
  for (int n = Ngram - 2; n >= 0; --n) {

    // rank of this core
    std::size_t Rn = cores(n).GetRightRank();
    // rank of next core
    std::size_t Rnp1 = cores(n+1).GetRightRank();

    // raw pointer to beginning of this gram matrix
    // this is a bit naughty
    Real *p = &GR(n, 0);
    View2DUnmanaged GRn(p, Rn, Rn);
    Real *pp1 = &GR(n+1, 0);
    View2DUnmanaged GRnp1(p, Rnp1, Rnp1);


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
              tmp += cores(n + 1)(a, i, b) * GRnp1(b, bp) * cores(n + 1)(ap, i, bp);
            },
            Kokkos::Sum<Real, parthenon::DevMemSpace>(accum)); // par_reduce_inner
        GRn(a, ap) = accum;
      }
    }
  }
}

KOKKOS_INLINE_FUNCTION
void TensorTrain::CalculateLeftGramMatrices(const ScratchPad2D<Real> &GL,
                               const parthenon::team_mbr_t &member) {
  const int Ngram = this->GetNumCores();
  // pull out cores object
  auto cores = this->cores_device_;

  // first gram matrix is 1x1 with element 1
  GL(0, 0) = 1.;
  for (int n = 1; n < Ngram; ++n) {

    // rank of this core
    std::size_t Rn = cores(n).GetRightRank();
    // rank of previous core
    std::size_t Rnm1 = cores(n-1).GetRightRank();

    // raw pointer to beginning of this gram matrix
    // this is a bit naughty
    Real *p = &GL(n, 0);
    View2DUnmanaged GLn(p, Rn, Rn);
    Real *pm1 = &GL(n-1, 0);
    View2DUnmanaged GLnm1(p, Rnm1, Rnm1);

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
              tmp += cores(n)(a, i, b) * GLnm1(a, ap) * cores(n)(ap, i, bp);
            },
            Kokkos::Sum<Real, parthenon::DevMemSpace>(accum)); // par_reduce_inner
        GLn(b, bp) = accum;
      }
    }
  }
}

// Calculate Gram SVD for bond space n given the left and right (Rn x Rn) Gram
// matrices
// * perform eigenvalue decompositions of G_L and G_R
// * construct the matrix we want to SVD
// * perform the SVD of that matrix
KOKKOS_INLINE_FUNCTION
void TensorTrain::CalculateGramSVD(const int n,
    View2DUnmanaged GL, View2DUnmanaged GR, 
    ScratchPad2D<Real> &EVL, ScratchPad2D<Real> &EVR,
    ScratchPad1D<Real> &ELamL, ScratchPad1D<Real> &ELamR,
    ScratchPad2D<Real> &svdU,
                       ScratchPad2D<Real> &svdV, ScratchPad1D<Real> &svdS,
                       parthenon::team_mbr_t member, const int scratch_level) {

  auto &ts = member.team_scratch(scratch_level);

  // pull out cores object
  auto cores = this->cores_device_;
  const std::size_t Rn = cores(n).GetRightRank();

  /////////////////////////////////////////////////////////////////////////////////////
  // LEFT GRAM
  /////////////////////////////////////////////////////////////////////////////////////

  // allocate scratch for input matrix (which is destroyed), eigenvalues
  // and output matrix (eigenvectors)
  ScratchPad2D<Real> A(ts, Rn, Rn);
  // allocate scratch required by SymmetricEVD::execute
  ScratchPad1D<Real> lscratch(ts, SymmetricEVD::sizet_scratch_size(Rn));
  ScratchPad1D<std::size_t> liscratch(ts, SymmetricEVD::sizet_scratch_size(Rn));

  // write left Gram matrix to A (so that it is not destroyed)
  par_for_inner(member, 0, Rn - 1, 0, Rn - 1,
                [&](const int b, const int bp) { A(b, bp) = GL(b, bp); });
  member.team_barrier();

  // perform the eigenvalue decomposition
  SymmetricEVD::execute(member, &A, &EVL, ELamL.data(), lscratch.data(),
                        liscratch.data());

  /////////////////////////////////////////////////////////////////////////////////////
  // RIGHT GRAM
  /////////////////////////////////////////////////////////////////////////////////////

  // write left Gram matrix to A (so that it is not destroyed)
  par_for_inner(member, 0, Rn - 1, 0, Rn - 1,
                [&](const int a, const int ap) { A(a, ap) = GR(a, ap); });
  member.team_barrier();

  // perform the eigenvalue decomposition
  SymmetricEVD::execute(member, &A, &EVR, ELamR.data(), lscratch.data(),
                        liscratch.data());

  //////////////////////////////////////////////////////////////////////////////////
  // Now we have the left gram's eigenvalues and eigenvectors ELamL, EVL
  // and the right gram's eigenvalues and eigenvectors ELamR, EVR.
  // Construct the matrix that we want to obtain the truncated SVD
  // of.
  ScratchPad2D<Real> M(ts, Rn, Rn);

  // as a par reduce inner
  for (int a = 0; a < Rn; a++) {
    for (int b = 0; b < Rn; b++) {
      Real accum{0.};
      par_reduce_inner(
          parthenon::InnerLoopPatternTTR(), member, 0, Rn - 1,
          [&](const int i, Real &tmp) { tmp += EVL(a, i) * EVR(i, b); },
          Kokkos::Sum<Real, parthenon::DevMemSpace>(accum)); // par_reduce_inner

      M(a, b) = std::sqrt(ELamL(a)) * accum * std::sqrt(ELamR(b));
    }
  }

  SquareSVD::execute(&M, &svdU, &svdV, svdS.data());
}

// Gram-SVD TT rounding with tolerance eps. Reduces TT ranks while
// preserving the tensor up to Frobenius error eps.
void TensorTrain::GramSVDRound(const Real eps) {
  // get max right ranks, which set max gram matrix sizes for malloc
  const std::size_t RMax = GetMaximumRightRank();
  const std::size_t PIMax = GetMaximumPhysicaIndexSize();

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
  const int s_core = RMax * RMax * PIMax; // temporary core for updates
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
  // Update:
  // * 1 temporary core
  // Totals:
  // * 2 * Ncores Gram matrices
  // * 4 eigenvector matrices
  // * 2 diagonal matrices
  const int scratch_size = s_RG + s_LG + 4 * s_EVec + 2 * s_EVal + s_SVD;
  const int scratch_level = 0; // ? team or thread?

  par_for_outer(
      DEFAULT_OUTER_LOOP_PATTERN, "Gram SVD rounding", DevExecSpace(), scratch_size,
      scratch_level, 0, 1, KOKKOS_LAMBDA(parthenon::team_mbr_t tm, const int dummy) {
        auto &ts = tm.team_scratch(scratch_level);

        // pull out cores object
        auto cores = cores_device_;

        // assign scratch space for right Gram matrices and compute all the right
        // Gram matrices (recursive sweep from right to left)
        // this is 2D: GR(ts, Ngram, RMax * RMax) so is contiguous and
        // kokkos 2d views handle indexing
        ScratchPad2D<Real> GR(ts, Ngram, RMax * RMax);
        CalculateRightGramMatrices(GR, tm);

        // assign scratch space for left Gram matrices and compute all the left
        // Gram matrices (recursive sweep from left to right)
        // this is 2D: GL(ts, Ngram, RMax * RMax) so is contiguous and
        // kokkos 2d views handle indexing
        ScratchPad2D<Real> GL(ts, Ngram, RMax * RMax);
        CalculateLeftGramMatrices(GL, tm);

        // allocate temporary cores needed for update
        const int PIMax = this->GetMaximumPhysicaIndexSize();
        ScratchPad1D<Real> temporary_core_scratch(ts, RMax * PIMax * RMax);

        Real *p0 = &temporary_core_scratch(0);
        View3DUnmanaged CTmp(p0, cores(0).GetLeftRank(), cores(0).GetPhysicalIndexSize(), 
            cores(0).GetRightRank());

        // copy out the first core into this temporary core, since there is no
        // left index space update for the first core
        par_for_inner(tm, 0, cores(0).GetLeftRank() - 1, 0, 
            cores(0).GetRightRank() - 1, 0, cores(0).GetPhysicalIndexSize(),
            [&](const int rl, const int rr, const int i) { 
            CTmp(rl, i, rr) = cores(0)(rl, i, rr); 
            });


        // loop over bond spaces and compute SVDs
        for (int n = 0; n < Ngram; ++n) {
          const std::size_t Rn = cores(n).GetRightRank();

          // assign scratch space for eigenvalue decompositions of left and
          // right Gram matrices
          ScratchPad2D<Real> EVL(ts, Rn, Rn); // left Gram's eigenvectors
          ScratchPad2D<Real> EVR(ts, Rn, Rn); // right Gram's eigenvectors
          ScratchPad1D<Real> ELamL(ts, Rn); // left Gram's eigenvalues (diag)
          ScratchPad1D<Real> ELamR(ts, Rn); // right Gram's eigenvalues (diag)

          // assign scratch space for SVD
          ScratchPad2D<Real> svdU(ts, Rn, Rn); // left singular vectors
          ScratchPad2D<Real> svdV(ts, Rn, Rn); // right singular vectors
          ScratchPad1D<Real> svdS(ts, Rn); // singular values (diag)

          // pull out views of the Gram matrices for this bond space
          // using raw pointer to beginning of the gram matrices
          // this is a bit naughty
          Real *pL = &GL(n, 0);
          View2DUnmanaged GLn(pL, Rn, Rn);
          Real *pR = &GR(n, 0);
          View2DUnmanaged GRn(pR, Rn, Rn);

          // compute SVD for this bond space, returning:
          // left Gram's eigenvectors, right Gram's eigenvectors
          // left Gram's eigenvalues (diag), right Gram's eigenvalues (diag)
          // SVD: left singular vectors, right singular vectors, singular values (diag)
          CalculateGramSVD(n, GLn, GRn, EVL, EVR, ELamL, ELamR, svdU, svdV, svdS, tm, scratch_level);

          // select singular modes and obtain the number of retained modes and
          // a map to them
          ScratchPad1D<int> keep(ts, Rn);
          ScratchPad1D<int> gamma_map(ts, Rn);
          int Rn_new = SelectSingularModes(n, svdS, keep, gamma_map, eps);

          // update the right index space of temporary core (which already had
          // its left index space updated) and write back into the TT's core.
          // Update the left index space of core n+1 and write the result into
          // the temporary core core n+1.
          UpdateCoreIndexSpaces(n, Rn_new, gamma_map, svdU, svdV, svdS, ELamL, EVL,
              ELamR, EVR, tm, temporary_core_scratch);

        } // loop over bond spaces

      }); // par_for_outer
}

KOKKOS_INLINE_FUNCTION
void TensorTrain::UpdateCoreIndexSpaces(const int n, const int Rn_new, ScratchPad1D<int> &gamma_map,
    ScratchPad2D<Real> &svdU, ScratchPad2D<Real> &svdV, ScratchPad1D<Real> &svdS,
    ScratchPad1D<Real> &ELamL, ScratchPad2D<Real> &EVL, ScratchPad1D<Real> &ELamR,
    ScratchPad2D<Real> &EVR, parthenon::team_mbr_t tm, ScratchPad1D<Real> &temporary_core_scratch) {

    // pull out cores object
    auto cores = this->cores_device_;
    //const int Ncores = cores.GetNumCores();
    const int Ncores = cores.size();
    const int Rn = static_cast<int>(cores(n).GetRightRank());

    // create a view of the temporary core that we are going to read from
    Real *p = &temporary_core_scratch(0);
    View3DUnmanaged CTmp(p, cores(n).GetLeftRank(), cores(n).GetPhysicalIndexSize(), 
        cores(n).GetRightRank());

    // first use what's in the temporary core (already has left index
    // space updated) to update the right index space of core n and store
    // in the actual tensor object.
    const int Rnm1 = static_cast<int>(cores(n).GetLeftRank());
    const int PIS = static_cast<int>(cores(n).GetPhysicalIndexSize());
    par_for_inner(tm, 0, Rnm1 - 1, 0, Rn_new - 1, 0, PIS - 1, 
        [&](const int alf, const int gam, const int i) {
      Real accum{0.};
      for (int bet = 0; bet < Rn; bet++) {
        for (int mu = 0; mu < Rn; mu++) {
          accum += CTmp(alf, i, bet) * EVL(bet, mu) 
          / std::sqrt(ELamL(mu)) * svdU(mu, gamma_map(gam));
        }
      }
      cores(n)(alf, i, gam) = accum;
    });
    tm.team_barrier();

    // now update the left index space of core n+1 and store the result
    // in the temporary core
    // (not for the last core)
    if (n < Ncores - 1) {

      // create a view of the temporary core that we are going to write to
      View3DUnmanaged CTmp(p, cores(n+1).GetLeftRank(), cores(n+1).GetPhysicalIndexSize(), 
          cores(n+1).GetRightRank());

      const int Rnp1 = static_cast<int>(cores(n+1).GetRightRank());
      const int PISp1 = static_cast<int>(cores(n+1).GetPhysicalIndexSize());
      par_for_inner(tm, 0, Rn_new - 1, 0, Rnp1 - 1, 0, PISp1 - 1, 
          [&](const int gam, const int alf, const int i) {
        Real accum{0.};
        for (int bet = 0; bet < Rn; bet++) {
          for (int nu = 0; nu < Rn; nu++) {
            accum += svdS(gamma_map(gam)) *svdV(nu, gamma_map(gam))
            / std::sqrt(ELamR(nu)) * EVR(bet,nu)
            * cores(n+1)(bet, i, alf);
          }
        }
        CTmp(gam, i, alf) = accum;
      });
    }

} // TensorTrain::UpdateCoreIndexSpaces

} // namespace tensors
} // namespace parthenon































