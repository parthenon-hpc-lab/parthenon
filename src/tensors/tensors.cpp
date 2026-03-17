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

KOKKOS_INLINE_FUNCTION
void GramSVDStorage::ComputeSVD(const int Rn, const int nnzL, const int nnzR,
                                const parthenon::team_mbr_t &tm) {

  // initialize SVD to zero
  par_for_inner(tm, 0, Rn - 1, 0, Rn - 1, [&](const int i, const int j) {
    SVDS()(i) = 0.0;
    SVDU()(i, j) = 0.0;
    SVDV()(i, j) = 0.0;
  });
  tm.team_barrier();

  if (nnzL == 0 || nnzR == 0) {
    return;
  }

  // effective rank 1 case
  else if (nnzL == 1 || nnzR == 1) {
    // Compute Frobenius norm
    Real sigma = 0.0;
    par_reduce_inner(
        parthenon::InnerLoopPatternTTR(), tm, 0, Rn - 1, 0, Rn - 1,
        [&](const int i, const int j, Real &tmp) { tmp += M()(i, j) * M()(i, j); },
        Kokkos::Sum<Real, parthenon::DevMemSpace>(sigma)); // par_reduce_inner

    sigma = safe_sqrt(sigma);

    if (sigma < 1e-15) return;

    SVDS()(0) = sigma;

    // --- Compute right singular vector first ---

    // Find a nonzero row (all threads; for larger ranks maybe we should parallelize)
    int pivot = -1;
    Real row_norm = 0.0;

    for (int i = 0; i < Rn; ++i) {
      row_norm = 0.0;
      for (int j = 0; j < Rn; ++j) {
        row_norm += M()(i, j) * M()(i, j);
      }

      if (row_norm > 1e-15) {
        pivot = i;
        break;
      }
    }

    row_norm = safe_sqrt(row_norm);

    // v = normalized pivot row
    par_for_inner(tm, 0, Rn - 1,
                  [&](const int j) { SVDV()(j, 0) = M()(pivot, j) / row_norm; });
    tm.team_barrier();

    // u = M v / sigma
    par_for_inner(tm, 0, Rn - 1, [&](const int i) {
      Real val = 0.0;
      for (int j = 0; j < Rn; ++j) {
        val += M()(i, j) * SVDV()(j, 0);
      }
      SVDU()(i, 0) = val / sigma;
    });
    tm.team_barrier();
  } else {

    // regular case
    // SquareSVD::execute(&M(), &SVDU(), &SVDV(), SVDS().data());
    SquareSVD::execute(tm, &M(), &SVDU(), &SVDV(), SVDS().data(), RealScratch().data(),
                       SizeTScratch().data());
  }
  // PrintRealMat(M(), Rn, tm, "M");
  // PrintRealMat(SVDU(), Rn, tm, "U");
  // PrintRealMat(SVDV(), Rn, tm, "V");
  // PrintRealVec(SVDS(), Rn, tm, "S");
}

// ============================================================
// GramSVDStorage Scratch size computation
// ============================================================
// Called from host, so we can use TT member functions to get sizes
size_t GramSVDStorage::GetScratchSize(const TensorTrain &TT, int evd_scratch_max) {
  const int RMax = TT.GetMaximumRightRank();
  const int PIMax = TT.GetMaximumPhysicalIndexSize();
  const int NGram = TT.GetNumCores() - 1;
  int total = 0;
  for (int n = 0; n < NGram; ++n)
    total += TT.GetRightRank(n) * TT.GetRightRank(n);

  return 2 * (NumRealCores * ScratchPad3D<Real>::shmem_size(RMax, PIMax, RMax) +
              NumRealMatrices * ScratchPad2D<Real>::shmem_size(RMax, RMax) +
              NumRealVecs * ScratchPad1D<Real>::shmem_size(RMax) +
              NumIntVecs * ScratchPad1D<int>::shmem_size(RMax) +
              NumRealAlgoVecs * ScratchPad1D<Real>::shmem_size(evd_scratch_max) +
              NumSizeTAlgoVecs * ScratchPad1D<std::size_t>::shmem_size(evd_scratch_max) +
              2 * ScratchPad1D<Real>::shmem_size(total) +
              2 * ScratchPad1D<int>::shmem_size(NGram));
}

// ============================================================
// GramSVDStorage Constructor
// ============================================================

KOKKOS_INLINE_FUNCTION
GramSVDStorage::GramSVDStorage(ScratchSpace ts, const TensorTrainDeviceView ttd,
                               int evd_scratch_max_, const parthenon::team_mbr_t &tm)
    : evd_scratch_max(evd_scratch_max_) {
  // ----- Compute packed Gram offsets -----

  gram_offsets_ = ScratchPad1D<int>(ts, ttd.NGram);
  gram_sizes_ = ScratchPad1D<int>(ts, ttd.NGram);

  int total = 0;

  for (int n = 0; n < ttd.NGram; ++n) {
    int Rn = ttd.cores(n).GetRightRank();
    gram_offsets_(n) = total;
    gram_sizes_(n) = Rn;
    total += Rn * Rn;
  }

  GL_storage_ = GramStorage(ts, total);
  GR_storage_ = GramStorage(ts, total);

  // ----- Core scratch -----

  for (int i = 0; i < NumRealCores; ++i)
    real_cores_storage_[i] = RealCoreStorage(ts, ttd.RMax, ttd.PIMax, ttd.RMax);

  // ----- Work matrices -----

  for (int i = 0; i < NumRealMatrices; ++i)
    real_mats_storage_[i] = RealMat(ts, ttd.RMax, ttd.RMax);

  // ----- Rank-sized vectors -----

  for (int i = 0; i < NumRealVecs; ++i)
    real_vecs_storage_[i] = RealVec(ts, ttd.RMax);

  for (int i = 0; i < NumIntVecs; ++i)
    int_vecs_storage_[i] = IntVec(ts, ttd.RMax);

  // ----- Algorithm scratch -----

  for (int i = 0; i < NumRealAlgoVecs; ++i)
    real_algo_vecs_storage_[i] = RealVec(ts, evd_scratch_max);

  for (int i = 0; i < NumSizeTAlgoVecs; ++i)
    sizet_algo_vecs_storage_[i] = SizeTVec(ts, evd_scratch_max);

  ResizeRankViews(ttd.RMax, evd_scratch_max);
}

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
// TODO(@SWJ): we can exploit the symmetry of the gram matrices so that we only
// perform the contractions for the upper triangular part and then copy into the
// lower triangular part
void TensorTrain::CalculateRightGramMatrices(GramSVDStorage &GS,
                                             TensorTrainDeviceView ttd,
                                             const parthenon::team_mbr_t &member) {
  const int Ngram = ttd.NGram;
  // pull out cores object
  auto cores = ttd.cores;

  for (int n = Ngram - 1; n >= 0; --n) {

    // rank of this core
    std::size_t Rn = cores(n).GetRightRank();
    // rank of next core
    std::size_t Rnp1 = cores(n + 1).GetRightRank();

    // loop over elements of this Gram matrix
    // TODO experiment with patterns for reductions to find what is optimal
    for (int a = 0; a < cores(n).GetRightRank(); a++) {
      for (int ap = 0; ap < cores(n).GetRightRank(); ap++) {

        Real accum{0.};
        if (n == Ngram - 1) {
          // Last gram matrix is 1x1, contract only over physical index space
          par_reduce_inner(
              parthenon::InnerLoopPatternTTR(), member, 0,
              cores(n + 1).GetPhysicalIndexSize() - 1,
              [&](const int i, Real &tmp) {
                tmp += cores(n + 1)(a, i, 0) * cores(n + 1)(ap, i, 0);
              },
              Kokkos::Sum<Real, parthenon::DevMemSpace>(accum)); // par_reduce_inner
        } else {
          par_reduce_inner(
              parthenon::InnerLoopPatternTTR(), member, 0, Rnp1 - 1, 0, Rnp1 - 1, 0,
              cores(n + 1).GetPhysicalIndexSize() - 1,
              [&](const int b, const int bp, const int i, Real &tmp) {
                tmp +=
                    cores(n + 1)(a, i, b) * GS.GR(n + 1)(b, bp) * cores(n + 1)(ap, i, bp);
              },
              Kokkos::Sum<Real, parthenon::DevMemSpace>(accum)); // par_reduce_inner
        } // if (n == Ngram - 1)
        GS.GR(n)(a, ap) = accum;
      }
    }
  }
}

KOKKOS_INLINE_FUNCTION
// TODO(@SWJ): we can exploit the symmetry of the gram matrices so that we only
// perform the contractions for the upper triangular part and then copy into the
// lower triangular part
void TensorTrain::CalculateLeftGramMatrices(GramSVDStorage &GS, TensorTrainDeviceView ttd,
                                            const parthenon::team_mbr_t &member) {
  const int Ngram = ttd.NGram;
  // pull out cores object
  auto cores = ttd.cores;

  for (int n = 0; n < Ngram; ++n) {

    // rank of this core
    std::size_t Rn = cores(n).GetRightRank();

    // loop over elements of this Gram matrix
    // TODO experiment with patterns for reductions to find what is optimal
    for (int b = 0; b < cores(n).GetRightRank(); b++) {
      for (int bp = 0; bp < cores(n).GetRightRank(); bp++) {
        // perform the contraction

        Real accum{0.};
        if (n == 0) {
          par_reduce_inner(
              parthenon::InnerLoopPatternTTR(), member, 0,
              cores(n).GetPhysicalIndexSize() - 1,
              [&](const int i, Real &tmp) {
                tmp += cores(n)(0, i, b) * cores(n)(0, i, bp);
              },
              Kokkos::Sum<Real, parthenon::DevMemSpace>(accum)); // par_reduce_inner
        }

        else {
          // rank of previous core
          std::size_t Rnm1 = cores(n - 1).GetRightRank();
          par_reduce_inner(
              parthenon::InnerLoopPatternTTR(), member, 0,
              cores(n).GetPhysicalIndexSize() - 1, 0, cores(n - 1).GetRightRank() - 1, 0,
              cores(n - 1).GetRightRank() - 1,
              [&](const int i, const int a, const int ap, Real &tmp) {
                tmp += cores(n)(a, i, b) * GS.GL(n - 1)(a, ap) * cores(n)(ap, i, bp);
              },
              Kokkos::Sum<Real, parthenon::DevMemSpace>(accum)); // par_reduce_inner
        }
        GS.GL(n)(b, bp) = accum;
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
void TensorTrain::CalculateGramSVD(const int n, parthenon::team_mbr_t member,
                                   TensorTrainDeviceView ttd, GramSVDStorage &GS) {

  // pull out cores object
  auto cores = ttd.cores;
  const std::size_t Rn = cores(n).GetRightRank();

  /////////////////////////////////////////////////////////////////////////////////////
  // LEFT GRAM
  /////////////////////////////////////////////////////////////////////////////////////

  // write left Gram matrix to A (so that it is not destroyed)
  // TODO(SWJ): We can actually probably just destroy it since it is not needed
  // once we have the eigenvalue decomposition of it
  par_for_inner(member, 0, Rn - 1, 0, Rn - 1,
                [&](const int b, const int bp) { GS.A()(b, bp) = GS.GL(n)(b, bp); });
  member.team_barrier();

  // PrintRealMat(GS.A(), Rn, member, "A left");

  // perform the eigenvalue decomposition
  int info = SymmetricEVD::execute(member, &GS.A(), &GS.EVL(), GS.EvalL().data(),
                                   GS.RealScratch().data(), GS.SizeTScratch().data());
  member.team_barrier();

  // PrintRealMat(GS.EVL(), Rn, member, "EVL");
  // PrintRealVec(GS.EvalL(), Rn, member, "EvalL");

  /////////////////////////////////////////////////////////////////////////////////////
  // RIGHT GRAM
  /////////////////////////////////////////////////////////////////////////////////////

  // write right Gram matrix to A (so that it is not destroyed)
  // TODO(SWJ): We can actually probably just destroy it since it is not needed
  // once we have the eigenvalue decomposition of it
  par_for_inner(member, 0, Rn - 1, 0, Rn - 1,
                [&](const int a, const int ap) { GS.A()(a, ap) = GS.GR(n)(a, ap); });
  member.team_barrier();

  // PrintRealMat(GS.A(), Rn, member, "A right");

  // perform the eigenvalue decomposition
  SymmetricEVD::execute(member, &GS.A(), &GS.EVR(), GS.EvalR().data(),
                        GS.RealScratch().data(), GS.SizeTScratch().data());

  // PrintRealMat(GS.EVR(), Rn, member, "EVR");
  // PrintRealVec(GS.EvalR(), Rn, member, "EvalR");

  // clean the eigensystems
  const Real eps{1e-12};
  int nnzL = GS.CleanAndCountNonZeroEigenValues(GS.EVL(), GS.EvalL(), Rn, eps);
  int nnzR = GS.CleanAndCountNonZeroEigenValues(GS.EVR(), GS.EvalR(), Rn, eps);

  //////////////////////////////////////////////////////////////////////////////////
  // Now we have the left gram's eigenvalues and eigenvectors ELamL, EVL
  // and the right gram's eigenvalues and eigenvectors ELamR, EVR.
  // Construct the matrix that we want to obtain the truncated SVD
  // of.

  for (int a = 0; a < Rn; a++) {
    for (int b = 0; b < Rn; b++) {
      Real accum{0.};
      par_reduce_inner(
          parthenon::InnerLoopPatternTTR(), member, 0, Rn - 1,
          [&](const int i, Real &tmp) { tmp += GS.EVL()(i, a) * GS.EVR()(i, b); },
          Kokkos::Sum<Real, parthenon::DevMemSpace>(accum)); // par_reduce_inner

      GS.M()(a, b) = safe_sqrt(GS.EvalL()(a)) * accum * safe_sqrt(GS.EvalR()(b));
    }
  }

  // compute the SVD of M
  GS.ComputeSVD(Rn, nnzL, nnzR, member);
}

// Gram-SVD TT rounding with tolerance eps. Reduces TT ranks while
// preserving the tensor up to Frobenius error eps.
void TensorTrain::GramSVDRound(const Real eps) {
  // get max right ranks, which set max gram matrix sizes for malloc
  const int RMax = GetMaximumRightRank();
  const int PIMax = GetMaximumPhysicalIndexSize();
  const int evd_scratch_max = std::max(SquareSVD::sizet_scratch_size(RMax),
                                       SymmetricEVD::sizet_scratch_size(RMax));
  // number of Gram matrices (this many left, this many right)
  const int NGram = GetNumCores() - 1;

  // create a device view of the TT with relevant metadata and pointer to device cores
  // TensorTrainDeviceView ttd = GetDeviceView();
  auto cores = cores_device_;

  const size_t scratch_size = GramSVDStorage::GetScratchSize(*this, evd_scratch_max);
  const int scratch_level = 0; // ? team or thread?

  par_for_outer(
      DEFAULT_OUTER_LOOP_PATTERN, "Gram SVD rounding", DevExecSpace(), scratch_size,
      scratch_level, 0, 0, KOKKOS_LAMBDA(parthenon::team_mbr_t tm, const int dummy) {
        auto &ts = tm.team_scratch(scratch_level);

        TensorTrainDeviceView ttd(cores, RMax, PIMax, NGram);

        // construct GRAMSVDStorage object
        GramSVDStorage GS(ts, ttd, evd_scratch_max, tm);

        // compute all the right Gram matrices (recursive sweep from right to
        // left)
        CalculateRightGramMatrices(GS, ttd, tm);

        // compute all the left Gram matrices (recursive sweep from left to
        // right)
        CalculateLeftGramMatrices(GS, ttd, tm);

        // loop over bond spaces and compute SVDs
        for (int n = 0; n < ttd.NGram; ++n) {
          const std::size_t Rn = cores(n).GetRightRank();
          const std::size_t PIn = cores(n).GetPhysicalIndexSize();
          const int evd_scratch_n = std::max(SquareSVD::sizet_scratch_size(Rn),
                                             SymmetricEVD::sizet_scratch_size(Rn));

          // resize views for this bond space
          GS.ResizeRankViews(Rn, evd_scratch_n);

          if (n == 0) {
            // copy out the first core into this temporary core, since there is no
            // left index space update for the first core
            GS.ResizeCoreView(1, PIn, Rn);
            par_for_inner(tm, 0, cores(n).GetRightRank() - 1, 0,
                          cores(n).GetPhysicalIndexSize() - 1,
                          [&](const int rr, const int i) {
                            GS.CTmp()(0, i, rr) = cores(n)(0, i, rr);
                          });
            tm.team_barrier();
          }

          // compute SVD for this bond space, returning:
          // left Gram's eigenvectors, right Gram's eigenvectors
          // left Gram's eigenvalues (diag), right Gram's eigenvalues (diag)
          // SVD: left singular vectors, right singular vectors, singular values (diag)
          CalculateGramSVD(n, tm, ttd, GS);

          // select singular modes and obtain the number of retained modes and
          // a map to them
          int Rn_new = SelectSingularModes(n, GS, ttd, eps);

          tm.team_barrier();

          // update the right index space of temporary core (which already had
          // its left index space updated) and write back into the TT's core.
          // Update the left index space of core n+1 and write the result into
          // the temporary core core n+1.
          UpdateCoreIndexSpaces(n, Rn_new, tm, ttd, GS);

        } // loop over bond spaces
      }); // par_for_outer

  // // Now, on host, resize the cores
  for (int n = 0; n < GetNumCores(); n++) {
    cores_host_(n).ResizeToNewShape();
  }
  // Sync the device cores to reflect the new resized cores on host
  SyncDeviceCores();
  Kokkos::fence();
}

// This is Algorithm 5 in Al Daas et al. Note the sqrt(sigma) used in both L/R
// core updates.
// Updating bond space n
KOKKOS_INLINE_FUNCTION
void TensorTrain::UpdateCoreIndexSpaces(const int n, const int Rn_new,
                                        parthenon::team_mbr_t tm,
                                        TensorTrainDeviceView ttd, GramSVDStorage &GS) {

  // pull out cores object
  auto cores = ttd.cores;
  // const int Ncores = cores.GetNumCores();
  const int Ncores = cores.size();
  const int Rnm1 = static_cast<int>(cores(n).GetLeftRank());
  const int Rn = static_cast<int>(cores(n).GetRightRank());
  const int PIn = static_cast<int>(cores(n).GetPhysicalIndexSize());

  // if (n > 0) return; // only update first bond space

  // first use what's in the temporary core (already has left index
  // space updated) to update the right index space of core n and store
  // in the actual tensor object.
  par_for_inner(tm, 0, Rnm1 - 1, 0, Rn_new - 1, 0, PIn - 1,
                [&](const int alf, const int gam, const int i) {
                  Real sqrt_sigma = safe_sqrt(GS.SVDS()(GS.ModeMap()(gam)));
                  Real accum{0.};
                  for (int bet = 0; bet < Rn; bet++) {
                    for (int mu = 0; mu < Rn; mu++) {
                      accum += GS.CTmp()(alf, i, bet) * GS.EVL()(bet, mu) /
                               (safe_sqrt(GS.EvalL()(mu)) + 1e-15) *
                               GS.SVDU()(mu, GS.ModeMap()(gam));
                    }
                  }
                  cores(n)(alf, i, gam) = accum * sqrt_sigma;
                });
  tm.team_barrier();

  // update the shape of the right index space
  cores(n).SetShape(Rnm1, PIn, Rn_new);

  // now update the left index space of core n+1 and store the result
  // in the temporary core
  const int Rnp1 = static_cast<int>(cores(n + 1).GetRightRank());
  const int PInp1 = static_cast<int>(cores(n + 1).GetPhysicalIndexSize());

  // Ensure view of temporary core is the same size as core n+1
  GS.ResizeCoreView(Rn_new, PInp1, Rnp1);

  par_for_inner(tm, 0, Rn_new - 1, 0, Rnp1 - 1, 0, PInp1 - 1,
                [&](const int gam, const int alf, const int i) {
                  Real sqrt_sigma = safe_sqrt(GS.SVDS()(GS.ModeMap()(gam)));
                  Real accum{0.};
                  for (int bet = 0; bet < Rn; bet++) {
                    for (int nu = 0; nu < Rn; nu++) {
                      accum += GS.SVDV()(nu, GS.ModeMap()(gam)) /
                               (safe_sqrt(GS.EvalR()(nu)) + 1e-15) * GS.EVR()(bet, nu) *
                               cores(n + 1)(bet, i, alf);
                    }
                  }
                  GS.CTmp()(gam, i, alf) = accum * sqrt_sigma;
                });
  tm.team_barrier();

  // update the shape of the left index space
  cores(n + 1).SetShape(Rn_new, PInp1, Rnp1);

  // The last bond needs to write the left-index-updated temporary core back to
  // the actual core
  if (n == Ncores - 2) {
    par_for_inner(tm, 0, Rn_new - 1, 0, Rnp1 - 1, 0, PInp1 - 1,
                  [&](const int iL, const int iR, const int ic) {
                    cores(n + 1)(iL, ic, iR) = GS.CTmp()(iL, ic, iR);
                  });
  }
  tm.team_barrier();

} // TensorTrain::UpdateCoreIndexSpaces

} // namespace tensors
} // namespace parthenon
