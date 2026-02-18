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

// ============================================================
// GramSVDStorage Scratch size computation
// ============================================================

size_t GramSVDStorage::GetScratchSize(const TensorTrain &TT, int evd_scratch_max) {
  int RMax = TT.GetMaximumRightRank();
  int PIMax = TT.GetMaximumPhysicalIndexSize();
  int Ngram = TT.GetNumCores() - 1;

  int total = 0;
  for (int n = 0; n < Ngram; ++n)
    total += TT.GetRightRank(n) * TT.GetRightRank(n);

  return NumRealCores * ScratchPad3D<Real>::shmem_size(RMax, PIMax, RMax) +
         NumRealMatrices * ScratchPad2D<Real>::shmem_size(RMax, RMax) +
         NumRealVecs * ScratchPad1D<Real>::shmem_size(RMax) +
         NumIntVecs * ScratchPad1D<int>::shmem_size(RMax) +
         NumRealAlgoVecs * ScratchPad1D<Real>::shmem_size(evd_scratch_max) +
         NumSizeTAlgoVecs * ScratchPad1D<std::size_t>::shmem_size(evd_scratch_max) +
         2 * ScratchPad1D<Real>::shmem_size(total) +
         2 * ScratchPad1D<int>::shmem_size(Ngram);
}

// ============================================================
// GramSVDStorage Constructor
// ============================================================

KOKKOS_INLINE_FUNCTION
GramSVDStorage::GramSVDStorage(ScratchSpace ts, const TensorTrain &TT,
                               int evd_scratch_max_)
    : evd_scratch_max(evd_scratch_max_) {
  RMax = TT.GetMaximumRightRank();
  PIMax = TT.GetMaximumPhysicalIndexSize();
  Ngram_ = TT.GetNumCores() - 1;

  // ----- Compute packed Gram offsets -----

  gram_offsets_ = ScratchPad1D<int>(ts, Ngram_);
  gram_sizes_ = ScratchPad1D<int>(ts, Ngram_);

  int total = 0;

  for (int n = 0; n < Ngram_; ++n) {
    int Rn = TT.GetRightRank(n);
    gram_offsets_(n) = total;
    gram_sizes_(n) = Rn;
    total += Rn * Rn;
  }

  GL_storage_ = GramStorage(ts, total);
  GR_storage_ = GramStorage(ts, total);

  // ----- Core scratch -----

  for (int i = 0; i < NumRealCores; ++i)
    real_cores_storage_[i] = RealCoreStorage(ts, RMax, PIMax, RMax);

  // ----- Work matrices -----

  for (int i = 0; i < NumRealMatrices; ++i)
    real_mats_storage_[i] = RealMat(ts, RMax, RMax);

  // ----- Rank-sized vectors -----

  for (int i = 0; i < NumRealVecs; ++i)
    real_vecs_storage_[i] = RealVec(ts, RMax);

  for (int i = 0; i < NumIntVecs; ++i)
    int_vecs_storage_[i] = IntVec(ts, RMax);

  // ----- Algorithm scratch -----

  for (int i = 0; i < NumRealAlgoVecs; ++i)
    real_algo_vecs_storage_[i] = RealVec(ts, evd_scratch_max);

  for (int i = 0; i < NumSizeTAlgoVecs; ++i)
    sizet_algo_vecs_storage_[i] = SizeTVec(ts, evd_scratch_max);

  ResizeRankViews(RMax, evd_scratch_max);
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
void TensorTrain::CalculateRightGramMatrices(GramSVDStorage &GS,
                                             const parthenon::team_mbr_t &member) {
  const int Ngram = this->GetNumCores() - 1;
  // pull out cores object
  auto cores = this->cores_device_;

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
        }                                                        // if (n == Ngram - 1)
        GS.GR(n)(a, ap) = accum;
        printf("Right gram %d %d = %22.15e\n", a, ap, accum);
      }
    }
  }
}

KOKKOS_INLINE_FUNCTION
void TensorTrain::CalculateLeftGramMatrices(GramSVDStorage &GS,
                                            const parthenon::team_mbr_t &member) {
  const int Ngram = this->GetNumCores() - 1;
  // pull out cores object
  auto cores = this->cores_device_;

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
        printf("Left gram %d %d = %23.15e\n", b, bp, accum);
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
                                   GramSVDStorage &GS) {

  // pull out cores object
  auto cores = this->cores_device_;
  const std::size_t Rn = cores(n).GetRightRank();

  /////////////////////////////////////////////////////////////////////////////////////
  // LEFT GRAM
  /////////////////////////////////////////////////////////////////////////////////////

  // write left Gram matrix to A (so that it is not destroyed)
  // TODO(SWJ) we can probably destroy it actually
  par_for_inner(member, 0, Rn - 1, 0, Rn - 1,
                [&](const int b, const int bp) { GS.A()(b, bp) = GS.GL(n)(b, bp); });
  member.team_barrier();

  // perform the eigenvalue decomposition
  SymmetricEVD::execute(member, &GS.A(), &GS.EVL(), GS.EvalL().data(),
                        GS.EVDRealScratch().data(), GS.EVDSizeTScratch().data());

  /////////////////////////////////////////////////////////////////////////////////////
  // RIGHT GRAM
  /////////////////////////////////////////////////////////////////////////////////////

  // write left Gram matrix to A (so that it is not destroyed)
  // TODO(SWJ): We can actually probably just destroy it since it is not needed
  // once we have the eigenvalue decomposition of it
  par_for_inner(member, 0, Rn - 1, 0, Rn - 1,
                [&](const int a, const int ap) { GS.A()(a, ap) = GS.GR(n)(a, ap); });
  member.team_barrier();

  // perform the eigenvalue decomposition
  SymmetricEVD::execute(member, &GS.A(), &GS.EVR(), GS.EvalR().data(),
                        GS.EVDRealScratch().data(), GS.EVDSizeTScratch().data());

  printf("Eigenvalues L: ");
  for (int i = 0; i < Rn; i++)
    printf("%e ", GS.EvalL()(i));
  printf("\n");
  printf("Eigenvalues R: ");
  for (int i = 0; i < Rn; i++)
    printf("%e ", GS.EvalR()(i));
  printf("\n");

  // clean the eigensystems
  Real eps{1e-12};
  Real LambdamaxL{0.};
  Real LambdamaxR{0.};
  for (int i = 0; i < Rn; i++) {
    LambdamaxL = std::max(LambdamaxL, GS.EvalL()(i));
    LambdamaxR = std::max(LambdamaxR, GS.EvalR()(i));
  }
  for (int i = 0; i < Rn; i++) {
    if (GS.EvalL()(i) < eps * LambdamaxL) GS.EvalL()(i) = 0.;
    if (GS.EvalR()(i) < eps * LambdamaxR) GS.EvalR()(i) = 0.;
  }

  printf("Cleaned Eigenvalues L: ");
  for (int i = 0; i < Rn; i++)
    printf("%e ", GS.EvalL()(i));
  printf("\n");
  printf("Cleaned Eigenvalues R: ");
  for (int i = 0; i < Rn; i++)
    printf("%e ", GS.EvalR()(i));
  printf("\n");

  printf("VL:\n");
  for (int i = 0; i < Rn; ++i) {
    for (int j = 0; j < Rn; ++j) {
      printf("  %12.5e", GS.EVL()(i, j));
    }
    printf("\n");
  }

  printf("VR:\n");
  for (int i = 0; i < Rn; ++i) {
    for (int j = 0; j < Rn; ++j) {
      printf("  %12.5e", GS.EVR()(i, j));
    }
    printf("\n");
  }

  //////////////////////////////////////////////////////////////////////////////////
  // Now we have the left gram's eigenvalues and eigenvectors ELamL, EVL
  // and the right gram's eigenvalues and eigenvectors ELamR, EVR.
  // Construct the matrix that we want to obtain the truncated SVD
  // of.

  // as a par reduce inner
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

  printf("M:\n");
  for (int i = 0; i < Rn; ++i) {
    for (int j = 0; j < Rn; ++j) {
      printf("  %12.5e", GS.M()(i, j));
    }
    printf("\n");
  }

  if (Rn == 1) {
    // Trivial case
    GS.SVDS()(0) = GS.M()(0, 0);
    GS.SVDU()(0, 0) = 1.0;
    GS.SVDV()(0, 0) = 1.0;
  } else {
    SquareSVD::execute(&GS.M(), &GS.SVDU(), &GS.SVDV(), GS.SVDS().data());
  }

  printf("SVDU:\n");
  for (int i = 0; i < Rn; ++i) {
    for (int j = 0; j < Rn; ++j) {
      printf("  %12.5e", GS.SVDU()(i, j));
    }
    printf("\n");
  }

  printf("SVDV:\n");
  for (int i = 0; i < Rn; ++i) {
    for (int j = 0; j < Rn; ++j) {
      printf("  %12.5e", GS.SVDV()(i, j));
    }
    printf("\n");
  }

  printf("SVDS:\n");
  for (int i = 0; i < Rn; ++i) {
    printf("  %12.5e", GS.SVDS()(i));
  }
  printf("\n");
}

// Gram-SVD TT rounding with tolerance eps. Reduces TT ranks while
// preserving the tensor up to Frobenius error eps.
void TensorTrain::GramSVDRound(const Real eps) {
  // get max right ranks, which set max gram matrix sizes for malloc
  const int RMax = GetMaximumRightRank();
  int evd_scratch_max = SymmetricEVD::sizet_scratch_size(RMax);

  // number of Gram matrices (this many left, this many right)
  const int Ngram = GetNumCores() - 1;

  const size_t scratch_size = GramSVDStorage::GetScratchSize(*this, evd_scratch_max);
  const int scratch_level = 0; // ? team or thread?

  par_for_outer(
      DEFAULT_OUTER_LOOP_PATTERN, "Gram SVD rounding", DevExecSpace(), scratch_size,
      scratch_level, 0, 0, KOKKOS_LAMBDA(parthenon::team_mbr_t tm, const int dummy) {
        auto &ts = tm.team_scratch(scratch_level);

        // construct GRAMSVDStorage object
        GramSVDStorage GS(ts, *this, evd_scratch_max);

        // pull out cores object
        auto cores = cores_device_;

        // compute all the right Gram matrices (recursive sweep from right to
        // left)
        CalculateRightGramMatrices(GS, tm);

        // compute all the left Gram matrices (recursive sweep from left to
        // right)
        CalculateLeftGramMatrices(GS, tm);

        for (int n = 0; n < Ngram; ++n) {
          Real traceL = 0;
          Real traceR = 0;
          int Rn = cores(n).GetRightRank();

          for (int i = 0; i < Rn; ++i) {
            traceL += GS.GL(n)(i, i);
            traceR += GS.GR(n)(i, i);
          }

          printf("bond %d traceL=%e traceR=%e\n", n, traceL, traceR);
        }

        // loop over bond spaces and compute SVDs
        for (int n = 0; n < Ngram; ++n) {
          const std::size_t Rn = cores(n).GetRightRank();
          const std::size_t PIn = cores(n).GetPhysicalIndexSize();
          const int evd_scratch_n = SymmetricEVD::sizet_scratch_size(Rn);

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
          CalculateGramSVD(n, tm, GS);

          // select singular modes and obtain the number of retained modes and
          // a map to them
          int Rn_new = SelectSingularModes(n, GS, eps);

          // update the right index space of temporary core (which already had
          // its left index space updated) and write back into the TT's core.
          // Update the left index space of core n+1 and write the result into
          // the temporary core core n+1.
          UpdateCoreIndexSpaces(n, Rn_new, tm, GS);

        } // loop over bond spaces
      }); // par_for_outer

  // Now, on host, resize the cores
  for (int n = 0; n < GetNumCores(); n++) {
    cores_host_(n).ResizeToNewShape();
  }
}

// This is Algorithm 5 in Al Daas et al. Note the sqrt(sigma) used in both L/R
// core updates.
// Updating bond space n
KOKKOS_INLINE_FUNCTION
void TensorTrain::UpdateCoreIndexSpaces(const int n, const int Rn_new,
                                        parthenon::team_mbr_t tm, GramSVDStorage &GS) {

  // pull out cores object
  auto cores = this->cores_device_;
  // const int Ncores = cores.GetNumCores();
  const int Ncores = cores.size();
  const int Rnm1 = static_cast<int>(cores(n).GetLeftRank());
  const int Rn = static_cast<int>(cores(n).GetRightRank());
  const int PIn = static_cast<int>(cores(n).GetPhysicalIndexSize());

  // if (n > 0) return; // only update first bond space

  printf("In UpdateCoreIndexSpaces:\n");
  printf("LamL, sigma:\n");
  for (int i = 0; i < Rn; i++) {
    printf("%23.15e   %23.15e\n", GS.EvalL()(i), GS.SVDS()(i));
  }

  printf("EVL:\n");
  for (int i = 0; i < Rn; ++i) {
    for (int j = 0; j < Rn; ++j) {
      printf("  %12.5e", GS.EVL()(i, j));
    }
    printf("\n");
  }

  printf("core n:\n");
  for (int i = 0; i < PIn; ++i) {
    for (int j = 0; j < Rn; ++j) {
      printf("  %12.5e", cores(n)(0, i, j));
    }
    printf("\n");
  }

  printf("coreTMP n:\n");
  for (int i = 0; i < PIn; ++i) {
    for (int j = 0; j < Rn; ++j) {
      printf("  %12.5e", GS.CTmp()(0, i, j));
    }
    printf("\n");
  }

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
                  printf("core(%d)(%d, %d, %d) = %23.15e \n", n, alf, i, gam, accum);
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
                  // printf("bond %d, sigma = %12.5e\n", n,
                  // GS.SVDS()(GS.ModeMap()(gam)));
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
    printf("last core:\n");
    par_for_inner(tm, 0, Rn_new - 1, 0, Rnp1 - 1, 0, PInp1 - 1,
                  [&](const int iL, const int iR, const int ic) {
                    cores(n + 1)(iL, ic, iR) = GS.CTmp()(iL, ic, iR);
                    printf("%12.5e\n", cores(n + 1)(iL, ic, iR));
                  });
  }
  tm.team_barrier();

} // TensorTrain::UpdateCoreIndexSpaces

} // namespace tensors
} // namespace parthenon
