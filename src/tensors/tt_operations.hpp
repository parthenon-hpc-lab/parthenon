//========================================================================================
// (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
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

#ifndef TENSORS_TT_OPERATIONS_HPP
#define TENSORS_TT_OPERATIONS_HPP

#include "basic_types.hpp"
#include "kokkos_abstraction.hpp"
#include "linear_algebra/symmetric_evd.hpp"
#include "linear_algebra/square_svd.hpp"
#include "tt_traits.hpp"
#include "tt_types.hpp"


namespace parthenon {
namespace tensor2 {
namespace impl {

// Copy all fibers from a source core into a destination core with optional
// offsets in left-rank and right-rank space. This is mainly used to assemble
// block-structured TT operations such as non-destructive sum.
template <class TTraits>
KOKKOS_INLINE_FUNCTION
void CopyCoreBlock(parthenon::team_mbr_t member,
                   const TensorCoreDeviceT<TTraits> &src,
                   TensorCoreDeviceT<TTraits> &dst,
                   int loffset = 0, int roffset = 0) {
  for (int l = 0; l < src.LR(); ++l) {
    for (int r = 0; r < src.RR(); ++r) {
      auto const *const fs = &src(l, 0, r);
      auto *fd = &dst(l + loffset, 0, r + roffset);
      parthenon::par_for_inner(member, 0, src.DD() - 1,
                               [&](const int j) { fd[j] = fs[j]; });
    }
  }
}

// Set a rectangular block in rank space to a constant value. This is used
// primarily to zero off-diagonal blocks created by TT addition.
template <class TTraits>
KOKKOS_INLINE_FUNCTION
void SetCoreBlock(parthenon::team_mbr_t member,
                  TensorCoreDeviceT<TTraits> &dst,
                  typename TTraits::real_t value,
                  std::pair<int, int> lrange,
                  std::pair<int, int> rrange) {
  for (int l = lrange.first; l < lrange.second; ++l) {
    for (int r = rrange.first; r < rrange.second; ++r) {
      auto *fd = &dst(l, 0, r);
      parthenon::par_for_inner(member, 0, dst.DD() - 1,
                               [&](const int j) { fd[j] = value; });
    }
  }
}

// Compute the Hadamard product of two tensor cores and write the result into
// the destination core. The destination rank space is indexed by the product
// of the input rank spaces.
template <class TTraits>
KOKKOS_INLINE_FUNCTION
void HadamardCoreBlocks(parthenon::team_mbr_t member,
                        const TensorCoreDeviceT<TTraits> &core_a,
                        const TensorCoreDeviceT<TTraits> &core_b,
                        TensorCoreDeviceT<TTraits> &core_c) {
  for (int la = 0; la < core_a.LR(); ++la) {
    for (int lb = 0; lb < core_b.LR(); ++lb) {
      const int lc = la * core_b.LR() + lb;

      for (int ra = 0; ra < core_a.RR(); ++ra) {
        for (int rb = 0; rb < core_b.RR(); ++rb) {
          const int rc = ra * core_b.RR() + rb;

          auto const *const fa = &core_a(la, 0, ra);
          auto const *const fb = &core_b(lb, 0, rb);
          auto *fc = &core_c(lc, 0, rc);

          parthenon::par_for_inner(member, 0, core_c.DD() - 1,
                                   [&](const int j) { fc[j] = fa[j] * fb[j]; });
        }
      }
    }
  }
}

} // namespace impl

// Set every entry in every core of a tensor-train pack to a single value.
template <class TTraits>
void SetTTPackToValue(TensorPackT<TTraits> &pack, typename TTraits::real_t value) {
  constexpr int unused_scratch_size = 0;
  constexpr int unused_scratch_level = 1;
  parthenon::par_for_outer(
      PARTHENON_AUTO_LABEL, unused_scratch_size, unused_scratch_level,
      0, pack.GetNBlocks() - 1, 0, pack.GetNCores() - 1,
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int c) {
        auto &core = pack(b, 0, c);
        impl::SetCoreBlock(member, core, value,  std::pair<int, int>{0, core.LR()},  std::pair<int, int>{0, core.RR()});
      });
}

// Form the non-destructive TT sum of two batches of tensor trains. The output
// trains are allocated on host, packed, and then filled on device by copying
// the two input trains into the appropriate diagonal blocks.
template <class TTraits>
std::vector<TensorTrainT<TTraits>>
NonDestructiveSum(const std::vector<TensorTrainT<TTraits>> &TrainsA,
                  const std::vector<TensorTrainT<TTraits>> &TrainsB) {
  PARTHENON_REQUIRE(TrainsA.size() == TrainsB.size(),
                    "Must be adding the same number of TTs.");

  std::vector<TensorTrainT<TTraits>> TrainsC;
  TrainsC.reserve(TrainsA.size());

  for (int t = 0; t < TrainsA.size(); ++t) {
    const auto &train_A = TrainsA[t];
    const auto &train_B = TrainsB[t];
    std::vector<int> phys_dims, target_ranks;
    PARTHENON_REQUIRE(train_A.NCores() == train_B.NCores(),
                      "Added trains must have the same number of cores.");
    for (int c = 0; c < train_A.NCores(); ++c) {
      PARTHENON_REQUIRE(train_A(c).DD() == train_B(c).DD(),
                        "Must have equivalent physical dims.");
      phys_dims.push_back(train_A(c).DD());
    }
    for (int c = 0; c < train_A.NCores() - 1; ++c) {
      target_ranks.push_back(train_A(c).RR() + train_B(c).RR());
    }
    TrainsC.emplace_back(phys_dims, target_ranks);
  }

  TensorPackT<TTraits> pack_a(TrainsA);
  TensorPackT<TTraits> pack_b(TrainsB);
  TensorPackT<TTraits> pack_c(TrainsC);

  constexpr int unused_scratch_size = 0;
  constexpr int unused_scratch_level = 1;
  parthenon::par_for_outer(
      PARTHENON_AUTO_LABEL, unused_scratch_size, unused_scratch_level,
      0, pack_a.GetNBlocks() - 1, 0, pack_a.GetNCores() - 1,
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int c) {
        auto &core_c = pack_c(b, 0, c);

        auto &core_a = pack_a(b, 0, c);
        impl::CopyCoreBlock(member, core_a, core_c, 0, 0);

        auto &core_b = pack_b(b, 0, c);
        const int loffset = (c > 0) * core_a.LR();
        const int roffset = (c != (pack_a.GetNCores() - 1)) * core_a.RR();
        impl::CopyCoreBlock(member, core_b, core_c, loffset, roffset);

        if (loffset && roffset) {
          impl::SetCoreBlock(member, core_c, typename TTraits::real_t(0),
                             std::pair<int, int>{0, core_a.LR()},
                             std::pair<int, int>{core_a.RR(), core_a.RR() + core_b.RR()});
          impl::SetCoreBlock(member, core_c, typename TTraits::real_t(0),
                             std::pair<int, int>{core_a.LR(), core_a.LR() + core_b.LR()},
                             std::pair<int, int>{0, core_a.RR()});
        }
      });
  return TrainsC;
}

// Form the Hadamard product of two batches of tensor trains. The output ranks
// are the products of the corresponding input ranks, and the core entries are
// filled by pairwise fiber multiplication.
template <class TTraits>
std::vector<TensorTrainT<TTraits>>
HadamardProduct(std::vector<TensorTrainT<TTraits>> &TrainsA,
                std::vector<TensorTrainT<TTraits>> &TrainsB) {
  PARTHENON_REQUIRE(TrainsA.size() == TrainsB.size(),
                    "Must be taking the Hadamard product of the same number of TTs.");

  std::vector<TensorTrainT<TTraits>> TrainsC;
  TrainsC.reserve(TrainsA.size());

  for (int t = 0; t < TrainsA.size(); ++t) {
    const auto &train_A = TrainsA[t];
    const auto &train_B = TrainsB[t];

    PARTHENON_REQUIRE(train_A.NCores() == train_B.NCores(),
                      "Hadamard product requires the same number of cores.");

    std::vector<int> phys_dims, target_ranks;
    for (int c = 0; c < train_A.NCores(); ++c) {
      PARTHENON_REQUIRE(train_A(c).DD() == train_B(c).DD(),
                        "Hadamard product requires matching physical dimensions.");
      phys_dims.push_back(train_A(c).DD());
    }
    for (int c = 0; c < train_A.NCores() - 1; ++c) {
      target_ranks.push_back(train_A(c).RR() * train_B(c).RR());
    }

    TrainsC.emplace_back(phys_dims, target_ranks);
  }

  TensorPackT<TTraits> pack_a(TrainsA);
  TensorPackT<TTraits> pack_b(TrainsB);
  TensorPackT<TTraits> pack_c(TrainsC);

  constexpr int unused_scratch_size = 0;
  constexpr int unused_scratch_level = 1;
  parthenon::par_for_outer(
      PARTHENON_AUTO_LABEL, unused_scratch_size, unused_scratch_level,
      0, pack_a.GetNBlocks() - 1, 0, pack_a.GetNCores() - 1,
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int c) {
        auto &core_a = pack_a(b, 0, c);
        auto &core_b = pack_b(b, 0, c);
        auto &core_c = pack_c(b, 0, c);
        impl::HadamardCoreBlocks(member, core_a, core_b, core_c);
      });

  return TrainsC;
}

template <class Real, class MatA, class MatB, class MatC,
          class Diag1, class Diag2, class Diag3>
KOKKOS_INLINE_FUNCTION
void MatMulDiag3(parthenon::team_mbr_t tm,
                 const Diag1 &D1,
                 const MatA &A,
                 const Diag2 &D2,
                 const MatB &B,
                 const Diag3 &D3,
                 MatC &C) {
  const int m = GetNrows(A);
  const int k = GetNcols(A);
  const int n = GetNcols(B);

  PARTHENON_REQUIRE(GetNrows(B) == k, "MatMulDiag3: incompatible inner dimensions.");
  PARTHENON_REQUIRE(GetNrows(C) == m, "MatMulDiag3: output row dimension mismatch.");
  PARTHENON_REQUIRE(GetNcols(C) == n, "MatMulDiag3: output column dimension mismatch.");

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      Real sum{0};

      parthenon::par_reduce_inner(
          parthenon::inner_loop_pattern_ttr_tag,
          tm, 0, k - 1,
          [&](int p, Real &lsum) {
            lsum += A(i, p) * D2(p) * B(p, j);
          },
          Kokkos::Sum<Real>(sum));

      Kokkos::single(Kokkos::PerTeam(tm), [&]() {
        C(i, j) = D1(i) * sum * D3(j);
      });
    }
  }
  tm.team_barrier();
}

template <class RealVec, class IntVec, class Real>
KOKKOS_INLINE_FUNCTION
void BuildDescendingPermutation(parthenon::team_mbr_t tm,
                                const RealVec &sig,
                                const int rank,
                                const Real eps0,
                                IntVec &perm,
                                int &rank_new) {
  Kokkos::single(Kokkos::PerTeam(tm), [&]() {
    for (int i = 0; i < rank; ++i) perm(i) = i;

    // Selection sort of the permutation by descending singular value.
    for (int i = 0; i < rank - 1; ++i) {
      int best = i;
      for (int j = i + 1; j < rank; ++j) {
        const int pj = perm(j);
        const int pb = perm(best);
        if ((sig(pj) > sig(pb)) ||
            ((sig(pj) == sig(pb)) && (pj < pb))) {
          best = j;
        }
      }
      if (best != i) {
        const int tmp = perm(i);
        perm(i) = perm(best);
        perm(best) = tmp;
      }
    }
  });
  tm.team_barrier();

  // All team members compute the same truncated rank from the sorted map.
  const Real eps02 = eps0 * eps0;
  Real tail2{0};
  rank_new = rank;

  for (int i = rank - 1; i >= 1; --i) {
    const Real s = sig(perm(i));
    const Real next_tail2 = tail2 + s * s;
    if (next_tail2 <= eps02) {
      tail2 = next_tail2;
      rank_new = i;
    } else {
      break;
    }
  }
}

struct no_core_mask {
  KOKKOS_FORCEINLINE_FUNCTION
  static constexpr bool active(int c, int j) {return true;}
};

template <class T>
struct wrap_3D {
  T *scratch;
  int nl, nd, nr;
  KOKKOS_FORCEINLINE_FUNCTION
  int LR() const {return nl;}
  KOKKOS_FORCEINLINE_FUNCTION
  int RR() const {return nr;}
  KOKKOS_FORCEINLINE_FUNCTION
  int DD() const {return nd;}

  KOKKOS_FORCEINLINE_FUNCTION
  T &operator()(int l, int j, int r) const {
    return scratch[nr * nd * l + nd * r + j];
  }
};

template <class TTraits, class F = no_core_mask>
void RoundGramSVD(std::vector<TensorTrainT<TTraits>> &trains,
                  typename TTraits::real_t eps, 
                  F core_mask = no_core_mask{}) {
  using real_t = typename TTraits::real_t;

  // Find the number of cores and maximum rank 
  int max_rank{0};
  int max_core_size{0};
  int n_cores{0};
  for (const auto &train : trains) {
    n_cores = train.NCores();
    for (int c = 0; c < train.NCores(); ++c) {
      max_rank = std::max(max_rank, train(c).RR());
      max_core_size = std::max(max_core_size, train(c).LR() * train(c).DD() * train(c).RR());
    }
  }
  
  int scratch_size{0};
  // Calculate the max storage for Gram matrices
  scratch_size += ScratchPad2D<real_t>::shmem_size(n_cores, max_rank * max_rank);
  scratch_size += ScratchPad1D<real_t>::shmem_size(max_rank * max_rank);

  // Storage for temporary when calculating right Gram matrices
  scratch_size += ScratchPad1D<real_t>::shmem_size(max_core_size);

  // Calculate the total storage for linear algebra scratch
  scratch_size += SymmetricEVD::total_shmem_scratch_size(max_rank); 

  // Calculate storage for eigen and singular value results
  scratch_size += 4 * ScratchPad1D<real_t>::shmem_size(max_rank * max_rank);
  scratch_size += 3 * ScratchPad1D<real_t>::shmem_size(max_rank);
  
  // Singular value permutation array
  scratch_size += ScratchPad1D<int>::shmem_size(max_rank);
  
  // GEMM storage
  const int storage_size = std::max(max_rank, 32) * std::max(max_rank, 32);
  scratch_size += 3 * ScratchPad1D<real_t>::shmem_size(storage_size);
  
  TensorPackT<TTraits> pack(trains);
  
  // Allocate array for storing final ranks to eventually copy back to host to
  // round
  using final_rank_arr_t = typename TTraits::template view_t<int**, ManagedTag>;
  final_rank_arr_t final_rank_arr("Final ranks", pack.GetNBlocks(), n_cores - 1);
  
  constexpr int scratch_level = 1;
  parthenon::par_for_outer(
      PARTHENON_AUTO_LABEL, scratch_size, scratch_level,
      0, pack.GetNBlocks() - 1,
      KOKKOS_LAMBDA(parthenon::team_mbr_t tm, const int b) {
        // Allocate scratch, we allocate flat in the rank dimensions to make 
        // it easier to reuse between cores of different rank size
        auto &tm_scratch = tm.team_scratch(scratch_level);
        // Gram matrices, need to store all right Gram matrices
        ScratchPad2D<real_t> GR(tm_scratch, n_cores, max_rank * max_rank);
        ScratchPad1D<real_t> GL(tm_scratch, max_rank * max_rank);
        ScratchPad1D<real_t> gram_temp_flat(tm_scratch, max_core_size);
        ScratchPad1D<real_t> a_scratch(tm_scratch, storage_size); 
        ScratchPad1D<real_t> b_scratch(tm_scratch, storage_size); 
        ScratchPad1D<real_t> c_scratch(tm_scratch, storage_size); 
        
        // R-to-L sweep over cores
        // Last Gram matrix requires a single reduction
        {
          int c = n_cores - 1;
          auto &core = pack(b, 0, c);
          const int rank = core.LR();
          matrix_wrapper_t<real_t> GR_mat(&GR(c, 0), rank, rank);
          auto Hc = GetHorizontalUnfolding(core);
          auto HcT = GetHorizontalUnfoldingTranspose(core);
          MatMulPacked<32, 32, 16, true>(tm, Hc, HcT, GR_mat,
                                       a_scratch, b_scratch, c_scratch); 
        }
        tm.team_barrier();

        for (int c = n_cores - 2; c >= 0; --c) {
          const auto &core = pack(b, 0, c);
          const auto lr = core.LR();
          const auto rr = core.RR();
          const auto dd = core.DD();
          
          wrap_3D<real_t> gram_temp{gram_temp_flat.data(), lr, dd, rr};
          matrix_wrapper_t<real_t> GR_prev_mat(&GR(c + 1, 0), rr, rr);
          auto gram_temp_vert = GetVerticalUnfolding(gram_temp);
          auto Vc = GetVerticalUnfolding(core);
          MatMulPacked<16, 16, 16>(tm, Vc, GR_prev_mat, gram_temp_vert,
                                a_scratch, b_scratch, c_scratch);
          
      
          matrix_wrapper_t<real_t> GR_mat(&GR(c, 0), lr, lr);
          auto gram_temp_horizT = GetHorizontalUnfoldingTranspose(gram_temp);
          auto Hc = GetHorizontalUnfolding(core);
          MatMulPacked<8, 8, 16, true>(tm, Hc, gram_temp_horizT, GR_mat,
                                       a_scratch, b_scratch, c_scratch);
        }
        
        // Calculate the absolute tolerance
        const real_t eps0 = safe_sqrt(GR(0, 0)) * eps / sqrt(std::max(n_cores, 2) - 1) + 1.e-16;

        // Eigen systems
        ScratchPad1D<real_t> QL(tm_scratch, max_rank * max_rank);
        ScratchPad1D<real_t> eigL(tm_scratch, max_rank);
        ScratchPad1D<real_t> QR(tm_scratch, max_rank * max_rank);
        ScratchPad1D<real_t> eigR(tm_scratch, max_rank);

        // SVD
        ScratchPad1D<real_t> U(tm_scratch, max_rank * max_rank); 
        ScratchPad1D<real_t> V(tm_scratch, max_rank * max_rank); 
        ScratchPad1D<real_t> sig(tm_scratch, max_rank);
        ScratchPad1D<int> perm(tm_scratch, max_rank);

        // Scratch that can be re-used amongst solves
        ScratchPad1D<real_t> real_scratch(tm_scratch, SymmetricEVD::double_scratch_size(max_rank));
        ScratchPad1D<std::size_t> szt_scratch(tm_scratch, SymmetricEVD::sizet_scratch_size(max_rank));

        // L-to-R sweep over bonds 
        for (int c = 0; c < n_cores - 1; ++c) {
          const auto &core = pack(b, 0, c);
          // Make sure we use the left rank that was updated in the previous iteration
          const auto lr = c == 0 ? core.LR() : final_rank_arr(b, c - 1);
          const auto rr = core.RR();
          const auto dd = core.DD();
          const int rank = rr;
          // Compute left Gram matrix
          matrix_wrapper_t<real_t> GL_mat(GL.data(), rank, rank);
          parthenon::par_for_inner(tm, 0, rank - 1, 0, rank - 1, 
                [&](int alpha, int beta){
                  GL_mat(alpha, beta) = 0.0;
              });
          tm.team_barrier();

          auto &temp = sig;
          for (int lambda = 0; lambda < lr; ++lambda) {
            for (int j = 0; j < core.DD() - 1; ++j) {
              parthenon::par_for_inner(tm, 0, rank - 1, [&](int alpha) {
                temp(alpha) = core(lambda, j, alpha);
              });
              tm.team_barrier();
              parthenon::par_for_inner(tm, 0, rank - 1, 0, rank - 1, [&](int alpha, int beta) {
                GL_mat(alpha, beta) += temp(alpha) * temp(beta);
              });
              tm.team_barrier();
            }
          }
          tm.team_barrier();

          // Compute eigen decomposition of L and R Gram matrices
          matrix_wrapper_t<real_t> QL_mat(QL.data(), rank, rank);
          SymmetricEVD::execute(tm, &GL_mat, &QL_mat, eigL.data(),
                                real_scratch.data(), szt_scratch.data());
          tm.team_barrier();

          matrix_wrapper_t<real_t> GR_mat(&GR(c + 1, 0), rank, rank);
          matrix_wrapper_t<real_t> QR_mat(QR.data(), rank, rank);
          SymmetricEVD::execute(tm, &GR_mat, &QR_mat, eigR.data(),
                                real_scratch.data(), szt_scratch.data());
          tm.team_barrier();

          // Compute M = eig_L^{1/2} Q_L^T Q_R eig_R^{1/2} 
          auto &M_mat = GL_mat; // Just reuse GL, since we are done with it
          real_t maxL{0.0};
          real_t maxR{0.0};
          parthenon::par_reduce_inner(parthenon::inner_loop_pattern_ttr_tag,
              tm, 0, rank - 1, [&](int r, real_t &lmax){
                lmax = std::max(lmax, std::abs(eigL[r]));
            }, Kokkos::Max<real_t>(maxL));
          parthenon::par_reduce_inner(parthenon::inner_loop_pattern_ttr_tag,
              tm, 0, rank - 1, [&](int r, real_t &lmax){
                lmax = std::max(lmax, std::abs(eigR[r]));
            }, Kokkos::Max<real_t>(maxR));
          tm.team_barrier();
          parthenon::par_for_inner(tm, 0, rank - 1, 
                [&](int r){
                  eigL[r] = abs(eigL[r]) > (1.e-16 * maxL + 1e-20) ? safe_sqrt(eigL[r]) : 0.0;
                  eigR[r] = abs(eigR[r]) > (1.e-16 * maxR + 1e-20) ? safe_sqrt(eigR[r]) : 0.0;
              });
          tm.team_barrier();
          MatMulDiag3<real_t>(tm, eigL, QL_mat.GetTranspose(), unity_vector_t(), QR_mat, eigR, M_mat);  
          tm.team_barrier();

          // Compute SVD of M
          matrix_wrapper_t<real_t> U_mat(U.data(), rank, rank);
          matrix_wrapper_t<real_t> V_mat(V.data(), rank, rank);
          SquareSVD::execute(tm, &M_mat, &U_mat, &V_mat, sig.data(), real_scratch.data(), szt_scratch.data());
          tm.team_barrier();

          // Truncate SVD to find new rank and store rank 
          int rank_new;
          BuildDescendingPermutation(tm, sig, rank, eps0, perm, rank_new);
          // printf("\n[%i] bond %i with rank_old %i and rank_new %i, eps0 = %e\n  ", b, c, rank, rank_new, eps0);
          // for (int i = 0; i < rank; ++i) {
          //   printf("%e, ", sig[perm[i]]);
          //   if (i % 12 == 11) printf("\n");
          // }
          // printf("\n\n");
          auto Ukeep_mat = U_mat.GetPermutedCols(perm, rank_new);
          auto VTkeep_mat = V_mat.GetTranspose().GetPermutedRows(perm, rank_new);
          auto sigkeep = GetPermuted(sig, perm, rank_new);
          Kokkos::single(Kokkos::PerTeam(tm), [&](){
                    final_rank_arr(b, c) = rank_new;
                });
          // Take the inverse, but filter out zero eigenmodes
          parthenon::par_for_inner(tm, 0, rank - 1, 
                [&](int r){
                  eigL[r] = abs(eigL[r]) > (1.e-16 * maxL + 1e-20) ? 1.0 / eigL[r] : 0.0;
                  eigR[r] = abs(eigR[r]) > (1.e-16 * maxR + 1e-20) ? 1.0 / eigR[r] : 0.0;
              });
          tm.team_barrier();

          // Push SVD U left [V(core_L) = V(core_L) Q_L eig_L^{-1/2} U]
          matrix_wrapper_t<real_t> T_Lmat(GL.data(), rank, rank_new);
          MatMulDiag3<real_t>(tm, unity_vector_t(), QL_mat, eigL, Ukeep_mat, unity_vector_t(), T_Lmat); 
          wrap_3D<real_t> corelp{gram_temp_flat.data(), lr, dd, rank_new};
          auto Vc = GetVerticalUnfolding(core, lr, dd, rr);
          auto Vcorelp = GetVerticalUnfolding(corelp);
          MatMulPacked<16, -1, -1>(tm, Vc, T_Lmat, Vcorelp,
                                a_scratch, b_scratch, c_scratch);
          
          parthenon::par_for_inner(tm, 0, lr - 1, 0, rank_new - 1, 0, dd - 1, 
                [&](int l, int r, int j){
                  core(l, j, r) = corelp(l, j, r);
              });
          tm.team_barrier();

          // Push SVD Sigma V right
          auto &coreR = pack(b, 0, c + 1);
          const int ddR = coreR.DD();
          const int rrR = coreR.RR();
          wrap_3D<real_t> corerp{gram_temp_flat.data(), rank_new, ddR, rrR};
          matrix_wrapper_t<real_t> T_Rmat(GL.data(), rank_new, rank);
          MatMulDiag3<real_t>(tm, sigkeep, VTkeep_mat, eigR, QR_mat.GetTranspose(), unity_vector_t(), T_Rmat); 
          
          auto Hc = GetHorizontalUnfolding(coreR);
          auto Hcorerp = GetHorizontalUnfolding(corerp);
          MatMulPacked<-1, 16, -1>(tm, T_Rmat, Hc, Hcorerp,
                                a_scratch, b_scratch, c_scratch);
          
          parthenon::par_for_inner(tm, 0, rank_new - 1, 0, rrR - 1, 0, ddR - 1,
              [&](int l, int r, int j) {
                coreR(l, j, r) = corerp(l, j, r);
              });
          tm.team_barrier(); 
        }
      });
  
  auto final_rank_arr_h = Kokkos::create_mirror_view(final_rank_arr);
  Kokkos::deep_copy(final_rank_arr_h, final_rank_arr);

  for (int b = 0; b < trains.size(); ++b) {
    auto &train = trains[b];
    const int ncores = train.NCores();

    // Bond c stores the rank between core c and core c+1, so:
    //   core 0:    (1,                r_0)
    //   core c:    (r_{c-1},          r_c)    for 1 <= c <= ncores-2
    //   core last: (r_{ncores-2},     1)
    if (ncores == 1) continue;

    train(0).ReduceSize(1, final_rank_arr_h(b, 0));

    for (int c = 1; c < ncores - 1; ++c) {
      train(c).ReduceSize(final_rank_arr_h(b, c - 1),
                          final_rank_arr_h(b, c));
    }

    train(ncores - 1).ReduceSize(final_rank_arr_h(b, ncores - 2), 1);
  }
}

} // namespace tensor2
} // namespace parthenon

#endif // TENSORS_TT_OPERATIONS_HPP