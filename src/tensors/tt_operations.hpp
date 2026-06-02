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
/*
template <class TTraits>
std::vector<TensorTrainT<TTraits>>
RoundGramSVD(std::vector<TensorTrainT<TTraits>> &trains,
             typename TTraits::real_t eps) {
  using real_t = typename TTraits::real_t;

  // Find the number of cores and maximum rank 
  int max_rank{0};
  int max_phys_dim{0};
  int n_cores{0};
  for (const auto &train : trains) {
    n_cores = train.Ncores();
    for (int c = 0; c < Ncores(); ++c)
      std::max(max_rank, train(c).RR());
    for (int c = 1; c < Ncores(); ++c)
      std::max(max_phys_dim, train(c).DD());
  }
  
  int scratch_size{0};
  // Calculate the max storage for Gram matrices
  scratch_size += ScratchPad2D<real_t>::shmen_size(n_cores, max_rank * max_rank);
  scratch_size += ScratchPad1D<real_t>::shmen_size(max_rank * max_rank);

  // Storage for temporary when calculating right Gram matrices
  scratch_size += ScratchPad3D<real_t>::shmen_size(max_rank, max_phys_dim, max_rank);

  // Calculate the total storage for linear algebra scratch
  scratch_size += SymmetricEVD::total_shmem_scratch_size(max_rank); 

  // Calculate storage for eigen and singular value results
  scratch_size += 4 * ScratchPad1D<real_t>::shmen_size(max_rank * max_rank);
  scratch_size += 3 * ScratchPad1D<real_t>::shmen_size(max_rank);
  
  // Allocate array for storing final ranks to eventually copy back to host to
  // round
  using final_rank_arr_t = typename TTraits::template view_t<int**, ManagedTag>;
  final_rank_arr_t final_rank_arr("Final ranks", pack.GetNBlocks(), n_cores - 1);
  
  TensorPackT<TTraits> pack(trains);

  constexpr int scratch_level = 1;
  parthenon::par_for_outer(
      PARTHENON_AUTO_LABEL, scratch_size, scratch_level,
      0, pack_a.GetNBlocks() - 1,
      KOKKOS_LAMBDA(parthenon::team_mbr_t tm, const int b) {
        // Allocate scratch, we allocate flat in the rank dimensions to make 
        // it easier to reuse between cores of different rank size
        auto &tm_scratch = tm.team_scratch(scratch_level);
        // Gram matrices, need to store all right Gram matrices
        ScratchPad2D<real_t> GR(tm_scratch, n_cores - 1, max_rank * max_rank);
        ScratchPad1D<real_t> GL(tm_scratch, max_rank * max_rank);
        ScratchPad3D<real_t> gram_temp(tm_scratch, max_rank, max_phys_dim, max_rank);
        
        // R-to-L sweep over cores
        // Last Gram matrix requires a single reduction
        {
          int c = n_cores - 1;
          auto &core = pack(b, 0, c);
          int rank = core.LR();
          matrix_wrap_t<real_t> GR_mat(&GR(c, 0), rank, rank);
          for (int alpha = 0; alpha < rank; ++alpha) {
            for (int beta = alpha; beta < rank; ++beta) {
              real_t const * const dat = &core(alpha, 0, beta);
              real_t sum{0.0};
              parthenon::par_reduce_inner(parthenon::inner_loop_pattern_ttr_tag, tm, 0, core.DD() - 1, [&](int d, real_t &lsum){
                lsum += dat * dat;
              }, Kokkos::Sum<real_t>(sum));
              Kokkos::single(Kokkos::PerTeam(tm), [&](){
                  GR_mat(alpha, beta) = sum;
                  GR_mat(beta, alpha) = sum;
                });
            }
          }
        }
        tm.team_barrier();

        for (int c = n_cores - 2; c > 0; --c) {
          const auto &core = pack(b, 0, c);
          const auto lr = core.LR();
          const auto rr = core.RR();
          const auto dd = core.RR();

          // Zero the temporary storage for the core contracted with 
          // neighboring gram matrix
          parthenon::par_for_inner(tm, 0, lr - 1, 0, rr - 1, 0, dd - 1, 
                [&](int l, int r, int j){
                  gram_temp(l, j, r) = 0.0;
              });
          tm.team_barrier();

          matrix_wrap_t<real_t> GR_prev_mat(&GR(c + 1, 0), rr, rr);
          for (int rp = 0; rp < rr; ++rp) {
            parthenon::par_for_inner(tm, 0, lr - 1, 0, rr - 1, 0, dd - 1, 
                [&](int l, int r, int j){
                  gram_temp(l, j, r) += core(l, j, rp) * GR_prev_mat(rp, r);
              }); 
            tm.team_barrier();
          }

          // Compute and store right Gram matrix
          matrix_wrap_t<real_t> GR_mat(&GR(c, 0), lr, lr);
          for (int alpha = 0; alpha < lr; ++alpha) {
            for (int beta = alpha; beta < lr; ++beta) {
              real_t sum{0.0};
              parthenon::par_reduce_inner(parthenon::inner_loop_pattern_ttr_tag,
                  tm, 0, rr - 1, 0, core.DD() - 1,
                  [&](int lambda, int d, real_t &lsum){
                    lsum += core(alpha, j, lambda) * gram_temp(beta, j, lambda);
                }, Kokkos::Sum<real_t>(sum));
              Kokkos::single(Kokkos::PerTeam(tm), [&](){
                    GR_mat(alpha, beta) = sum;
                    GR_mat(beta, alpha) = sum;
                });
            }
          }
          tm.team_barrier();
        }
        
        // Eigen systems
        ScratchPad1D<real_t> QL(tm_scratch, max_rank * max_rank);
        ScratchPad1D<real_t> eigL(tm_scratch, max_rank);
        ScratchPad1D<real_t> QR(tm_scratch, max_rank * max_rank);
        ScratchPad1D<real_t> eigR(tm_scratch, max_rank);

        // SVD
        ScratchPad1D<real_t> U(tm_scratch, max_rank * max_rank); 
        ScratchPad1D<real_t> V(tm_scratch, max_rank * max_rank); 
        ScratchPad1D<real_t> sig(tm_scratch, max_rank);

        // Scratch that can be re-used amongst solves
        ScratchPad1D<real_t> real_scratch(tm_scratch, real_scratch_size_max);
        ScratchPad1D<std::size_t> szt_scratch(tm_scratch, szt_scratch_size_max);
        
        // L-to-R sweep over bonds 
        for (int c = 0; c < n_cores - 1; ++c) {
          const auto &core = pack(b, 0, c);
          const auto lr = core.LR();
          const auto rr = core.RR();
          const auto dd = core.DD();

          // Compute left Gram matrix
          matrix_wrap_t<real_t> GL_mat(GL.data(), rank, rank);
          for (int alpha = 0; alpha < lr; ++alpha) {
            for (int beta = alpha; beta < lr; ++beta) {
              real_t sum{0.0};
              parthenon::par_reduce_inner(parthenon::inner_loop_pattern_ttr_tag,
                  tm, 0, lr - 1, 0, core.DD() - 1,
                  [&](int lambda, int d, real_t &lsum){
                    lsum += core(lambda, j, alpha) * core(lambda, j, beta);
                }, Kokkos::Sum<real_t>(sum));
              Kokkos::single(Kokkos::PerTeam(tm), [&](){
                    GL_mat(alpha, beta) = sum;
                    GL_mat(beta, alpha) = sum;
                });
            }
          } 
          tm.team_barrier();

          // Compute eigen decomposition of L and R Gram matrices
          matrix_wrap_t<real_t> QL_mat(QL.data(), rank, rank);
          SymmetricEVD::execute(tm, &GL_mat, &QL_mat, eigL.data(),
                                real_scratch.data(), szt_scratch.data());
          tm.team_barrier();

          matrix_wrap_t<real_t> GR_mat(&GR(c, 0), rank, rank);
          matrix_wrap_t<real_t> QR_mat(QR.data(), rank, rank);
          SymmetricEVD::execute(tm, &GR_mat, &QR_mat, eigR.data(),
                                real_scratch.data(), szt_scratch.data());
          tm.team_barrier();

          // Compute M = Σ_L Q_L^T Q_R Σ_R 
   
          // Compute truncated SVD of M
                    
          // Store rank 

          // Push SVD U left

          // Push SVD Sigma V right
        }
      });

  return TrainsC;
}
*/
} // namespace tensor2
} // namespace parthenon

#endif // TENSORS_TT_OPERATIONS_HPP