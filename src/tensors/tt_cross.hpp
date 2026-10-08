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

// This file was made in part with generative AI.

#ifndef TENSORS_TT_CROSS_HPP_
#define TENSORS_TT_CROSS_HPP_

#include <algorithm>
#include <type_traits>
#include <vector>

#include "batched_linear_algebra/execution_utils.hpp"
#include "batched_linear_algebra/maxvol.hpp"
#include "batched_linear_algebra/qr_decomposition.hpp"
#include "kokkos_abstraction.hpp"
#include "tt_pack.hpp"
#include "tt_traits.hpp"
#include "tt_types.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {
namespace tensor {
using namespace batched_linear_algebra; // NOLINT(build/namespaces)

// One full multi-index (i_0, ..., i_{d-1}) of a black-box tensor, as seen by the
// functor passed to TTCross. Entry c is a flat index into [0, DD_c) of core c;
// decoding it into physical dimensions (e.g. through the core's Indexer6D) is up
// to the functor. The entries are not stored contiguously: those left of core k
// come from a stored left tuple, those right of it from a stored right tuple, so
// building a multi-index copies nothing.
struct tt_multi_index {
  const int *lrow;
  const int *rrow;
  int k, j, d;

  KOKKOS_FORCEINLINE_FUNCTION int size() const { return d; }

  KOKKOS_FORCEINLINE_FUNCTION int operator()(int c) const {
    return c < k ? lrow[c] : (c > k ? rrow[c] : j);
  }
};

// The nested multi-index sets of a TT cross approximation of a batch of tensors.
// Holding on to this object between TTCross calls lets a later call (e.g. on a
// slowly changing function) warm-start from the previous sets.
//
// For core k of a train with ranks r_{k-1} x n_k x r_k:
//   - left(b, k, a, c), a < r_{k-1}, c < k, is tuple a of the left set I_k.
//   - right(b, k, a, c), a < r_k, c > k, is tuple a of the right set J_k.
//   - left_piv(b, k, a), k >= 1, is the row of core k-1's vertical unfolding
//     that Maxvol picked to form I_k[a] = (I_{k-1}[l], j).
//   - right_piv(b, k, a), k <= d-2, is the row of core k+1's transposed
//     horizontal unfolding that Maxvol picked to form J_k[a] = (j, J_{k+1}[r]).
// The tuples are stored as full length-d rows so that a tt_multi_index can
// point straight into them; the entries outside a tuple's range are unused.
template <class TTraits>
class TTCrossIndexSetsT {
 public:
  using memory_space = typename TTraits::memory_space;
  using tuples_t = Kokkos::View<int ****, Kokkos::LayoutRight, memory_space>;
  using pivots_t = Kokkos::View<int ***, Kokkos::LayoutRight, memory_space>;

  tuples_t left, right;
  pivots_t left_piv, right_piv;

  // Whether the left/right pivots came from Maxvol (rather than being fresh or
  // the evenly spaced starting right sets), and whether the last call that used
  // these sets stopped because a full round changed no pivots.
  bool left_valid{false};
  bool right_valid{false};
  bool converged{false};

  // Make the sets match the given batch size, physical dimensions and bond
  // ranks. Matching sets are kept as they are; otherwise they are reallocated and
  // marked invalid.
  void Prepare(int nbatch, const std::vector<int> &dims, const std::vector<int> &ranks) {
    if (nbatch == nbatch_ && dims == dims_ && ranks == ranks_) return;
    nbatch_ = nbatch;
    dims_ = dims;
    ranks_ = ranks;
    const int ncores = static_cast<int>(dims.size());
    int max_rank = 1;
    for (const int r : ranks)
      max_rank = std::max(max_rank, r);
    left = tuples_t("TT cross left sets", nbatch, ncores, max_rank, ncores);
    right = tuples_t("TT cross right sets", nbatch, ncores, max_rank, ncores);
    left_piv = pivots_t("TT cross left pivots", nbatch, ncores, max_rank);
    right_piv = pivots_t("TT cross right pivots", nbatch, ncores, max_rank);
    Invalidate();
  }

  void Invalidate() {
    left_valid = false;
    right_valid = false;
    converged = false;
  }

  const std::vector<int> &Ranks() const { return ranks_; }

 private:
  int nbatch_{-1};
  std::vector<int> dims_;
  std::vector<int> ranks_;
};

using TTCrossIndexSets = TTCrossIndexSetsT<DefaultTTraits>;

namespace impl {

// Reduce the requested bond ranks to the largest ranks a nested cross can have:
// r_k <= r_{k-1} n_k and r_k <= n_{k+1} r_{k+1}, with unit boundary ranks. This
// also bounds r_k by both the left and right products of the mode sizes, so every
// unfolding the sweeps factor is at least as tall as it is wide.
inline std::vector<int> ClipCrossRanks(const std::vector<int> &dims,
                                       const std::vector<int> &ranks) {
  const int ncores = static_cast<int>(dims.size());
  PARTHENON_REQUIRE(static_cast<int>(ranks.size()) == ncores - 1,
                    "TTCross needs one rank per bond.");
  std::vector<int> out(ranks);
  for (int k = 0; k < ncores - 1; ++k) {
    PARTHENON_REQUIRE(out[k] >= 1, "TTCross ranks must be at least one.");
    const int rl = (k == 0) ? 1 : out[k - 1];
    out[k] = std::min(out[k], rl * dims[k]);
  }
  for (int k = ncores - 2; k >= 0; --k) {
    const int rr = (k == ncores - 2) ? 1 : out[k + 1];
    out[k] = std::min(out[k], dims[k + 1] * rr);
  }
  return out;
}

} // namespace impl

// Fixed-rank TT cross approximation of variable `var` of every train in a host
// pack (helper for the public TTCross overloads).
template <class TTraits, class F>
int TTCrossVar_(TensorTrainHostPackT<TTraits> &pack_host, const int var, const F &f,
                const std::vector<int> &ranks, const int nsweeps,
                TTCrossIndexSetsT<TTraits> *sets, const double tau) {
  using real_t = typename TTraits::real_t;
  static_assert(std::is_same_v<real_t, double>,
                "TTCross uses the double-precision batched linear algebra.");
  PARTHENON_REQUIRE(nsweeps >= 1, "TTCross needs at least one sweep.");

  const int nbatch = pack_host.NumBlocks();
  const int ncores = pack_host(0, var).NCores();
  PARTHENON_REQUIRE(ncores >= 1, "TTCross trains must already have their cores.");
  std::vector<int> dims(ncores);
  for (int c = 0; c < ncores; ++c)
    dims[c] = pack_host(0, var)(c).DD();

  // Size the output trains. Their previous contents are ignored.
  const std::vector<int> clipped = impl::ClipCrossRanks(dims, ranks);
  for (int b = 0; b < nbatch; ++b)
    pack_host.BuildFreshTrain(b, var, clipped);

  TTCrossIndexSetsT<TTraits> local_sets;
  if (!sets) sets = &local_sets;
  sets->Prepare(nbatch, dims, clipped);

  int max_rank{1};
  int max_rows{1};
  int max_core_size{1};
  for (int c = 0; c < ncores; ++c) {
    const int lr = (c == 0) ? 1 : clipped[c - 1];
    const int rr = (c == ncores - 1) ? 1 : clipped[c];
    max_rank = std::max({max_rank, lr, rr});
    max_rows = std::max({max_rows, lr * dims[c], dims[c] * rr});
    max_core_size = std::max(max_core_size, lr * dims[c] * rr);
  }

  const std::size_t work_size =
      std::max(QRDecomposition::double_scratch_size(max_rows, max_rank),
               Maxvol::double_scratch_size(max_rows, max_rank));
  int scratch_size{0};
  scratch_size += ScratchPad1D<real_t>::shmem_size(max_core_size);
  scratch_size += ScratchPad1D<double>::shmem_size(work_size);
  scratch_size += ScratchPad1D<int>::shmem_size(max_rank);

  TensorPackT<TTraits> pack = pack_host.MakeDevicePackForVar(var);
  auto left = sets->left;
  auto right = sets->right;
  auto left_piv = sets->left_piv;
  auto right_piv = sets->right_piv;
  const bool left_valid = sets->left_valid;
  const bool right_valid = sets->right_valid;
  const bool warm_converged = sets->converged;

  using int_view_t = typename TTraits::template view_t<int *, ManagedTag>;
  int_view_t sweeps_d("TT cross sweeps", nbatch);
  int_view_t converged_d("TT cross converged", nbatch);

  constexpr int scratch_level = 1;
  parthenon::par_for_outer(
      PARTHENON_AUTO_LABEL, scratch_size, scratch_level, 0, nbatch - 1,
      KOKKOS_LAMBDA(parthenon::team_mbr_t tm, const int b) {
        auto &tm_scratch = tm.team_scratch(scratch_level);
        ScratchPad1D<real_t> q_flat(tm_scratch, max_core_size);
        ScratchPad1D<double> work(tm_scratch, work_size);
        ScratchPad1D<int> old_piv(tm_scratch, max_rank);

        // Overwrite core k with the fiber A(I_k, :, J_k) of the black box.
        auto evaluate_fiber = [&](const int k) {
          auto &core = pack(b, 0, k);
          parthenon::par_for_inner(
              tm, 0, core.LR() - 1, 0, core.RR() - 1, 0, core.DD() - 1,
              [&](const int l, const int r, const int j) {
                const tt_multi_index mi{&left(b, k, l, 0), &right(b, k, r, 0), k, j,
                                        ncores};
                core(l, j, r) = f(b, mi);
              });
        };

        // Number of entries of piv[0, n) that differ from old_piv.
        auto count_changes = [&](const int *piv, const int n) {
          int nchanged{0};
          parthenon::par_reduce_inner(
              parthenon::inner_loop_pattern_ttr_tag, tm, 0, n - 1,
              [&](const int a, int &lsum) { lsum += (piv[a] != old_piv(a)); },
              Kokkos::Sum<int>(nchanged));
          return nchanged;
        };

        // Left-to-right step at core k < d-1: select I_{k+1} from the fiber and
        // leave the interpolant B = Q Q(I,:)^{-1} in core k.
        auto left_step = [&](const int k, const bool warm) {
          auto &core = pack(b, 0, k);
          const int lr = core.LR();
          const int dd = core.DD();
          const int rr = core.RR();
          int *piv = &left_piv(b, k + 1, 0);
          evaluate_fiber(k);
          parallel_loop(tm, 0, rr - 1, [&](const int a) { old_piv(a) = piv[a]; });
          barrier(tm);

          ScratchCore<TTraits> q_core{lr, dd, rr, q_flat.data()};
          auto V = TTraits::GetVerticalUnfolding(core);
          auto VQ = TTraits::GetVerticalUnfolding(q_core);
          QRDecomposition::execute(tm, &V, &VQ, work.data());
          barrier(tm);
          Maxvol::execute(tm, VQ, &V, piv, work.data(), !warm, tau);
          barrier(tm);

          const int nchanged = count_changes(piv, rr);
          parallel_loop(tm, 0, rr - 1, 0, k, [&](const int a, const int c) {
            int l, j;
            V.RowIndices(piv[a], l, j);
            left(b, k + 1, a, c) = (c < k) ? left(b, k, l, c) : j;
          });
          barrier(tm);
          return nchanged;
        };

        // Right-to-left step at core k > 0: select J_{k-1} from the fiber and
        // leave the interpolant in core k.
        auto right_step = [&](const int k, const bool warm) {
          auto &core = pack(b, 0, k);
          const int lr = core.LR();
          const int dd = core.DD();
          const int rr = core.RR();
          int *piv = &right_piv(b, k - 1, 0);
          evaluate_fiber(k);
          parallel_loop(tm, 0, lr - 1, [&](const int a) { old_piv(a) = piv[a]; });
          barrier(tm);

          ScratchCore<TTraits> q_core{lr, dd, rr, q_flat.data()};
          auto H = TTraits::GetHorizontalUnfoldingTranspose(core);
          auto HQ = TTraits::GetHorizontalUnfoldingTranspose(q_core);
          QRDecomposition::execute(tm, &H, &HQ, work.data());
          barrier(tm);
          Maxvol::execute(tm, HQ, &H, piv, work.data(), !warm, tau);
          barrier(tm);

          const int nchanged = count_changes(piv, lr);
          parallel_loop(tm, 0, lr - 1, k, ncores - 1, [&](const int a, const int c) {
            int j, r;
            H.RowIndices(piv[a], j, r);
            right(b, k - 1, a, c) = (c > k) ? right(b, k, r, c) : j;
          });
          barrier(tm);
          return nchanged;
        };

        // A bond is warm-started only if its pivots came from Maxvol and the set
        // they extend did not change earlier in the same pass.
        auto left_pass = [&](const bool pivots_valid) {
          int total{0};
          int upstream{0};
          for (int k = 0; k < ncores - 1; ++k) {
            upstream = left_step(k, pivots_valid && upstream == 0);
            total += upstream;
          }
          evaluate_fiber(ncores - 1);
          barrier(tm);
          return total;
        };

        auto right_pass = [&](const bool pivots_valid) {
          int total{0};
          int upstream{0};
          for (int k = ncores - 1; k > 0; --k) {
            upstream = right_step(k, pivots_valid && upstream == 0);
            total += upstream;
          }
          return total;
        };

        // Evenly spaced, nested starting right sets.
        if (!right_valid) {
          for (int k = ncores - 2; k >= 0; --k) {
            const auto &next = pack(b, 0, k + 1);
            const int rk = next.LR();
            const int nrows = next.DD() * next.RR();
            auto H = TTraits::GetHorizontalUnfoldingTranspose(next);
            parallel_loop(tm, 0, rk - 1,
                          [&](const int a) { right_piv(b, k, a) = (a * nrows) / rk; });
            parallel_loop(tm, 0, rk - 1, k + 1, ncores - 1,
                          [&](const int a, const int c) {
                            int j, r;
                            H.RowIndices((a * nrows) / rk, j, r);
                            right(b, k, a, c) = (c > k + 1) ? right(b, k + 1, r, c) : j;
                          });
            barrier(tm);
          }
        }

        // A converged set of pivots that the first pass reproduces is a fixed
        // point of the sweeps, so there is nothing left to do.
        int sweeps = 1;
        bool done = (left_pass(left_valid) == 0) && warm_converged;
        bool right_pivots_valid = right_valid;
        while (!done && sweeps < nsweeps) {
          const int nright = right_pass(right_pivots_valid);
          right_pivots_valid = true;
          const int nleft = left_pass(true);
          ++sweeps;
          done = (nright == 0) && (nleft == 0);
        }

        once_per_team(tm, [&]() {
          sweeps_d(b) = sweeps;
          converged_d(b) = done;
        });
      });

  auto sweeps_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), sweeps_d);
  auto converged_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), converged_d);
  int max_sweeps{0};
  bool all_converged{true};
  for (int b = 0; b < nbatch; ++b) {
    max_sweeps = std::max(max_sweeps, sweeps_h(b));
    all_converged = all_converged && converged_h(b);
  }
  sets->left_valid = true;
  sets->right_valid = right_valid || (ncores > 1 && nsweeps > 1);
  sets->converged = all_converged;
  return max_sweeps;
}

// Fixed-rank TT cross (interpolation) approximation of a batch of black-box
// tensors, written into variable `var` of the trains in a host pack.
//
// Batch entry b of the pack approximates the tensor A_b(i_0, ..., i_{d-1}) =
// f(b, mi), where mi is a tt_multi_index and mi(c) is a flat index into
// [0, DD_c) of core c. f must be callable on device. The batch index b has no
// meaning to this routine beyond selecting the tensor.
//
// The algorithm is the one-site alternating cross of Oseledets and
// Tyrtyshnikov. Each core holds a fiber A(I_k, :, J_k) for nested left and right
// index sets. Going left to right, the fiber of core k is factored by QR, Maxvol
// on Q picks I_{k+1} from I_k x [n_k], and core k becomes Q Q(I,:)^{-1}. Going
// right to left does the same on the transposed horizontal unfolding to pick
// J_{k-1} from [n_k] x J_k. Every call ends on a left-to-right pass, so the
// result interpolates A on the final index sets.
//
// On entry:
//   - Each train must already have its cores (they provide the mode sizes); the
//     values and ranks are ignored and overwritten.
//   - ranks holds one target rank per bond. They are reduced to the largest
//     ranks a nested cross can have (see impl::ClipCrossRanks).
//   - sets may be null. If non-null, matching sets from a previous call are
//     used as a warm start, and the sets are left for the next call.
//
// Notes:
//   - The ranks are fixed. No error estimate is made.
//   - A sweep is a right-to-left pass followed by a left-to-right pass. Each
//     batch entry stops once a sweep changes no pivots.
//   - tau is passed to Maxvol and should be greater than 1.
//
// Returns:
//   - The largest number of left-to-right passes over the batch. A value equal
//     to nsweeps means the index sets may still be changing.
template <class TTraits, class F>
int TTCross(TensorTrainHostPackT<TTraits> &pack_host, const int var, const F &f,
            const std::vector<int> &ranks, const int nsweeps = 10,
            TTCrossIndexSetsT<TTraits> *sets = nullptr, const double tau = 1.05) {
  return TTCrossVar_(pack_host, var, f, ranks, nsweeps, sets, tau);
}

// Convenience overload for a batch held in a plain vector (e.g. unit tests).
template <class TTraits, class F>
int TTCross(std::vector<TensorTrainT<TTraits>> &trains, const F &f,
            const std::vector<int> &ranks, const int nsweeps = 10,
            TTCrossIndexSetsT<TTraits> *sets = nullptr, const double tau = 1.05) {
  auto pack_host = TensorTrainHostPackT<TTraits>::FromVector(trains);
  return TTCrossVar_(pack_host, 0, f, ranks, nsweeps, sets, tau);
}

} // namespace tensor
} // namespace parthenon

#endif // TENSORS_TT_CROSS_HPP_
