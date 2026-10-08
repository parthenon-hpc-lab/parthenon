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

// Run each batched_linear_algebra decomposition on a batch of matrices inside a
// device kernel and compare against the serial host result. The numerical tests
// in the other linalg unit test files only run on host; these exist to make sure
// the device code paths compile and are free of races. Each decomposition is
// tested in both supported kernel styles:
//   - hierarchical: one team per matrix in par_for_outer, with the matrices and
//     workspace in team scratch and the team handle passed as tm
//   - flat: one thread per matrix in par_for, with the matrices and workspace
//     in global device arrays and serial_tm_t() passed as tm
// QR is also tested in a flat kernel with compile-time sizes and per-thread
// local arrays, which relies on the scratch sizes being constexpr.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <type_traits>
#include <vector>

#include <catch2/catch.hpp>

#include "batched_linear_algebra/matrix_cross.hpp"
#include "batched_linear_algebra/maxvol.hpp"
#include "batched_linear_algebra/qr_decomposition.hpp"
#include "batched_linear_algebra/qr_solve.hpp"
#include "batched_linear_algebra/square_svd.hpp"
#include "batched_linear_algebra/symmetric_evd.hpp"
#include "kokkos_abstraction.hpp"
#include "linalg_test_utils.hpp"

using namespace parthenon::batched_linear_algebra; // NOLINT(build/namespaces)
using parthenon::ParArray2D;
using parthenon::ParArray3D;
using parthenon::ScratchPad1D;
using parthenon::ScratchPad2D;
using parthenon::team_mbr_t;

namespace linalg_device_test {

constexpr int nbatch = 8;
using View3D = ParArray3D<double>::base_t;

// Copy a batch of equally sized host matrices into a device array.
ParArray3D<double> ToDevice(const std::vector<Matrix> &mats) {
  const int m = mats[0].nrows();
  const int n = mats[0].ncols();
  ParArray3D<double> dev("batch", mats.size(), m, n);
  auto host = Kokkos::create_mirror_view(dev);
  for (std::size_t b = 0; b < mats.size(); ++b) {
    for (int r = 0; r < m; ++r) {
      for (int c = 0; c < n; ++c) {
        host(b, r, c) = mats[b](r, c);
      }
    }
  }
  Kokkos::deep_copy(dev, host);
  return dev;
}

// Copy a device batch back to host as a vector of matrices.
std::vector<Matrix> ToHost(const ParArray3D<double> &dev) {
  auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dev);
  std::vector<Matrix> mats;
  for (int b = 0; b < dev.extent_int(0); ++b) {
    Matrix M(dev.extent_int(1), dev.extent_int(2));
    for (int r = 0; r < M.nrows(); ++r) {
      for (int c = 0; c < M.ncols(); ++c) {
        M(r, c) = host(b, r, c);
      }
    }
    mats.push_back(M);
  }
  return mats;
}

std::vector<std::vector<double>> ToHost(const ParArray2D<double> &dev) {
  auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dev);
  std::vector<std::vector<double>> vals(dev.extent_int(0));
  for (int b = 0; b < dev.extent_int(0); ++b) {
    for (int i = 0; i < dev.extent_int(1); ++i) {
      vals[b].push_back(host(b, i));
    }
  }
  return vals;
}

std::vector<Matrix> RandomBatch(int m, int n, unsigned seed) {
  std::vector<Matrix> mats;
  for (int b = 0; b < nbatch; ++b) {
    mats.push_back(Matrix::RandomGaussian(m, n, seed + b));
  }
  return mats;
}

std::vector<Matrix> RandomSymmetricBatch(int n, unsigned seed) {
  std::vector<Matrix> mats;
  for (const Matrix &G : RandomBatch(n, n, seed)) {
    Matrix S(n, n);
    for (int r = 0; r < n; ++r) {
      for (int c = 0; c < n; ++c) {
        S(r, c) = G(r, c) + G(c, r);
      }
    }
    mats.push_back(S);
  }
  return mats;
}

double MaxAbsDiff(const Matrix &A, const Matrix &B) {
  double mx = 0.0;
  for (int r = 0; r < A.nrows(); ++r) {
    for (int c = 0; c < A.ncols(); ++c) {
      mx = std::max(mx, std::abs(A(r, c) - B(r, c)));
    }
  }
  return mx;
}

double MaxAbsDiff(std::vector<double> a, std::vector<double> b) {
  std::sort(a.begin(), a.end());
  std::sort(b.begin(), b.end());
  double mx = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    mx = std::max(mx, std::abs(a[i] - b[i]));
  }
  return mx;
}

// max |(Aᵀ A - I)_ij| for the columns of A
double ColumnOrthoError(const Matrix &A) {
  double mx = 0.0;
  for (int i = 0; i < A.ncols(); ++i) {
    for (int j = 0; j < A.ncols(); ++j) {
      double s = 0.0;
      for (int r = 0; r < A.nrows(); ++r) {
        s += A(r, i) * A(r, j);
      }
      mx = std::max(mx, std::abs(s - (i == j)));
    }
  }
  return mx;
}

// max |A V - U diag(s)|, i.e. the residual of the pairs (s_i, u_i, v_i). With
// U = V this is the eigenpair residual A Q - Q diag(eigs).
double PairResidual(const Matrix &A, const Matrix &U, const Matrix &V,
                    const std::vector<double> &s) {
  double mx = 0.0;
  for (int i = 0; i < V.ncols(); ++i) {
    for (int r = 0; r < A.nrows(); ++r) {
      double av = 0.0;
      for (int c = 0; c < A.ncols(); ++c) {
        av += A(r, c) * V(c, i);
      }
      mx = std::max(mx, std::abs(av - s[i] * U(r, i)));
    }
  }
  return mx;
}

// ---------------------------------------------------------------------------
// Kernels. Each overwrites A_dev in place and fills the output arrays.
// ---------------------------------------------------------------------------

template <class Decomposition>
void FactorTeam(ParArray3D<double> A_dev, ParArray3D<double> Q_dev) {
  const int m = A_dev.extent_int(1);
  const int n = A_dev.extent_int(2);
  const int qm = Q_dev.extent_int(1);
  const int qn = Q_dev.extent_int(2);
  // team_scratch takes the level by reference on CUDA, so it must be a local
  // captured by the lambda rather than a namespace-scope constant
  const int scratch_level = 0;
  const std::size_t nwork = Decomposition::double_scratch_size(m, n);
  const std::size_t scratch_bytes = Decomposition::total_shmem_scratch_size(m, n) +
                                    ScratchPad2D<double>::shmem_size(m, n) +
                                    ScratchPad2D<double>::shmem_size(qm, qn);
  parthenon::par_for_outer(
      "FactorTeam", scratch_bytes, scratch_level, 0, nbatch - 1,
      KOKKOS_LAMBDA(team_mbr_t member, const int b) {
        ScratchPad1D<double> work(member.team_scratch(scratch_level), nwork);
        ScratchPad2D<double> A(member.team_scratch(scratch_level), m, n);
        ScratchPad2D<double> Q(member.team_scratch(scratch_level), qm, qn);
        parthenon::par_for_inner(
            member, 0, m - 1, 0, n - 1,
            [&](const int r, const int c) { A(r, c) = A_dev(b, r, c); });
        member.team_barrier();

        Decomposition::execute(member, &A, &Q, work.data());
        member.team_barrier();

        parthenon::par_for_inner(
            member, 0, m - 1, 0, n - 1,
            [&](const int r, const int c) { A_dev(b, r, c) = A(r, c); });
        parthenon::par_for_inner(
            member, 0, qm - 1, 0, qn - 1,
            [&](const int r, const int c) { Q_dev(b, r, c) = Q(r, c); });
      });
}

template <class Decomposition>
void FactorFlat(ParArray3D<double> A_dev, ParArray3D<double> Q_dev) {
  const int m = A_dev.extent_int(1);
  const int n = A_dev.extent_int(2);
  ParArray2D<double> work("work", nbatch, Decomposition::double_scratch_size(m, n));
  // The ParArray subview overload is host only, so slice the underlying views
  const View3D A_view = A_dev;
  const View3D Q_view = Q_dev;
  parthenon::par_for(
      "FactorFlat", 0, nbatch - 1, KOKKOS_LAMBDA(const int b) {
        auto A = Kokkos::subview(A_view, b, Kokkos::ALL(), Kokkos::ALL());
        auto Q = Kokkos::subview(Q_view, b, Kokkos::ALL(), Kokkos::ALL());
        Decomposition::execute(serial_tm_t(), &A, &Q, &work(b, 0));
      });
}

// Flat kernel with compile-time sizes, keeping the matrices and workspace in
// per-thread local arrays (the pattern shown in the docs).
template <class Decomposition, int m, int n>
void FactorFlatLocal(ParArray3D<double> A_dev, ParArray3D<double> Q_dev) {
  parthenon::par_for(
      "FactorFlatLocal", 0, nbatch - 1, KOKKOS_LAMBDA(const int b) {
        constexpr std::size_t kWorkSize = Decomposition::double_scratch_size(m, n);
        double a_data[m * n], q_data[m * n], work[kWorkSize];
        matrix_wrapper_t<double> A(a_data, m, n);
        matrix_wrapper_t<double> Q(q_data, m, n);
        for (int r = 0; r < m; ++r) {
          for (int c = 0; c < n; ++c) {
            A(r, c) = A_dev(b, r, c);
          }
        }

        Decomposition::execute(serial_tm_t(), &A, &Q, work);

        for (int r = 0; r < m; ++r) {
          for (int c = 0; c < n; ++c) {
            A_dev(b, r, c) = A(r, c);
            Q_dev(b, r, c) = Q(r, c);
          }
        }
      });
}

// The number of right-hand sides is the number of columns of B for QRSolve
// (A X = B) and the number of rows of B for QRSolveRight (X A = B).
template <class Solver>
int NumRhs(const ParArray3D<double> &B_dev) {
  return std::is_same_v<Solver, QRSolveRight> ? B_dev.extent_int(1) : B_dev.extent_int(2);
}

template <class Solver>
void SolveTeam(ParArray3D<double> A_dev, ParArray3D<double> B_dev) {
  const int am = A_dev.extent_int(1);
  const int an = A_dev.extent_int(2);
  const int bm = B_dev.extent_int(1);
  const int bn = B_dev.extent_int(2);
  const int nrhs = NumRhs<Solver>(B_dev);
  const int scratch_level = 0;
  const std::size_t nwork = Solver::double_scratch_size(am, an, nrhs);
  const std::size_t scratch_bytes = Solver::total_shmem_scratch_size(am, an, nrhs) +
                                    ScratchPad2D<double>::shmem_size(am, an) +
                                    ScratchPad2D<double>::shmem_size(bm, bn);
  parthenon::par_for_outer(
      "SolveTeam", scratch_bytes, scratch_level, 0, nbatch - 1,
      KOKKOS_LAMBDA(team_mbr_t member, const int b) {
        ScratchPad1D<double> work(member.team_scratch(scratch_level), nwork);
        ScratchPad2D<double> A(member.team_scratch(scratch_level), am, an);
        ScratchPad2D<double> B(member.team_scratch(scratch_level), bm, bn);
        parthenon::par_for_inner(
            member, 0, am - 1, 0, an - 1,
            [&](const int r, const int c) { A(r, c) = A_dev(b, r, c); });
        parthenon::par_for_inner(
            member, 0, bm - 1, 0, bn - 1,
            [&](const int r, const int c) { B(r, c) = B_dev(b, r, c); });
        member.team_barrier();

        Solver::execute(member, &A, &B, work.data());
        member.team_barrier();

        parthenon::par_for_inner(
            member, 0, bm - 1, 0, bn - 1,
            [&](const int r, const int c) { B_dev(b, r, c) = B(r, c); });
      });
}

template <class Solver>
void SolveFlat(ParArray3D<double> A_dev, ParArray3D<double> B_dev) {
  const int am = A_dev.extent_int(1);
  const int an = A_dev.extent_int(2);
  const int nrhs = NumRhs<Solver>(B_dev);
  ParArray2D<double> work("work", nbatch, Solver::double_scratch_size(am, an, nrhs));
  const View3D A_view = A_dev;
  const View3D B_view = B_dev;
  parthenon::par_for(
      "SolveFlat", 0, nbatch - 1, KOKKOS_LAMBDA(const int b) {
        auto A = Kokkos::subview(A_view, b, Kokkos::ALL(), Kokkos::ALL());
        auto B = Kokkos::subview(B_view, b, Kokkos::ALL(), Kokkos::ALL());
        Solver::execute(serial_tm_t(), &A, &B, &work(b, 0));
      });
}

constexpr double kMaxvolTau = 1.05;

void MaxvolTeam(ParArray3D<double> A_dev, ParArray3D<double> B_dev,
                ParArray2D<int> I_dev) {
  const int n = A_dev.extent_int(1);
  const int r = A_dev.extent_int(2);
  const int scratch_level = 0;
  const std::size_t nwork = Maxvol::double_scratch_size(n, r);
  const std::size_t scratch_bytes = Maxvol::total_shmem_scratch_size(n, r) +
                                    2 * ScratchPad2D<double>::shmem_size(n, r) +
                                    ScratchPad1D<int>::shmem_size(r);
  const double tau = kMaxvolTau;
  parthenon::par_for_outer(
      "MaxvolTeam", scratch_bytes, scratch_level, 0, nbatch - 1,
      KOKKOS_LAMBDA(team_mbr_t member, const int b) {
        ScratchPad1D<double> work(member.team_scratch(scratch_level), nwork);
        ScratchPad2D<double> A(member.team_scratch(scratch_level), n, r);
        ScratchPad2D<double> B(member.team_scratch(scratch_level), n, r);
        ScratchPad1D<int> I(member.team_scratch(scratch_level), r);
        parthenon::par_for_inner(
            member, 0, n - 1, 0, r - 1,
            [&](const int i, const int c) { A(i, c) = A_dev(b, i, c); });
        member.team_barrier();

        Maxvol::execute(member, A, &B, I.data(), work.data(), true, tau);
        member.team_barrier();

        parthenon::par_for_inner(
            member, 0, n - 1, 0, r - 1,
            [&](const int i, const int c) { B_dev(b, i, c) = B(i, c); });
        parthenon::par_for_inner(member, 0, r - 1,
                                 [&](const int j) { I_dev(b, j) = I(j); });
      });
}

void MaxvolFlat(ParArray3D<double> A_dev, ParArray3D<double> B_dev,
                ParArray2D<int> I_dev) {
  const int n = A_dev.extent_int(1);
  const int r = A_dev.extent_int(2);
  ParArray2D<double> work("work", nbatch, Maxvol::double_scratch_size(n, r));
  const View3D A_view = A_dev;
  const View3D B_view = B_dev;
  const double tau = kMaxvolTau;
  parthenon::par_for(
      "MaxvolFlat", 0, nbatch - 1, KOKKOS_LAMBDA(const int b) {
        auto A = Kokkos::subview(A_view, b, Kokkos::ALL(), Kokkos::ALL());
        auto B = Kokkos::subview(B_view, b, Kokkos::ALL(), Kokkos::ALL());
        Maxvol::execute(serial_tm_t(), A, &B, &I_dev(b, 0), &work(b, 0), true, tau);
      });
}

// Entries 1/(i + j + 1 + shift) computed on demand inside the kernel
struct DeviceHilbert {
  int nrows, ncols;
  double shift;
  KOKKOS_INLINE_FUNCTION double operator()(const int i, const int j) const {
    return 1.0 / (i + j + 1 + shift);
  }
};
KOKKOS_INLINE_FUNCTION int GetNrows(const DeviceHilbert &h) { return h.nrows; }
KOKKOS_INLINE_FUNCTION int GetNcols(const DeviceHilbert &h) { return h.ncols; }

// The matrix for batch entry b, either a slice of a device array or a lazy one
struct StoredSource {
  View3D A;
  KOKKOS_INLINE_FUNCTION auto operator()(const int b) const {
    return Kokkos::subview(A, b, Kokkos::ALL(), Kokkos::ALL());
  }
};
struct HilbertSource {
  int nrows, ncols;
  KOKKOS_INLINE_FUNCTION DeviceHilbert operator()(const int b) const {
    return DeviceHilbert{nrows, ncols, static_cast<double>(b)};
  }
};

template <class Source>
void MatrixCrossTeam(const Source source, const int n, const int m, const int r,
                     ParArray3D<double> C_dev, ParArray3D<double> R_dev,
                     ParArray2D<int> I_dev, ParArray2D<int> J_dev) {
  const int scratch_level = 0;
  const std::size_t nwork = MatrixCross::double_scratch_size(n, m, r);
  const std::size_t scratch_bytes = MatrixCross::total_shmem_scratch_size(n, m, r) +
                                    ScratchPad2D<double>::shmem_size(n, r) +
                                    ScratchPad2D<double>::shmem_size(r, m) +
                                    2 * ScratchPad1D<int>::shmem_size(r);
  parthenon::par_for_outer(
      "MatrixCrossTeam", scratch_bytes, scratch_level, 0, nbatch - 1,
      KOKKOS_LAMBDA(team_mbr_t member, const int b) {
        ScratchPad1D<double> work(member.team_scratch(scratch_level), nwork);
        ScratchPad2D<double> C(member.team_scratch(scratch_level), n, r);
        ScratchPad2D<double> R(member.team_scratch(scratch_level), r, m);
        ScratchPad1D<int> I(member.team_scratch(scratch_level), r);
        ScratchPad1D<int> J(member.team_scratch(scratch_level), r);
        const auto A = source(b);

        MatrixCross::execute(member, A, &C, &R, I.data(), J.data(), r, work.data());
        member.team_barrier();

        parthenon::par_for_inner(
            member, 0, n - 1, 0, r - 1,
            [&](const int i, const int k) { C_dev(b, i, k) = C(i, k); });
        parthenon::par_for_inner(
            member, 0, r - 1, 0, m - 1,
            [&](const int k, const int j) { R_dev(b, k, j) = R(k, j); });
        parthenon::par_for_inner(member, 0, r - 1, [&](const int k) {
          I_dev(b, k) = I(k);
          J_dev(b, k) = J(k);
        });
      });
}

template <class Source>
void MatrixCrossFlat(const Source source, const int n, const int m, const int r,
                     ParArray3D<double> C_dev, ParArray3D<double> R_dev,
                     ParArray2D<int> I_dev, ParArray2D<int> J_dev) {
  ParArray2D<double> work("work", nbatch, MatrixCross::double_scratch_size(n, m, r));
  const View3D C_view = C_dev;
  const View3D R_view = R_dev;
  parthenon::par_for(
      "MatrixCrossFlat", 0, nbatch - 1, KOKKOS_LAMBDA(const int b) {
        auto C = Kokkos::subview(C_view, b, Kokkos::ALL(), Kokkos::ALL());
        auto R = Kokkos::subview(R_view, b, Kokkos::ALL(), Kokkos::ALL());
        const auto A = source(b);
        MatrixCross::execute(serial_tm_t(), A, &C, &R, &I_dev(b, 0), &J_dev(b, 0), r,
                             &work(b, 0));
      });
}

void SVDTeam(ParArray3D<double> A_dev, ParArray3D<double> U_dev, ParArray3D<double> V_dev,
             ParArray2D<double> s_dev) {
  const int m = A_dev.extent_int(1);
  const int n = A_dev.extent_int(2);
  const int scratch_level = 0;
  const std::size_t nwork = SquareSVD::double_scratch_size(m, n);
  const std::size_t niwork = SquareSVD::sizet_scratch_size(n);
  const std::size_t scratch_bytes = SquareSVD::total_shmem_scratch_size(m, n) +
                                    2 * ScratchPad2D<double>::shmem_size(m, n) +
                                    ScratchPad2D<double>::shmem_size(n, n) +
                                    ScratchPad1D<double>::shmem_size(n);
  parthenon::par_for_outer(
      "SVDTeam", scratch_bytes, scratch_level, 0, nbatch - 1,
      KOKKOS_LAMBDA(team_mbr_t member, const int b) {
        ScratchPad1D<double> work(member.team_scratch(scratch_level), nwork);
        ScratchPad1D<std::size_t> iwork(member.team_scratch(scratch_level), niwork);
        ScratchPad2D<double> A(member.team_scratch(scratch_level), m, n);
        ScratchPad2D<double> U(member.team_scratch(scratch_level), m, n);
        ScratchPad2D<double> V(member.team_scratch(scratch_level), n, n);
        ScratchPad1D<double> s(member.team_scratch(scratch_level), n);
        parthenon::par_for_inner(
            member, 0, m - 1, 0, n - 1,
            [&](const int r, const int c) { A(r, c) = A_dev(b, r, c); });
        member.team_barrier();

        SquareSVD::execute(member, &A, &U, &V, s.data(), work.data(), iwork.data());
        member.team_barrier();

        parthenon::par_for_inner(
            member, 0, m - 1, 0, n - 1,
            [&](const int r, const int c) { U_dev(b, r, c) = U(r, c); });
        parthenon::par_for_inner(
            member, 0, n - 1, 0, n - 1,
            [&](const int r, const int c) { V_dev(b, r, c) = V(r, c); });
        parthenon::par_for_inner(member, 0, n - 1,
                                 [&](const int i) { s_dev(b, i) = s(i); });
      });
}

void SVDFlat(ParArray3D<double> A_dev, ParArray3D<double> U_dev, ParArray3D<double> V_dev,
             ParArray2D<double> s_dev) {
  const int m = A_dev.extent_int(1);
  const int n = A_dev.extent_int(2);
  ParArray2D<double> work("work", nbatch, SquareSVD::double_scratch_size(m, n));
  ParArray2D<std::size_t> iwork("iwork", nbatch, SquareSVD::sizet_scratch_size(n));
  const View3D A_view = A_dev;
  const View3D U_view = U_dev;
  const View3D V_view = V_dev;
  parthenon::par_for(
      "SVDFlat", 0, nbatch - 1, KOKKOS_LAMBDA(const int b) {
        auto A = Kokkos::subview(A_view, b, Kokkos::ALL(), Kokkos::ALL());
        auto U = Kokkos::subview(U_view, b, Kokkos::ALL(), Kokkos::ALL());
        auto V = Kokkos::subview(V_view, b, Kokkos::ALL(), Kokkos::ALL());
        SquareSVD::execute(serial_tm_t(), &A, &U, &V, &s_dev(b, 0), &work(b, 0),
                           &iwork(b, 0));
      });
}

void EVDTeam(ParArray3D<double> A_dev, ParArray3D<double> Q_dev,
             ParArray2D<double> eigs_dev) {
  const int n = A_dev.extent_int(1);
  const int scratch_level = 0;
  const std::size_t nwork = SymmetricEVD::double_scratch_size(n);
  const std::size_t niwork = SymmetricEVD::sizet_scratch_size(n);
  const std::size_t scratch_bytes = SymmetricEVD::total_shmem_scratch_size(n) +
                                    2 * ScratchPad2D<double>::shmem_size(n, n) +
                                    ScratchPad1D<double>::shmem_size(n);
  parthenon::par_for_outer(
      "EVDTeam", scratch_bytes, scratch_level, 0, nbatch - 1,
      KOKKOS_LAMBDA(team_mbr_t member, const int b) {
        ScratchPad1D<double> work(member.team_scratch(scratch_level), nwork);
        ScratchPad1D<std::size_t> iwork(member.team_scratch(scratch_level), niwork);
        ScratchPad2D<double> A(member.team_scratch(scratch_level), n, n);
        ScratchPad2D<double> Q(member.team_scratch(scratch_level), n, n);
        ScratchPad1D<double> eigs(member.team_scratch(scratch_level), n);
        parthenon::par_for_inner(
            member, 0, n - 1, 0, n - 1,
            [&](const int r, const int c) { A(r, c) = A_dev(b, r, c); });
        member.team_barrier();

        SymmetricEVD::execute(member, &A, &Q, eigs.data(), work.data(), iwork.data());
        member.team_barrier();

        parthenon::par_for_inner(
            member, 0, n - 1, 0, n - 1,
            [&](const int r, const int c) { Q_dev(b, r, c) = Q(r, c); });
        parthenon::par_for_inner(member, 0, n - 1,
                                 [&](const int i) { eigs_dev(b, i) = eigs(i); });
      });
}

void EVDFlat(ParArray3D<double> A_dev, ParArray3D<double> Q_dev,
             ParArray2D<double> eigs_dev) {
  const int n = A_dev.extent_int(1);
  ParArray2D<double> work("work", nbatch, SymmetricEVD::double_scratch_size(n));
  ParArray2D<std::size_t> iwork("iwork", nbatch, SymmetricEVD::sizet_scratch_size(n));
  const View3D A_view = A_dev;
  const View3D Q_view = Q_dev;
  parthenon::par_for(
      "EVDFlat", 0, nbatch - 1, KOKKOS_LAMBDA(const int b) {
        auto A = Kokkos::subview(A_view, b, Kokkos::ALL(), Kokkos::ALL());
        auto Q = Kokkos::subview(Q_view, b, Kokkos::ALL(), Kokkos::ALL());
        SymmetricEVD::execute(serial_tm_t(), &A, &Q, &eigs_dev(b, 0), &work(b, 0),
                              &iwork(b, 0));
      });
}

// ---------------------------------------------------------------------------
// Checks against the serial host result.
// ---------------------------------------------------------------------------

// QR and LQ run the same algorithm on host and device, so the factors should
// agree up to rounding.
template <class Decomposition>
void CheckFactorization(const std::vector<Matrix> &A0, const std::vector<Matrix> &F,
                        const std::vector<Matrix> &Q) {
  for (int b = 0; b < nbatch; ++b) {
    Matrix F_host = A0[b].GetDeepCopy();
    Matrix Q_host(Q[b].nrows(), Q[b].ncols());
    REQUIRE(Decomposition::execute(&F_host, &Q_host) == 0);
    REQUIRE(MaxAbsDiff(F[b], F_host) < 1e-12 * A0[b].FrobeniusNorm());
    REQUIRE(MaxAbsDiff(Q[b], Q_host) < 1e-12);
  }
}

template <class Decomposition>
void TestFactorization(const bool team, const int m, const int n, const unsigned seed) {
  const auto A0 = RandomBatch(m, n, seed);
  auto A_dev = ToDevice(A0);
  ParArray3D<double> Q_dev("Q", nbatch, m, n);
  if (team) {
    FactorTeam<Decomposition>(A_dev, Q_dev);
  } else {
    FactorFlat<Decomposition>(A_dev, Q_dev);
  }
  CheckFactorization<Decomposition>(A0, ToHost(A_dev), ToHost(Q_dev));
}

template <class Decomposition, int m, int n>
void TestFactorizationLocal(const unsigned seed) {
  const auto A0 = RandomBatch(m, n, seed);
  auto A_dev = ToDevice(A0);
  ParArray3D<double> Q_dev("Q", nbatch, m, n);
  FactorFlatLocal<Decomposition, m, n>(A_dev, Q_dev);
  CheckFactorization<Decomposition>(A0, ToHost(A_dev), ToHost(Q_dev));
}

// Team reductions and fused multiply-adds on GPUs change the rounding, so a
// direct comparison with the host solution would scale with cond(A). Instead
// check that the device X satisfies the normal equations Aᵀ (A X - B) = 0 to
// within a backward-stable tolerance, which holds for square and tall A alike.
// A, B and X are given in the left-solve form A X = B.
void CheckNormalEquations(const Matrix &A, const Matrix &B, const Matrix &X) {
  Matrix res(B.nrows(), B.ncols());
  Multiply(A, X, res);
  for (int r = 0; r < B.nrows(); ++r) {
    for (int c = 0; c < B.ncols(); ++c) {
      res(r, c) -= B(r, c);
    }
  }
  Matrix normal(X.nrows(), X.ncols());
  Multiply(Matrix::Transpose(A), res, normal);
  const double a = A.FrobeniusNorm();
  REQUIRE(normal.FrobeniusNorm() <
          1e-13 * a * (a * X.FrobeniusNorm() + B.FrobeniusNorm()));
}

// Solves A X = B with QRSolve, or X A = B with QRSolveRight, where A is
// a_rows x a_cols and there are nrhs right-hand sides.
template <class Solver>
void TestSolve(const bool team, const int a_rows, const int a_cols, const int nrhs,
               const unsigned seed) {
  constexpr bool right = std::is_same_v<Solver, QRSolveRight>;
  const int b_rows = right ? nrhs : a_rows;
  const int b_cols = right ? a_cols : nrhs;
  const auto A0 = RandomBatch(a_rows, a_cols, seed);
  const auto B0 = RandomBatch(b_rows, b_cols, seed + 1000u);
  auto A_dev = ToDevice(A0);
  auto B_dev = ToDevice(B0);
  if (team) {
    SolveTeam<Solver>(A_dev, B_dev);
  } else {
    SolveFlat<Solver>(A_dev, B_dev);
  }

  const auto B = ToHost(B_dev);
  for (int b = 0; b < nbatch; ++b) {
    // For X A = B, check the equivalent left solve Aᵀ Xᵀ = Bᵀ
    const Matrix A = right ? Matrix::Transpose(A0[b]) : A0[b];
    const Matrix B_rhs = right ? Matrix::Transpose(B0[b]) : B0[b];
    const Matrix B_out = right ? Matrix::Transpose(B[b]) : B[b];
    const int n = A.ncols();
    Matrix X(n, nrhs);
    for (int r = 0; r < n; ++r) {
      for (int c = 0; c < nrhs; ++c) {
        X(r, c) = B_out(r, c);
      }
    }
    CheckNormalEquations(A, B_rhs, X);
  }
}

// Rounding and tie-breaking differ between host and device, so the selected
// rows may differ while being equally valid. Check each result on its own:
// distinct rows, B(I,:) = identity, B A(I,:) = A, and max |B| <= tau.
void TestMaxvol(const bool team, const int n, const int r, const unsigned seed) {
  const auto A0 = RandomBatch(n, r, seed);
  auto A_dev = ToDevice(A0);
  ParArray3D<double> B_dev("B", nbatch, n, r);
  ParArray2D<int> I_dev("I", nbatch, r);
  if (team) {
    MaxvolTeam(A_dev, B_dev, I_dev);
  } else {
    MaxvolFlat(A_dev, B_dev, I_dev);
  }

  const auto B = ToHost(B_dev);
  const auto I_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), I_dev);
  for (int b = 0; b < nbatch; ++b) {
    std::vector<int> I(r);
    for (int j = 0; j < r; ++j) {
      I[j] = I_host(b, j);
      REQUIRE(I[j] >= 0);
      REQUIRE(I[j] < n);
    }
    std::vector<int> sorted = I;
    std::sort(sorted.begin(), sorted.end());
    REQUIRE(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end());

    Matrix S(r, r);
    for (int a = 0; a < r; ++a) {
      for (int c = 0; c < r; ++c) {
        S(a, c) = A0[b](I[a], c);
        REQUIRE(std::abs(B[b](I[a], c) - (a == c)) < 1e-12);
      }
    }
    Matrix BS(n, r);
    Multiply(B[b], S, BS);
    REQUIRE(MaxAbsDiff(BS, A0[b]) < 1e-12 * A0[b].FrobeniusNorm());

    double bmax = 0.0;
    for (int i = 0; i < n; ++i) {
      for (int c = 0; c < r; ++c) {
        bmax = std::max(bmax, std::abs(B[b](i, c)));
      }
    }
    REQUIRE(bmax <= kMaxvolTau * (1.0 + 1e-12));
  }
}

// Runs the cross on device for either stored low-rank matrices or lazily
// evaluated Hilbert-like ones, and checks each result on its own since the
// selected indices may differ from host. Low-rank inputs must be recovered
// exactly; the Hilbert-like ones must be within the maximum-volume bound
// (r + 1) sigma_{r+1} in the max norm.
void TestMatrixCross(const bool team, const bool lazy, const int n, const int m,
                     const int r, const unsigned seed) {
  std::vector<Matrix> A0;
  for (int b = 0; b < nbatch; ++b) {
    Matrix A(n, m);
    if (lazy) {
      const DeviceHilbert H{n, m, static_cast<double>(b)};
      for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
          A(i, j) = H(i, j);
        }
      }
    } else {
      Multiply(Matrix::RandomGaussian(n, r, seed + b),
               Matrix::RandomGaussian(r, m, seed + b + 1000u), A);
    }
    A0.push_back(A);
  }

  ParArray3D<double> C_dev("C", nbatch, n, r);
  ParArray3D<double> R_dev("R", nbatch, r, m);
  ParArray2D<int> I_dev("I", nbatch, r);
  ParArray2D<int> J_dev("J", nbatch, r);
  auto run = [&](const auto source) {
    if (team) {
      MatrixCrossTeam(source, n, m, r, C_dev, R_dev, I_dev, J_dev);
    } else {
      MatrixCrossFlat(source, n, m, r, C_dev, R_dev, I_dev, J_dev);
    }
  };
  if (lazy) {
    run(HilbertSource{n, m});
  } else {
    run(StoredSource{ToDevice(A0)});
  }

  const auto C = ToHost(C_dev);
  const auto R = ToHost(R_dev);
  const auto I_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), I_dev);
  const auto J_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), J_dev);
  for (int b = 0; b < nbatch; ++b) {
    std::vector<int> I(r), J(r);
    for (int k = 0; k < r; ++k) {
      I[k] = I_host(b, k);
      J[k] = J_host(b, k);
      REQUIRE(I[k] >= 0);
      REQUIRE(I[k] < n);
      REQUIRE(J[k] >= 0);
      REQUIRE(J[k] < m);
    }
    for (auto idx : {I, J}) {
      std::sort(idx.begin(), idx.end());
      REQUIRE(std::adjacent_find(idx.begin(), idx.end()) == idx.end());
    }

    for (int a = 0; a < r; ++a) {
      for (int k = 0; k < r; ++k) {
        REQUIRE(std::abs(C[b](I[a], k) - (a == k)) < 1e-12);
      }
      for (int j = 0; j < m; ++j) {
        REQUIRE(std::abs(R[b](a, j) - A0[b](I[a], j)) <=
                1e-14 * std::abs(A0[b](I[a], j)));
      }
    }

    Matrix CR(n, m);
    Multiply(C[b], R[b], CR);
    if (lazy) {
      Matrix dense = A0[b].GetDeepCopy();
      std::vector<double> sings(m);
      SquareSVD::execute(&dense, sings.data());
      std::sort(sings.begin(), sings.end(), std::greater<double>());
      REQUIRE(MaxAbsDiff(CR, A0[b]) <= (r + 1) * sings[r]);
    } else {
      REQUIRE(MaxAbsDiff(CR, A0[b]) < 1e-12 * A0[b].FrobeniusNorm());
    }
  }
}

// The SVD and EVD iterate, so compare the (sorted) values against host and
// check the vectors through orthogonality and the residual.
void TestSVD(const bool team, const int m, const int n, const unsigned seed) {
  const auto A0 = RandomBatch(m, n, seed);
  auto A_dev = ToDevice(A0);
  ParArray3D<double> U_dev("U", nbatch, m, n);
  ParArray3D<double> V_dev("V", nbatch, n, n);
  ParArray2D<double> s_dev("s", nbatch, n);
  if (team) {
    SVDTeam(A_dev, U_dev, V_dev, s_dev);
  } else {
    SVDFlat(A_dev, U_dev, V_dev, s_dev);
  }

  const auto U = ToHost(U_dev);
  const auto V = ToHost(V_dev);
  const auto s = ToHost(s_dev);
  for (int b = 0; b < nbatch; ++b) {
    Matrix A_host = A0[b].GetDeepCopy();
    std::vector<double> s_host(n);
    SquareSVD::execute(&A_host, s_host.data());

    const double scale = A0[b].FrobeniusNorm();
    REQUIRE(MaxAbsDiff(s[b], s_host) < 1e-12 * scale);
    REQUIRE(ColumnOrthoError(U[b]) < 1e-12);
    REQUIRE(ColumnOrthoError(V[b]) < 1e-12);
    REQUIRE(PairResidual(A0[b], U[b], V[b], s[b]) < 1e-12 * scale);
  }
}

void TestEVD(const bool team, const int n, const unsigned seed) {
  const auto A0 = RandomSymmetricBatch(n, seed);
  auto A_dev = ToDevice(A0);
  ParArray3D<double> Q_dev("Q", nbatch, n, n);
  ParArray2D<double> eigs_dev("eigs", nbatch, n);
  if (team) {
    EVDTeam(A_dev, Q_dev, eigs_dev);
  } else {
    EVDFlat(A_dev, Q_dev, eigs_dev);
  }

  const auto Q = ToHost(Q_dev);
  const auto eigs = ToHost(eigs_dev);
  for (int b = 0; b < nbatch; ++b) {
    Matrix A_host = A0[b].GetDeepCopy();
    std::vector<double> eigs_host(n);
    REQUIRE(SymmetricEVD::execute(&A_host, eigs_host.data()) >= 0);

    const double scale = A0[b].FrobeniusNorm();
    REQUIRE(MaxAbsDiff(eigs[b], eigs_host) < 1e-12 * scale);
    REQUIRE(ColumnOrthoError(Q[b]) < 1e-12);
    REQUIRE(PairResidual(A0[b], Q[b], Q[b], eigs[b]) < 1e-12 * scale);
  }
}

} // namespace linalg_device_test

using namespace linalg_device_test; // NOLINT(build/namespaces)

// The 40-row cases make the matrices larger than a warp, so a team spans more
// than one warp and missing team barriers can show up as wrong answers.

TEST_CASE("Batched QR decomposition on device", "[qr][device]") {
  for (const bool team : {true, false}) {
    for (const int m : {12, 40}) {
      TestFactorization<QRDecomposition>(team, m, m / 2, 100u);
    }
  }
  TestFactorizationLocal<QRDecomposition, 6, 3>(150u);
}

TEST_CASE("Batched LQ decomposition on device", "[qr][device]") {
  for (const bool team : {true, false}) {
    for (const int n : {12, 40}) {
      TestFactorization<LQDecomposition>(team, n / 2, n, 200u);
    }
  }
}

TEST_CASE("Batched QR solve on device", "[qr_solve][device]") {
  for (const bool team : {true, false}) {
    // Square and tall systems, with more and fewer right-hand sides than columns
    TestSolve<QRSolve>(team, 12, 12, 5, 500u);
    TestSolve<QRSolve>(team, 40, 20, 3, 600u);
    TestSolve<QRSolve>(team, 40, 40, 50, 700u);
    TestSolve<QRSolveRight>(team, 12, 12, 5, 800u);
    TestSolve<QRSolveRight>(team, 20, 40, 50, 900u);
  }
}

TEST_CASE("Batched maxvol on device", "[maxvol][device]") {
  for (const bool team : {true, false}) {
    TestMaxvol(team, 12, 4, 1000u);
    TestMaxvol(team, 40, 6, 1100u);
  }
}

TEST_CASE("Batched matrix cross on device", "[matrix_cross][device]") {
  for (const bool team : {true, false}) {
    for (const bool lazy : {false, true}) {
      TestMatrixCross(team, lazy, 12, 10, 3, 1200u);
      TestMatrixCross(team, lazy, 40, 24, 5, 1300u);
    }
  }
}

TEST_CASE("Batched SVD on device", "[svd][device]") {
  for (const bool team : {true, false}) {
    for (const int m : {12, 40}) {
      TestSVD(team, m, m / 2, 300u);
    }
  }
}

TEST_CASE("Batched symmetric EVD on device", "[eig][device]") {
  for (const bool team : {true, false}) {
    for (const int n : {8, 40}) {
      TestEVD(team, n, 400u);
    }
  }
}
