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
#include <vector>

#include <catch2/catch.hpp>

#include "batched_linear_algebra/qr_decomposition.hpp"
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

// QRDecomposition and LQDecomposition are also the names of free functions,
// which hide the class names when used as template arguments.
using QR = class parthenon::batched_linear_algebra::QRDecomposition;
using LQ = class parthenon::batched_linear_algebra::LQDecomposition;

// The 40-row cases make the matrices larger than a warp, so a team spans more
// than one warp and missing team barriers can show up as wrong answers.

TEST_CASE("Batched QR decomposition on device", "[qr][device]") {
  for (const bool team : {true, false}) {
    for (const int m : {12, 40}) {
      TestFactorization<QR>(team, m, m / 2, 100u);
    }
  }
  TestFactorizationLocal<QR, 6, 3>(150u);
}

TEST_CASE("Batched LQ decomposition on device", "[qr][device]") {
  for (const bool team : {true, false}) {
    for (const int n : {12, 40}) {
      TestFactorization<LQ>(team, n / 2, n, 200u);
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
