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

// Run each batched_linear_algebra decomposition once per team inside a
// par_for_outer kernel, with all matrices and workspace in team scratch, and
// compare against the serial host result. The numerical tests in the other
// linalg unit test files only exercise the serial path; these exist to make
// sure the team path compiles for device and is free of races.

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

namespace {

constexpr int scratch_level = 0;
constexpr int nbatch = 8;

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

} // namespace

TEST_CASE("Batched QR decomposition on device", "[qr][device]") {
  for (const int m : {12, 40}) {
    const int n = m / 2;
    const auto A0 = RandomBatch(m, n, 100u);
    auto A_dev = ToDevice(A0);
    ParArray3D<double> Q_dev("Q", nbatch, m, n);

    const std::size_t nwork = QRDecomposition::double_scratch_size(m, n);
    const std::size_t scratch_bytes = QRDecomposition::total_shmem_scratch_size(m, n) +
                                      2 * ScratchPad2D<double>::shmem_size(m, n);
    parthenon::par_for_outer(
        "DeviceQR", scratch_bytes, scratch_level, 0, nbatch - 1,
        KOKKOS_LAMBDA(team_mbr_t member, const int b) {
          ScratchPad1D<double> work(member.team_scratch(scratch_level), nwork);
          ScratchPad2D<double> A(member.team_scratch(scratch_level), m, n);
          ScratchPad2D<double> Q(member.team_scratch(scratch_level), m, n);
          parthenon::par_for_inner(
              member, 0, m - 1, 0, n - 1,
              [&](const int r, const int c) { A(r, c) = A_dev(b, r, c); });
          member.team_barrier();

          QRDecomposition::execute(member, &A, &Q, work.data());
          member.team_barrier();

          parthenon::par_for_inner(member, 0, m - 1, 0, n - 1,
                                   [&](const int r, const int c) {
                                     A_dev(b, r, c) = A(r, c);
                                     Q_dev(b, r, c) = Q(r, c);
                                   });
        });

    const auto R = ToHost(A_dev);
    const auto Q = ToHost(Q_dev);
    for (int b = 0; b < nbatch; ++b) {
      Matrix R_host = A0[b].GetDeepCopy();
      Matrix Q_host(m, n);
      REQUIRE(QRDecomposition::execute(&R_host, &Q_host) == 0);
      REQUIRE(MaxAbsDiff(R[b], R_host) < 1e-12 * A0[b].FrobeniusNorm());
      REQUIRE(MaxAbsDiff(Q[b], Q_host) < 1e-12);
    }
  }
}

TEST_CASE("Batched LQ decomposition on device", "[qr][device]") {
  for (const int n : {12, 40}) {
    const int m = n / 2;
    const auto A0 = RandomBatch(m, n, 200u);
    auto A_dev = ToDevice(A0);
    ParArray3D<double> Q_dev("Q", nbatch, m, n);

    const std::size_t nwork = LQDecomposition::double_scratch_size(m, n);
    const std::size_t scratch_bytes = LQDecomposition::total_shmem_scratch_size(m, n) +
                                      2 * ScratchPad2D<double>::shmem_size(m, n);
    parthenon::par_for_outer(
        "DeviceLQ", scratch_bytes, scratch_level, 0, nbatch - 1,
        KOKKOS_LAMBDA(team_mbr_t member, const int b) {
          ScratchPad1D<double> work(member.team_scratch(scratch_level), nwork);
          ScratchPad2D<double> A(member.team_scratch(scratch_level), m, n);
          ScratchPad2D<double> Q(member.team_scratch(scratch_level), m, n);
          parthenon::par_for_inner(
              member, 0, m - 1, 0, n - 1,
              [&](const int r, const int c) { A(r, c) = A_dev(b, r, c); });
          member.team_barrier();

          LQDecomposition::execute(member, &A, &Q, work.data());
          member.team_barrier();

          parthenon::par_for_inner(member, 0, m - 1, 0, n - 1,
                                   [&](const int r, const int c) {
                                     A_dev(b, r, c) = A(r, c);
                                     Q_dev(b, r, c) = Q(r, c);
                                   });
        });

    const auto L = ToHost(A_dev);
    const auto Q = ToHost(Q_dev);
    for (int b = 0; b < nbatch; ++b) {
      Matrix L_host = A0[b].GetDeepCopy();
      Matrix Q_host(m, n);
      REQUIRE(LQDecomposition::execute(&L_host, &Q_host) == 0);
      REQUIRE(MaxAbsDiff(L[b], L_host) < 1e-12 * A0[b].FrobeniusNorm());
      REQUIRE(MaxAbsDiff(Q[b], Q_host) < 1e-12);
    }
  }
}

TEST_CASE("Batched SVD on device", "[svd][device]") {
  for (const int m : {12, 40}) {
    const int n = m / 2;
    const auto A0 = RandomBatch(m, n, 300u);
    auto A_dev = ToDevice(A0);
    ParArray3D<double> U_dev("U", nbatch, m, n);
    ParArray3D<double> V_dev("V", nbatch, n, n);
    ParArray2D<double> s_dev("s", nbatch, n);

    const std::size_t nwork = SquareSVD::double_scratch_size(m, n);
    const std::size_t niwork = SquareSVD::sizet_scratch_size(n);
    const std::size_t scratch_bytes = SquareSVD::total_shmem_scratch_size(m, n) +
                                      2 * ScratchPad2D<double>::shmem_size(m, n) +
                                      ScratchPad2D<double>::shmem_size(n, n) +
                                      ScratchPad1D<double>::shmem_size(n);
    parthenon::par_for_outer(
        "DeviceSVD", scratch_bytes, scratch_level, 0, nbatch - 1,
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
}

TEST_CASE("Batched symmetric EVD on device", "[eig][device]") {
  for (const int n : {8, 40}) {
    std::vector<Matrix> A0;
    for (int b = 0; b < nbatch; ++b) {
      const Matrix G = Matrix::RandomGaussian(n, n, 400u + b);
      Matrix S(n, n);
      for (int r = 0; r < n; ++r) {
        for (int c = 0; c < n; ++c) {
          S(r, c) = G(r, c) + G(c, r);
        }
      }
      A0.push_back(S);
    }
    auto A_dev = ToDevice(A0);
    ParArray3D<double> Q_dev("Q", nbatch, n, n);
    ParArray2D<double> eigs_dev("eigs", nbatch, n);

    const std::size_t nwork = SymmetricEVD::double_scratch_size(n);
    const std::size_t niwork = SymmetricEVD::sizet_scratch_size(n);
    const std::size_t scratch_bytes = SymmetricEVD::total_shmem_scratch_size(n) +
                                      2 * ScratchPad2D<double>::shmem_size(n, n) +
                                      ScratchPad1D<double>::shmem_size(n);
    parthenon::par_for_outer(
        "DeviceEVD", scratch_bytes, scratch_level, 0, nbatch - 1,
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
}
