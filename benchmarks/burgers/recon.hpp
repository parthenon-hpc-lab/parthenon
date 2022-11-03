//========================================================================================
// (C) (or copyright) 2020-2022. Triad National Security, LLC. All rights reserved.
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

#ifndef BENCHMARKS_BURGERS_RECON_HPP_
#define BENCHMARKS_BURGERS_RECON_HPP_

#include "basic_types.hpp"
using parthenon::Real;

namespace recon {

enum class ReconType {WENO5, Linear};

KOKKOS_INLINE_FUNCTION
Real mc(const Real dm, const Real dp) {
  const Real dc = (dm * dp > 0.0) * 0.5 * (dm + dp);
  return std::copysign(
      std::min(std::fabs(dc), 2.0 * std::min(std::fabs(dm), std::fabs(dp))), dc);
}

KOKKOS_INLINE_FUNCTION
void Linear(const Real qm, const Real q0, const Real qp, Real &ql, Real &qr) {
  Real dq = qp - q0;
  dq = 0.5 * mc(q0 - qm, dq);
  ql = q0 + dq;
  qr = q0 - dq;
}

KOKKOS_INLINE_FUNCTION
void WENO5Z(const Real q0, const Real q1, const Real q2, const Real q3, const Real q4,
            Real &ql, Real &qr) {
  constexpr Real w5alpha[3][3] = {{1.0 / 3.0, -7.0 / 6.0, 11.0 / 6.0},
                                  {-1.0 / 6.0, 5.0 / 6.0, 1.0 / 3.0},
                                  {1.0 / 3.0, 5.0 / 6.0, -1.0 / 6.0}};
  constexpr Real w5gamma[3] = {0.1, 0.6, 0.3};
  constexpr Real eps = 1e-100;
  constexpr Real thirteen_thirds = 13.0 / 3.0;

  Real a = q0 - 2 * q1 + q2;
  Real b = q0 - 4.0 * q1 + 3.0 * q2;
  Real beta0 = thirteen_thirds * a * a + b * b + eps;
  a = q1 - 2.0 * q2 + q3;
  b = q3 - q1;
  Real beta1 = thirteen_thirds * a * a + b * b + eps;
  a = q2 - 2.0 * q3 + q4;
  b = q4 - 4.0 * q3 + 3.0 * q2;
  Real beta2 = thirteen_thirds * a * a + b * b + eps;
  const Real tau5 = std::abs(beta2 - beta0);

  beta0 = (beta0 + tau5) / beta0;
  beta1 = (beta1 + tau5) / beta1;
  beta2 = (beta2 + tau5) / beta2;

  Real w0 = w5gamma[0] * beta0 + eps;
  Real w1 = w5gamma[1] * beta1 + eps;
  Real w2 = w5gamma[2] * beta2 + eps;
  Real wsum = 1.0 / (w0 + w1 + w2);
  ql = w0 * (w5alpha[0][0] * q0 + w5alpha[0][1] * q1 + w5alpha[0][2] * q2);
  ql += w1 * (w5alpha[1][0] * q1 + w5alpha[1][1] * q2 + w5alpha[1][2] * q3);
  ql += w2 * (w5alpha[2][0] * q2 + w5alpha[2][1] * q3 + w5alpha[2][2] * q4);
  ql *= wsum;
  const Real alpha_l =
      3.0 * wsum * w0 * w1 * w2 /
          (w5gamma[2] * w0 * w1 + w5gamma[1] * w0 * w2 + w5gamma[0] * w1 * w2) +
      eps;

  w0 = w5gamma[0] * beta2 + eps;
  w1 = w5gamma[1] * beta1 + eps;
  w2 = w5gamma[2] * beta0 + eps;
  wsum = 1.0 / (w0 + w1 + w2);
  qr = w0 * (w5alpha[0][0] * q4 + w5alpha[0][1] * q3 + w5alpha[0][2] * q2);
  qr += w1 * (w5alpha[1][0] * q3 + w5alpha[1][1] * q2 + w5alpha[1][2] * q1);
  qr += w2 * (w5alpha[2][0] * q2 + w5alpha[2][1] * q1 + w5alpha[2][2] * q0);
  qr *= wsum;
  const Real alpha_r =
      3.0 * wsum * w0 * w1 * w2 /
          (w5gamma[2] * w0 * w1 + w5gamma[1] * w0 * w2 + w5gamma[0] * w1 * w2) +
      eps;

  Real dq = q3 - q2;
  dq = 0.5 * mc(q2 - q1, dq);

  const Real alpha_lin = 2.0 * alpha_l * alpha_r / (alpha_l + alpha_r);
  ql = alpha_lin * ql + (1.0 - alpha_lin) * (q2 + dq);
  qr = alpha_lin * qr + (1.0 - alpha_lin) * (q2 - dq);
}

} // namespace recon

#endif // BENCHMARKS_BURGERS_RECON_HPP_
