//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2022 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2023. Triad National Security, LLC. All rights reserved.
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

#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include <catch2/catch.hpp>

#include "basic_types.hpp"
#include "time_integration/staged_integrator.hpp"
#include "parameter_input.hpp"

using parthenon::ParameterInput;
using parthenon::Real;
using parthenon::StagedIntegrator;
using parthenon::LowStorageIntegrator;

// Test our integrators by integrating the equation
// for a harmonic oscillator:
//
// d^2 u/dt^2 = - k^2 u
//
// reduced in 1d to:
//
// du/dt = v
// dv/dt = -k^2 u
//
// This is a strenuous test since any integrator that isn't
// time-reversible will accrue error every period.

constexpr Real K = 2*M_PI;
constexpr std::size_t NVARS = 2;
using State_t = std::array<Real, NVARS>;
void GetRHS(const State_t &u, State_t &rhs) {
  rhs[0] = u[1];
  rhs[1] = -K*K*u[0];
}

void GetTrueSolution(const Real t, State_t &u) {
  u[0] = std::cos(K*t);
  u[1] = -K*std::sin(K*t);
}

void GetInitialData(State_t &u) {
  GetTrueSolution(0, u);
}

auto Make2SStarIntegrator(const std::string &integration_strategy) {
  ParameterInput in;
  in.SetString("parthenon/time", "integrator", integration_strategy);
  return LowStorageIntegrator(&in);
}

/*
 * See Equation 14 in
 * Ketchson, Jcomp 229 (2010) 1763-1773
 */
template<typename Integrator_t>
void Step2SStar(const Integrator_t &integrator,
                Real dt, State_t &u) {
  const int nstages = integrator.nstages;
  State_t S0, S1;
  for (int v = 0; v < NVARS; ++v) {
    S0[v] = u[v];
    S1[v] = 0; // set "last step" to prepare for next cycle
  }
  for (int stage = 1; stage <= nstages; stage++) {
    State_t rhs;
    const Real delta = integrator.delta[stage-1];
    const Real beta = integrator.beta[stage-1];
    const Real gam0 = integrator.gam0[stage-1];
    const Real gam1 = integrator.gam1[stage-1];
    GetRHS(S0, rhs);
    for (int v = 0; v < NVARS; ++v) {
      S1[v] = S1[v] + delta * S0[v];
      // printf("%.14e %d: %.14e * %.14e + %.14e * %.14e + %.14e * %.14e\n",
      //        t, stage, gam0, S0[v], gam1, S1[v], beta, rhs[v]);
      S0[v] = gam0 * S0[v] + gam1 * S1[v] + beta * dt * rhs[v];
    }
  }
  for(int v = 0; v < NVARS; ++v) {
    u[v] = S0[v];
  }
}
template<typename Integrator_t>
void Integrate2SStar(const Integrator_t &integrator,
                     const Real tf, Real dt, State_t &u0) {
  assert( tf > 0 );
  assert( dt > 0 );
  assert( dt < tf );
  // stupid game to align to EXACTLY tf
  int NT = std::ceil(tf / dt);
  Real t = 0;
  for (int i = 0; i < NT; ++i) {
    Step2SStar(integrator, dt, u0);
    t += dt;
  }
  if ((t < tf) && (std::abs(tf - t) > 1e-12)) {
    dt = tf - t;
    Step2SStar(integrator, dt, u0);
  }
}

TEST_CASE("Integrators", "[StagedIntegrator]") {
  GIVEN("A state with an initial condition") {
    Real t0 = 0, tf = 1.15; // delibarately not a nice fraction of a period
    State_t ufinal;
    GetTrueSolution(tf, ufinal);
    WHEN("We integrate with rk1") {
      constexpr Real dt = 1e-5;
      auto integrator = Make2SStarIntegrator("rk1");
      State_t u;
      GetInitialData(u);
      Integrate2SStar(integrator, tf, dt, u);
      THEN("The final state doesn't differ too much from the true solution") {
        REQUIRE(std::abs(u[0] - ufinal[0]) <= 1e-2);
        REQUIRE(std::abs(u[1] - ufinal[1]) <= 1e-2);
      }
    }
    WHEN("We integrate with rk2") {
      constexpr Real dt = 1e-3;
      auto integrator = Make2SStarIntegrator("rk2");
      State_t u;
      GetInitialData(u);
      Integrate2SStar(integrator, tf, dt, u);
      THEN("The final state doesn't differ too much from the true solution") {
        REQUIRE(std::abs(u[0] - ufinal[0]) <= 1e-2);
        REQUIRE(std::abs(u[1] - ufinal[1]) <= 1e-2);
      }
    }
    WHEN("We integrate with vl2") {
      constexpr Real dt = 1e-3;
      auto integrator = Make2SStarIntegrator("vl2");
      State_t u;
      GetInitialData(u);
      Integrate2SStar(integrator, tf, dt, u);
      THEN("The final state doesn't differ too much from the true solution") {
        REQUIRE(std::abs(u[0] - ufinal[0]) <= 1e-2);
        REQUIRE(std::abs(u[1] - ufinal[1]) <= 1e-2);
      }
    }
    WHEN("We integrate with rk3") {
      constexpr Real dt = 1e-2;
      auto integrator = Make2SStarIntegrator("rk3");
      State_t u;
      GetInitialData(u);
      Integrate2SStar(integrator, tf, dt, u);
      THEN("The final state doesn't differ too much from the true solution") {
        REQUIRE(std::abs(u[0] - ufinal[0]) <= 1e-2);
        REQUIRE(std::abs(u[1] - ufinal[1]) <= 1e-2);
      }
    }
    WHEN("We integrate with rk4") {
      constexpr Real dt = 1e-2;
      auto integrator = Make2SStarIntegrator("rk4");
      State_t u;
      GetInitialData(u);
      Integrate2SStar(integrator, tf, dt, u);
      THEN("The final state doesn't differ too much from the true solution") {
        // debug
        std::printf("\t%.14e\t%.14e\t%.14e\t%.14e\t%.14e\t%.14e\n",
                    u[0], u[1], ufinal[0], ufinal[1],
                    std::abs(u[0] - ufinal[0]),
                    std::abs(u[1] - ufinal[1]));
        REQUIRE(std::abs(u[0] - ufinal[0]) <= 1e-2);
        REQUIRE(std::abs(u[1] - ufinal[1]) <= 1e-2);
      }
    }
  }
}
