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
#include "parameter_input.hpp"
#include "time_integration/staged_integrator.hpp"

using parthenon::ButcherIntegrator;
using parthenon::LowStorageIntegrator;
using parthenon::ParameterInput;
using parthenon::Real;
using parthenon::StagedIntegrator;

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

constexpr Real K = 2 * M_PI;
constexpr std::size_t NVARS = 2;
using State_t = std::array<Real, NVARS>;
void GetRHS(const State_t &u, State_t &rhs) {
  rhs[0] = u[1];
  rhs[1] = -K * K * u[0];
}

void GetTrueSolution(const Real t, State_t &u) {
  u[0] = std::cos(K * t);
  u[1] = -K * std::sin(K * t);
}

void GetInitialData(State_t &u) { GetTrueSolution(0, u); }

template <typename T>
auto MakeIntegrator(const std::string &integration_strategy) {
  ParameterInput in;
  in.SetString("parthenon/time", "integrator", integration_strategy);
  return T(&in);
}

/*
 * See Equation 14 in
 * Ketcheson, Jcomp 229 (2010) 1763-1773
 */
void Step2SStar(const LowStorageIntegrator &integrator, Real dt, State_t &u) {
  const int nstages = integrator.nstages;
  State_t &S0 = u;
  State_t S1;
  for (int v = 0; v < NVARS; ++v) {
    S1[v] = u[v]; // set "last step" to prepare for next cycle
  }
  for (int stage = 1; stage <= nstages; stage++) {
    State_t rhs;
    const Real delta = integrator.delta[stage - 1];
    const Real beta = integrator.beta[stage - 1];
    const Real gam0 = integrator.gam0[stage - 1];
    const Real gam1 = integrator.gam1[stage - 1];
    GetRHS(S0, rhs);
    for (int v = 0; v < NVARS; ++v) {
      // S1[v] = S1[v] + delta * S0[v];
      S0[v] = gam0 * S0[v] + gam1 * S1[v] + beta * dt * rhs[v];
    }
  }
  for (int v = 0; v < NVARS; ++v) {
    u[v] = S0[v];
  }
}

void StepButcher(const ButcherIntegrator &integrator, Real dt, State_t &u) {
  const int nstages = integrator.nstages;
  std::vector<State_t> K(nstages);
  for (int stage = 0; stage < nstages; ++stage) {
    State_t scratch;
    for (int v = 0; v < NVARS; ++v) {
      scratch[v] = u[v];
    }
    for (int prev = 0; prev < stage; ++prev) {
      for (int v = 0; v < NVARS; ++v) {
        scratch[v] += dt * integrator.a[stage][prev] * K[prev][v];
      }
    }
    GetRHS(scratch, K[stage]);
  }
  for (int stage = 0; stage < nstages; ++stage) {
    for (int v = 0; v < NVARS; ++v) {
      u[v] += dt * integrator.b[stage] * K[stage][v];
    }
  }
}

template <typename Integrator, typename Stepper>
void Integrate(const Integrator &integrator, const Stepper &step, const Real tf, Real dt,
               State_t &u0) {
  assert(tf > 0);
  assert(dt > 0);
  assert(dt < tf);
  // stupid game to align to EXACTLY tf
  int NT = std::ceil(tf / dt);
  Real t = 0;
  for (int i = 0; i < NT; ++i) {
    step(integrator, dt, u0);
    t += dt;
  }
  if ((t < tf) && (std::abs(tf - t) > 1e-12)) {
    dt = tf - t;
    step(integrator, dt, u0);
    t += dt;
  }
}

TEST_CASE("Low storage integrator", "[StagedIntegrator]") {
  GIVEN("A state with an initial condition") {
    Real tf = 1.15; // delibarately not a nice fraction of a period
    State_t ufinal;
    GetTrueSolution(tf, ufinal);
    WHEN("We integrate with LowStorage rk1") {
      constexpr Real dt = 1e-5;
      auto integrator = MakeIntegrator<LowStorageIntegrator>("rk1");
      State_t u;
      GetInitialData(u);
      Integrate(integrator, Step2SStar, tf, dt, u);
      THEN("The final state doesn't differ too much from the true solution") {
        REQUIRE(std::abs(u[0] - ufinal[0]) <= 1e-2);
        REQUIRE(std::abs(u[1] - ufinal[1]) <= 1e-2);
      }
    }
    WHEN("We integrate with LowStorage rk2") {
      constexpr Real dt = 1e-3;
      auto integrator = MakeIntegrator<LowStorageIntegrator>("rk2");
      State_t u;
      GetInitialData(u);
      Integrate(integrator, Step2SStar, tf, dt, u);
      THEN("The final state doesn't differ too much from the true solution") {
        REQUIRE(std::abs(u[0] - ufinal[0]) <= 1e-2);
        REQUIRE(std::abs(u[1] - ufinal[1]) <= 1e-2);
      }
    }
    WHEN("We integrate with LowStorage vl2") {
      constexpr Real dt = 1e-3;
      auto integrator = MakeIntegrator<LowStorageIntegrator>("vl2");
      State_t u;
      GetInitialData(u);
      Integrate(integrator, Step2SStar, tf, dt, u);
      THEN("The final state doesn't differ too much from the true solution") {
        REQUIRE(std::abs(u[0] - ufinal[0]) <= 1e-2);
        REQUIRE(std::abs(u[1] - ufinal[1]) <= 1e-2);
      }
    }
    WHEN("We integrate with LowStorage rk3") {
      constexpr Real dt = 1e-2;
      auto integrator = MakeIntegrator<LowStorageIntegrator>("rk3");
      State_t u;
      GetInitialData(u);
      Integrate(integrator, Step2SStar, tf, dt, u);
      THEN("The final state doesn't differ too much from the true solution") {
        REQUIRE(std::abs(u[0] - ufinal[0]) <= 1e-2);
        REQUIRE(std::abs(u[1] - ufinal[1]) <= 1e-2);
      }
    }
    WHEN("We integrate with LowStorage rk4") {
      // still accurate with large timestep
      constexpr Real dt = 1e-2;
      auto integrator = MakeIntegrator<LowStorageIntegrator>("rk4");
      State_t u;
      GetInitialData(u);
      Integrate(integrator, Step2SStar, tf, dt, u);
      THEN("The final state doesn't differ too much from the true solution") {
        REQUIRE(std::abs(u[0] - ufinal[0]) <= 1e-2);
        REQUIRE(std::abs(u[1] - ufinal[1]) <= 1e-2);
      }
    }
    WHEN("We integrate with butcher rk1") {
      constexpr Real dt = 1e-5;
      auto integrator = MakeIntegrator<ButcherIntegrator>("rk1");
      State_t u;
      GetInitialData(u);
      Integrate(integrator, StepButcher, tf, dt, u);
      THEN("The final state doesn't differ too much from the true solution") {
        REQUIRE(std::abs(u[0] - ufinal[0]) <= 1e-2);
        REQUIRE(std::abs(u[1] - ufinal[1]) <= 1e-2);
      }
    }
    WHEN("We integrate with butcher rk2") {
      constexpr Real dt = 1e-5;
      auto integrator = MakeIntegrator<ButcherIntegrator>("rk2");
      State_t u;
      GetInitialData(u);
      Integrate(integrator, StepButcher, tf, dt, u);
      THEN("The final state doesn't differ too much from the true solution") {
        REQUIRE(std::abs(u[0] - ufinal[0]) <= 1e-2);
        REQUIRE(std::abs(u[1] - ufinal[1]) <= 1e-2);
      }
    }
    WHEN("We integrate with butcher rk4") {
      // Still good accuracy with large timestep.
      constexpr Real dt = 1e-2;
      auto integrator = MakeIntegrator<ButcherIntegrator>("rk4");
      State_t u;
      GetInitialData(u);
      Integrate(integrator, StepButcher, tf, dt, u);
      THEN("The final state doesn't differ too much from the true solution") {
        REQUIRE(std::abs(u[0] - ufinal[0]) <= 1e-4);
        REQUIRE(std::abs(u[1] - ufinal[1]) <= 1e-4);
      }
    }
    WHEN("We integrate with butcher rk10") {
      constexpr Real dt = 5e-2; // appears to be largest stable timestep
      auto integrator = MakeIntegrator<ButcherIntegrator>("rk10");
      State_t u;
      GetInitialData(u);
      Integrate(integrator, StepButcher, tf, dt, u);
      THEN("The final state doesn't differ too much from the true solution") {
        REQUIRE(std::abs(u[0] - ufinal[0]) <= 1e-9);
        REQUIRE(std::abs(u[1] - ufinal[1]) <= 1e-9);
      }
    }
  }
}
