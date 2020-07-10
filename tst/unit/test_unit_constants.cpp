//========================================================================================
// (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
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

#include <catch2/catch.hpp>

#include "utils/utils.hpp"

using parthenon::constants::CGS;
using parthenon::constants::PhysicalConstants;
using parthenon::constants::SI;

TEST_CASE("Physical constants", "[SI, CGS]") {
  SECTION("SI units") {
    PhysicalConstants<SI> pc;
    REQUIRE(pc.avogadro == Approx(6.02214129e23));
    REQUIRE(pc.na == Approx(pc.avogadro));
    REQUIRE(pc.fineStructure == Approx(7.2973525698e-3));
    REQUIRE(pc.alpha == Approx(pc.fineStructure));
    REQUIRE(pc.planck == Approx(6.62606957e-34));
    REQUIRE(pc.h == Approx(pc.planck));
    REQUIRE(pc.reducedPlanck == Approx(1.05457172e-34));
    REQUIRE(pc.hbar == Approx(pc.reducedPlanck));
    REQUIRE(pc.gasConstant == Approx(8.3144621));
    REQUIRE(pc.rGas == Approx(pc.gasConstant));
    REQUIRE(pc.boltzmann == Approx(1.380648800e-23));
    REQUIRE(pc.kb == Approx(pc.boltzmann));
    REQUIRE(pc.electronCharge == Approx(1.602176565e-19));
    REQUIRE(pc.qe == Approx(pc.electronCharge));
    REQUIRE(pc.speedOfLight == Approx(2.99792458e8));
    REQUIRE(pc.c == Approx(pc.speedOfLight));
    REQUIRE(pc.gravitationalConstant == Approx(6.67384e-11));
    REQUIRE(pc.gNewt == Approx(pc.gravitationalConstant));
    REQUIRE(pc.accelerationFromGravity == Approx(9.80665));
    REQUIRE(pc.gAccel == Approx(pc.accelerationFromGravity));
    REQUIRE(pc.electronMass == Approx(9.10938291e-31));
    REQUIRE(pc.me == Approx(pc.electronMass));
    REQUIRE(pc.protonMass == Approx(1.672621777e-27));
    REQUIRE(pc.mp == Approx(pc.protonMass));
    REQUIRE(pc.stefanBoltzmann == Approx(5.67037262e-8));
    REQUIRE(pc.sb == Approx(pc.stefanBoltzmann));
    REQUIRE(pc.faradayConstant == Approx(96485.33645957));
    REQUIRE(pc.faraday == Approx(pc.faradayConstant));
    REQUIRE(pc.permeabilityOfVacuum == Approx(1.25663706e-06));
    REQUIRE(pc.mu0 == Approx(pc.permeabilityOfVacuum));
    REQUIRE(pc.permittivityOfVacuum == Approx(8.85418782e-12));
    REQUIRE(pc.eps0 == Approx(pc.permittivityOfVacuum));
    REQUIRE(pc.classicalElectronRadius == Approx(2.81794033e-15));
    REQUIRE(pc.re == Approx(pc.classicalElectronRadius));
    REQUIRE(pc.electronVolt == Approx(1.602176565e-19));
    REQUIRE(pc.eV == Approx(pc.electronVolt));
    REQUIRE(pc.atomicMassUnit == Approx(1.660538921e-27));
    REQUIRE(pc.amu == Approx(pc.atomicMassUnit));
  }

  SECTION("CGS units") {
    PhysicalConstants<CGS> pc;
    REQUIRE(pc.avogadro == Approx(6.02214129e23));
    REQUIRE(pc.fineStructure == Approx(7.2973525698e-3));
    REQUIRE(pc.planck == Approx(6.62606957e-27));
    REQUIRE(pc.reducedPlanck == Approx(1.05457172e-27));
    REQUIRE(pc.gasConstant == Approx(83144621));
    REQUIRE(pc.boltzmann == Approx(1.3806488e-16));
    REQUIRE(pc.electronCharge == Approx(4.80320451e-10));
    REQUIRE(pc.speedOfLight == Approx(2.99792458e10));
    REQUIRE(pc.gravitationalConstant == Approx(6.67384e-8));
    REQUIRE(pc.accelerationFromGravity == Approx(9.80665e2));
    REQUIRE(pc.electronMass == Approx(9.10938291e-28));
    REQUIRE(pc.protonMass == Approx(1.672621777e-24));
    REQUIRE(pc.stefanBoltzmann == Approx(5.67037262e-5));
    REQUIRE(pc.faradayConstant == Approx(8.66742090e16));
    REQUIRE(pc.permeabilityOfVacuum == Approx(12.56637061));
    REQUIRE(pc.permittivityOfVacuum == Approx(0.079538483));
    REQUIRE(pc.classicalElectronRadius == Approx(2.81794033e-13));
    REQUIRE(pc.electronVolt == Approx(1.602176565e-12));
    REQUIRE(pc.atomicMassUnit == Approx(1.660538921e-24));
  }
}
