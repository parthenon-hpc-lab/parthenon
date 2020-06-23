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

using parthenon::constants::PhysicalConstants;
using parthenon::constants::SI;
using parthenon::constants::CGS;

TEST_CASE("Physical constants", "[SI, CGS]") {
  SECTION("SI units") {
    PhysicalConstants<SI> pc;
    REQUIRE(pc.Avogadro == Approx(6.02214129e23));
    REQUIRE(pc.Na == Approx(pc.Avogadro));
    REQUIRE(pc.FineStructure == Approx(7.2973525698e-3));
    REQUIRE(pc.alpha == Approx(pc.FineStructure));
    REQUIRE(pc.Planck == Approx(6.62606957e-34));
    REQUIRE(pc.h == Approx(pc.Planck));
    REQUIRE(pc.ReducedPlanck == Approx(1.05457172e-34));
    REQUIRE(pc.hbar == Approx(pc.ReducedPlanck));
    REQUIRE(pc.GasConstant == Approx(8.3144621));
    REQUIRE(pc.R == Approx(pc.GasConstant));
    REQUIRE(pc.Boltzmann == Approx(1.380648800e-23));
    REQUIRE(pc.kb == Approx(pc.Boltzmann));
    REQUIRE(pc.ElectronCharge == Approx(1.602176565e-19));
    REQUIRE(pc.qe == Approx(pc.ElectronCharge));
    REQUIRE(pc.SpeedOfLight == Approx(2.99792458e8));
    REQUIRE(pc.c == Approx(pc.SpeedOfLight));
    REQUIRE(pc.GravitationalConstant == Approx(6.67384e-11));
    REQUIRE(pc.G == Approx(pc.GravitationalConstant));
    REQUIRE(pc.AccelerationFromGravity == Approx(9.80665));
    REQUIRE(pc.g == Approx(pc.AccelerationFromGravity));
    REQUIRE(pc.ElectronMass == Approx(9.10938291e-31));
    REQUIRE(pc.me == Approx(pc.ElectronMass));
    REQUIRE(pc.ProtonMass == Approx(1.672621777e-27));
    REQUIRE(pc.mp == Approx(pc.ProtonMass));
    REQUIRE(pc.StefanBoltzmann == Approx(5.67037262e-8));
    REQUIRE(pc.sb == Approx(pc.StefanBoltzmann));
    REQUIRE(pc.FaradayConstant == Approx(96485.33645957));
    REQUIRE(pc.F == Approx(pc.FaradayConstant));
    REQUIRE(pc.PermeabilityOfVacuum == Approx(1.25663706e-06));
    REQUIRE(pc.mu0 == Approx(pc.PermeabilityOfVacuum));
    REQUIRE(pc.PermittivityOfVacuum == Approx(8.85418782e-12));
    REQUIRE(pc.eps0 == Approx(pc.PermittivityOfVacuum));
    REQUIRE(pc.ClassicalElectronRadius == Approx(2.81794033e-15));
    REQUIRE(pc.re == Approx(pc.ClassicalElectronRadius));
    REQUIRE(pc.ElectronVolt == Approx(1.602176565e-19));
    REQUIRE(pc.eV == Approx(pc.ElectronVolt));
    REQUIRE(pc.AtomicMassUnit == Approx(1.660538921e-27));
    REQUIRE(pc.amu == Approx(pc.AtomicMassUnit));
  }

  SECTION("CGS units") {
    PhysicalConstants<CGS> pc;
    REQUIRE(pc.Avogadro == Approx(6.02214129e23));
    REQUIRE(pc.FineStructure == Approx(7.2973525698e-3));
    REQUIRE(pc.Planck == Approx(6.62606957e-27));
    REQUIRE(pc.ReducedPlanck == Approx(1.05457172e-27));
    REQUIRE(pc.GasConstant == Approx(83144621));
    REQUIRE(pc.Boltzmann == Approx(1.3806488e-16));
    REQUIRE(pc.ElectronCharge == Approx(4.80320451e-10));
    REQUIRE(pc.SpeedOfLight == Approx(2.99792458e10));
    REQUIRE(pc.GravitationalConstant == Approx(6.67384e-8));
    REQUIRE(pc.AccelerationFromGravity == Approx(9.80665e2));
    REQUIRE(pc.ElectronMass == Approx(9.10938291e-28));
    REQUIRE(pc.ProtonMass == Approx(1.672621777e-24));
    REQUIRE(pc.StefanBoltzmann == Approx(5.67037262e-5));
    //REQUIRE(pc.FaradayConstant == Approx(8.66742090e16));
    REQUIRE(pc.PermeabilityOfVacuum == Approx(12.56637061));
    REQUIRE(pc.PermittivityOfVacuum == Approx(0.079538483));
    REQUIRE(pc.ClassicalElectronRadius == Approx(2.81794033e-13));
    REQUIRE(pc.ElectronVolt == Approx(1.602176565e-12));
    REQUIRE(pc.AtomicMassUnit == Approx(1.660538921e-24));
  }
}
