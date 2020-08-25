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

TEST_CASE("Physical constants", "[SI][CGS]") {
  SECTION("SI units") {
    PhysicalConstants<SI> pc;
    REQUIRE(pc.avogadro == Approx(6.02214129e23));
    REQUIRE(pc.na == Approx(pc.avogadro));
    REQUIRE(pc.fine_structure == Approx(7.2973525698e-3));
    REQUIRE(pc.alpha == Approx(pc.fine_structure));
    REQUIRE(pc.planck == Approx(6.62606957e-34));
    REQUIRE(pc.h == Approx(pc.planck));
    REQUIRE(pc.reduced_planck == Approx(1.05457172e-34));
    REQUIRE(pc.hbar == Approx(pc.reduced_planck));
    REQUIRE(pc.gas_constant == Approx(8.3144621));
    REQUIRE(pc.r_gas == Approx(pc.gas_constant));
    REQUIRE(pc.boltzmann == Approx(1.380648800e-23));
    REQUIRE(pc.kb == Approx(pc.boltzmann));
    REQUIRE(pc.electron_charge == Approx(1.602176565e-19));
    REQUIRE(pc.qe == Approx(pc.electron_charge));
    REQUIRE(pc.speed_of_light == Approx(2.99792458e8));
    REQUIRE(pc.c == Approx(pc.speed_of_light));
    REQUIRE(pc.gravitational_constant == Approx(6.67384e-11));
    REQUIRE(pc.g_newt == Approx(pc.gravitational_constant));
    REQUIRE(pc.acceleration_from_gravity == Approx(9.80665));
    REQUIRE(pc.g_accel == Approx(pc.acceleration_from_gravity));
    REQUIRE(pc.electron_mass == Approx(9.10938291e-31));
    REQUIRE(pc.me == Approx(pc.electron_mass));
    REQUIRE(pc.proton_mass == Approx(1.672621777e-27));
    REQUIRE(pc.mp == Approx(pc.proton_mass));
    REQUIRE(pc.stefan_boltzmann == Approx(5.67037262e-8));
    REQUIRE(pc.sb == Approx(pc.stefan_boltzmann));
    REQUIRE(pc.faraday_constant == Approx(96485.33645957));
    REQUIRE(pc.faraday == Approx(pc.faraday_constant));
    REQUIRE(pc.vacuum_permeability == Approx(1.25663706e-06));
    REQUIRE(pc.mu0 == Approx(pc.vacuum_permeability));
    REQUIRE(pc.vacuum_permittivity == Approx(8.85418782e-12));
    REQUIRE(pc.eps0 == Approx(pc.vacuum_permittivity));
    REQUIRE(pc.classical_electron_radius == Approx(2.81794033e-15));
    REQUIRE(pc.re == Approx(pc.classical_electron_radius));
    REQUIRE(pc.electron_volt == Approx(1.602176565e-19));
    REQUIRE(pc.eV == Approx(pc.electron_volt));
    REQUIRE(pc.atomic_mass_unit == Approx(1.660538921e-27));
    REQUIRE(pc.amu == Approx(pc.atomic_mass_unit));
  }

  SECTION("CGS units") {
    PhysicalConstants<CGS> pc;
    REQUIRE(pc.avogadro == Approx(6.02214129e23));
    REQUIRE(pc.fine_structure == Approx(7.2973525698e-3));
    REQUIRE(pc.planck == Approx(6.62606957e-27));
    REQUIRE(pc.reduced_planck == Approx(1.05457172e-27));
    REQUIRE(pc.gas_constant == Approx(83144621));
    REQUIRE(pc.boltzmann == Approx(1.3806488e-16));
    REQUIRE(pc.electron_charge == Approx(4.80320451e-10));
    REQUIRE(pc.speed_of_light == Approx(2.99792458e10));
    REQUIRE(pc.gravitational_constant == Approx(6.67384e-8));
    REQUIRE(pc.acceleration_from_gravity == Approx(9.80665e2));
    REQUIRE(pc.electron_mass == Approx(9.10938291e-28));
    REQUIRE(pc.proton_mass == Approx(1.672621777e-24));
    REQUIRE(pc.stefan_boltzmann == Approx(5.67037262e-5));
    REQUIRE(pc.faraday_constant == Approx(8.66742090e16));
    REQUIRE(pc.vacuum_permeability == Approx(12.56637061));
    REQUIRE(pc.vacuum_permittivity == Approx(0.079538483));
    REQUIRE(pc.classical_electron_radius == Approx(2.81794033e-13));
    REQUIRE(pc.electron_volt == Approx(1.602176565e-12));
    REQUIRE(pc.atomic_mass_unit == Approx(1.660538921e-24));
  }
}
