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

#ifndef UTILS_CONSTANTS_HPP_
#define UTILS_CONSTANTS_HPP_

#include <cmath>

namespace parthenon {
namespace constants {

// Conversion factors from the SI unit system
struct SI {
  static constexpr double length = 1.;      // meter
  static constexpr double mass = 1.;        // kilogram
  static constexpr double time = 1.;        // second
  static constexpr double temperature = 1.; // Kelvin
  static constexpr double current = 1.;     // Amp
  static constexpr double charge = 1.;      // Coulomb
  static constexpr double capacitance = 1.; // Farad
  static constexpr double angle = 1.;       // Radian
};

struct CGS {
  static constexpr double length = 1.e2;                    // centimeter
  static constexpr double mass = 1.e3;                      // gram
  static constexpr double time = 1.;                        // second
  static constexpr double temperature = 1.;                 // Kelvin
  static constexpr double current = 1.e-1;                  // Biot
  static constexpr double charge = 2.997924580e9;           // Statcoulomb
  static constexpr double capacitance = 8.9831483395497e11; // Statfarad
  static constexpr double angle = 1.;                       // Radian
};

// Defines and encapsulates physical and mathematical constants in a purely
// constexpr way, with support for multiple unit systems.
template <typename UNITSYSTEM>
class PhysicalConstants {
 protected:
  static constexpr double length = UNITSYSTEM::length;
  static constexpr double mass = UNITSYSTEM::mass;
  static constexpr double time = UNITSYSTEM::time;
  static constexpr double temperature = UNITSYSTEM::temperature;
  static constexpr double current = UNITSYSTEM::current;
  static constexpr double charge = UNITSYSTEM::charge;
  static constexpr double capacitance = UNITSYSTEM::capacitance;
  static constexpr double angle = UNITSYSTEM::angle;

  // Derived unit conversions
  static constexpr double force = mass * length / (time * time);
  static constexpr double energy = force * length;
  static constexpr double power = energy / time;

 public:
  constexpr PhysicalConstants() {}

  // Avogadro constant (CODATA 2010 value)
  static constexpr double avogadro = 6.02214129e23;
  static constexpr double na = avogadro;

  // Fine structure constant (CODATA 2010 value)
  static constexpr double fineStructure = 7.2973525698e-3;
  static constexpr double alpha = fineStructure;

  // Planck constant (CODATA 2010 value)
  static constexpr double planck = 6.62606957e-34 * energy * time;
  static constexpr double h = planck;

  // Reduced Planck constant
  static constexpr double reducedPlanck = planck / (2.0 * M_PI);
  static constexpr double hbar = reducedPlanck;

  // Molar gas constant (CODATA 2010 value)
  static constexpr double gasConstant = 8.3144621 * energy / temperature;
  static constexpr double rGas = gasConstant;

  // Boltzmann constant (CODATA 2010 value)
  static constexpr double boltzmann = 1.380648800e-23 * energy / temperature;
  static constexpr double kb = boltzmann;

  // Electron charge (CODATA 2018 exact value)
  static constexpr double electronCharge = 1.602176565e-19 * charge;
  static constexpr double qe = electronCharge;

  // Speed of light (CODATA 2018 exact value)
  static constexpr double speedOfLight = 2.99792458e8 * length / time;
  static constexpr double c = speedOfLight;

  // Gravitational constant (CODATA 2010 value)
  static constexpr double gravitationalConstant =
      6.67384e-11 * length * length * length / (mass * time * time);
  static constexpr double gNewt = gravitationalConstant;

  // Standard acceleration of gravity (CODATA 2010 value)
  static constexpr double accelerationFromGravity = 9.80665 * length / (time * time);
  static constexpr double gAccel = accelerationFromGravity;

  // Electron rest mass (CODATA 2010 value)
  static constexpr double electronMass = 9.10938291e-31 * mass;
  static constexpr double me = electronMass;

  // Proton rest mass (CODATA 2010 value)
  static constexpr double protonMass = 1.672621777e-27 * mass;
  static constexpr double mp = protonMass;

  // Stefan-Boltzmann constant
  static constexpr double stefanBoltzmann = 2.0 * M_PI * M_PI * M_PI * M_PI * M_PI * kb *
                                            kb * kb * kb / (15.0 * h * h * h * c * c);
  static constexpr double sb = stefanBoltzmann;

  // Faraday constant
  static constexpr double faradayConstant = 96485.33645957 * capacitance;
  static constexpr double faraday = faradayConstant;

  // Permeability of free space
  static constexpr double permeabilityOfVacuum =
      4.0 * M_PI * 1.0e-7 * force / (current * current);
  static constexpr double mu0 = permeabilityOfVacuum;

  // Permittivity of free space
  static constexpr double permittivityOfVacuum = 8.85418782e-12 * capacitance / length;
  static constexpr double eps0 = permittivityOfVacuum;

  // Classical electron radius
  static constexpr double classicalElectronRadius = 2.81794033e-15 * length;
  static constexpr double re = classicalElectronRadius;

  // Electron volt
  static constexpr double electronVolt = 1.602176565e-19 * energy;
  static constexpr double eV = electronVolt;

  // Atomic mass unit (CODATA 2010 value)
  static constexpr double atomicMassUnit = 1.660538921e-27 * mass;
  static constexpr double amu = atomicMassUnit;
};

// These can be removed for C++17 and up
template <typename T>
constexpr double PhysicalConstants<T>::avogadro;
template <typename T>
constexpr double PhysicalConstants<T>::na;
template <typename T>
constexpr double PhysicalConstants<T>::fineStructure;
template <typename T>
constexpr double PhysicalConstants<T>::alpha;
template <typename T>
constexpr double PhysicalConstants<T>::planck;
template <typename T>
constexpr double PhysicalConstants<T>::h;
template <typename T>
constexpr double PhysicalConstants<T>::reducedPlanck;
template <typename T>
constexpr double PhysicalConstants<T>::hbar;
template <typename T>
constexpr double PhysicalConstants<T>::gasConstant;
template <typename T>
constexpr double PhysicalConstants<T>::rGas;
template <typename T>
constexpr double PhysicalConstants<T>::boltzmann;
template <typename T>
constexpr double PhysicalConstants<T>::kb;
template <typename T>
constexpr double PhysicalConstants<T>::electronCharge;
template <typename T>
constexpr double PhysicalConstants<T>::qe;
template <typename T>
constexpr double PhysicalConstants<T>::speedOfLight;
template <typename T>
constexpr double PhysicalConstants<T>::c;
template <typename T>
constexpr double PhysicalConstants<T>::gravitationalConstant;
template <typename T>
constexpr double PhysicalConstants<T>::gNewt;
template <typename T>
constexpr double PhysicalConstants<T>::accelerationFromGravity;
template <typename T>
constexpr double PhysicalConstants<T>::gAccel;
template <typename T>
constexpr double PhysicalConstants<T>::electronMass;
template <typename T>
constexpr double PhysicalConstants<T>::me;
template <typename T>
constexpr double PhysicalConstants<T>::protonMass;
template <typename T>
constexpr double PhysicalConstants<T>::mp;
template <typename T>
constexpr double PhysicalConstants<T>::stefanBoltzmann;
template <typename T>
constexpr double PhysicalConstants<T>::sb;
template <typename T>
constexpr double PhysicalConstants<T>::faradayConstant;
template <typename T>
constexpr double PhysicalConstants<T>::faraday;
template <typename T>
constexpr double PhysicalConstants<T>::permeabilityOfVacuum;
template <typename T>
constexpr double PhysicalConstants<T>::mu0;
template <typename T>
constexpr double PhysicalConstants<T>::permittivityOfVacuum;
template <typename T>
constexpr double PhysicalConstants<T>::eps0;
template <typename T>
constexpr double PhysicalConstants<T>::classicalElectronRadius;
template <typename T>
constexpr double PhysicalConstants<T>::re;
template <typename T>
constexpr double PhysicalConstants<T>::electronVolt;
template <typename T>
constexpr double PhysicalConstants<T>::eV;
template <typename T>
constexpr double PhysicalConstants<T>::atomicMassUnit;
template <typename T>
constexpr double PhysicalConstants<T>::amu;

} // namespace constants

} // namespace parthenon

#endif // UTILS_CONSTANTS_HPP_
