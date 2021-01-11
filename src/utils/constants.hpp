//========================================================================================
// (C) (or copyright) 2021. Triad National Security, LLC. All rights reserved.
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
  static constexpr double fine_structure = 7.2973525698e-3;
  static constexpr double alpha = fine_structure;

  // Planck constant (CODATA 2010 value)
  static constexpr double planck = 6.62606957e-34 * energy * time;
  static constexpr double h = planck;

  // Reduced Planck constant
  static constexpr double reduced_planck = planck / (2.0 * M_PI);
  static constexpr double hbar = reduced_planck;

  // Molar gas constant (CODATA 2010 value)
  static constexpr double gas_constant = 8.3144621 * energy / temperature;
  static constexpr double r_gas = gas_constant;

  // Boltzmann constant (CODATA 2010 value)
  static constexpr double boltzmann = 1.380648800e-23 * energy / temperature;
  static constexpr double kb = boltzmann;

  // Electron charge (CODATA 2018 exact value)
  static constexpr double electron_charge = 1.602176565e-19 * charge;
  static constexpr double qe = electron_charge;

  // Speed of light (CODATA 2018 exact value)
  static constexpr double speed_of_light = 2.99792458e8 * length / time;
  static constexpr double c = speed_of_light;

  // Gravitational constant (CODATA 2010 value)
  static constexpr double gravitational_constant =
      6.67384e-11 * length * length * length / (mass * time * time);
  static constexpr double g_newt = gravitational_constant;

  // Standard acceleration of gravity (CODATA 2010 value)
  static constexpr double acceleration_from_gravity = 9.80665 * length / (time * time);
  static constexpr double g_accel = acceleration_from_gravity;

  // Electron rest mass (CODATA 2010 value)
  static constexpr double electron_mass = 9.10938291e-31 * mass;
  static constexpr double me = electron_mass;

  // Proton rest mass (CODATA 2010 value)
  static constexpr double proton_mass = 1.672621777e-27 * mass;
  static constexpr double mp = proton_mass;

  // Stefan-Boltzmann constant
  static constexpr double stefan_boltzmann = 2.0 * M_PI * M_PI * M_PI * M_PI * M_PI * kb *
                                             kb * kb * kb / (15.0 * h * h * h * c * c);
  static constexpr double sb = stefan_boltzmann;

  // Faraday constant
  static constexpr double faraday_constant = 96485.33645957 * capacitance;
  static constexpr double faraday = faraday_constant;

  // Permeability of free space
  static constexpr double vacuum_permeability =
      4.0 * M_PI * 1.0e-7 * force / (current * current);
  static constexpr double mu0 = vacuum_permeability;

  // Permittivity of free space
  static constexpr double vacuum_permittivity = 8.85418782e-12 * capacitance / length;
  static constexpr double eps0 = vacuum_permittivity;

  // Classical electron radius
  static constexpr double classical_electron_radius = 2.81794033e-15 * length;
  static constexpr double re = classical_electron_radius;

  // Electron volt
  static constexpr double electron_volt = 1.602176565e-19 * energy;
  static constexpr double eV = electron_volt;

  // Atomic mass unit (CODATA 2010 value)
  static constexpr double atomic_mass_unit = 1.660538921e-27 * mass;
  static constexpr double amu = atomic_mass_unit;
};

// These can be removed for C++17 and up
template <typename T>
constexpr double PhysicalConstants<T>::avogadro;
template <typename T>
constexpr double PhysicalConstants<T>::na;
template <typename T>
constexpr double PhysicalConstants<T>::fine_structure;
template <typename T>
constexpr double PhysicalConstants<T>::alpha;
template <typename T>
constexpr double PhysicalConstants<T>::planck;
template <typename T>
constexpr double PhysicalConstants<T>::h;
template <typename T>
constexpr double PhysicalConstants<T>::reduced_planck;
template <typename T>
constexpr double PhysicalConstants<T>::hbar;
template <typename T>
constexpr double PhysicalConstants<T>::gas_constant;
template <typename T>
constexpr double PhysicalConstants<T>::r_gas;
template <typename T>
constexpr double PhysicalConstants<T>::boltzmann;
template <typename T>
constexpr double PhysicalConstants<T>::kb;
template <typename T>
constexpr double PhysicalConstants<T>::electron_charge;
template <typename T>
constexpr double PhysicalConstants<T>::qe;
template <typename T>
constexpr double PhysicalConstants<T>::speed_of_light;
template <typename T>
constexpr double PhysicalConstants<T>::c;
template <typename T>
constexpr double PhysicalConstants<T>::gravitational_constant;
template <typename T>
constexpr double PhysicalConstants<T>::g_newt;
template <typename T>
constexpr double PhysicalConstants<T>::acceleration_from_gravity;
template <typename T>
constexpr double PhysicalConstants<T>::g_accel;
template <typename T>
constexpr double PhysicalConstants<T>::electron_mass;
template <typename T>
constexpr double PhysicalConstants<T>::me;
template <typename T>
constexpr double PhysicalConstants<T>::proton_mass;
template <typename T>
constexpr double PhysicalConstants<T>::mp;
template <typename T>
constexpr double PhysicalConstants<T>::stefan_boltzmann;
template <typename T>
constexpr double PhysicalConstants<T>::sb;
template <typename T>
constexpr double PhysicalConstants<T>::faraday_constant;
template <typename T>
constexpr double PhysicalConstants<T>::faraday;
template <typename T>
constexpr double PhysicalConstants<T>::vacuum_permeability;
template <typename T>
constexpr double PhysicalConstants<T>::mu0;
template <typename T>
constexpr double PhysicalConstants<T>::vacuum_permittivity;
template <typename T>
constexpr double PhysicalConstants<T>::eps0;
template <typename T>
constexpr double PhysicalConstants<T>::classical_electron_radius;
template <typename T>
constexpr double PhysicalConstants<T>::re;
template <typename T>
constexpr double PhysicalConstants<T>::electron_volt;
template <typename T>
constexpr double PhysicalConstants<T>::eV;
template <typename T>
constexpr double PhysicalConstants<T>::atomic_mass_unit;
template <typename T>
constexpr double PhysicalConstants<T>::amu;

} // namespace constants

} // namespace parthenon

#endif // UTILS_CONSTANTS_HPP_
