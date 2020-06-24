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
  static constexpr double Length = 1.;      // meter
  static constexpr double Mass = 1.;        // kilogram
  static constexpr double Time = 1.;        // second
  static constexpr double Temperature = 1.; // Kelvin
  static constexpr double Current = 1.;     // Amp
  static constexpr double Charge = 1.;      // Coulomb
  static constexpr double Capacitance = 1.; // Farad
  static constexpr double Angle = 1.;       // Radian
};

struct CGS {
  static constexpr double Length = 1.e2;                    // centimeter
  static constexpr double Mass = 1.e3;                      // gram
  static constexpr double Time = 1.;                        // second
  static constexpr double Temperature = 1.;                 // Kelvin
  static constexpr double Current = 1.e-1;                  // Biot
  static constexpr double Charge = 2.997924580e9;           // Statcoulomb
  static constexpr double Capacitance = 8.9831483395497e11; // Statfarad
  static constexpr double Angle = 1.;                       // Radian
};

// Defines and encapsulates physical and mathematical constants in a purely
// constexpr way, with support for multiple unit systems.
template <typename UNITSYSTEM>
class PhysicalConstants {
 private:
  static constexpr double Length = UNITSYSTEM::Length;
  static constexpr double Mass = UNITSYSTEM::Mass;
  static constexpr double Time = UNITSYSTEM::Time;
  static constexpr double Temperature = UNITSYSTEM::Temperature;
  static constexpr double Current = UNITSYSTEM::Current;
  static constexpr double Charge = UNITSYSTEM::Charge;
  static constexpr double Capacitance = UNITSYSTEM::Capacitance;
  static constexpr double Angle = UNITSYSTEM::Angle;
  static constexpr double Force = Mass * Length / (Time * Time);
  static constexpr double Energy = Force * Length;
  static constexpr double Power = Energy / Time;

 public:
  constexpr PhysicalConstants() {}

  // Avogadro constant (CODATA 2010 value)
  static constexpr double Avogadro = 6.02214129e23;
  static constexpr double Na = Avogadro;

  // Fine structure constant (CODATA 2010 value)
  static constexpr double FineStructure = 7.2973525698e-3;
  static constexpr double alpha = FineStructure;

  // Planck constant (CODATA 2010 value)
  static constexpr double Planck = 6.62606957e-34 * Energy * Time;
  static constexpr double h = Planck;

  // Reduced Planck constant
  static constexpr double ReducedPlanck = Planck / (2.0 * M_PI);
  static constexpr double hbar = ReducedPlanck;

  // Molar gas constant (CODATA 2010 value)
  static constexpr double GasConstant = 8.3144621 * Energy / Temperature;
  static constexpr double R = GasConstant;

  // Boltzmann constant (CODATA 2010 value)
  static constexpr double Boltzmann = 1.380648800e-23 * Energy / Temperature;
  static constexpr double kb = Boltzmann;

  // Electron charge (CODATA 2018 exact value)
  static constexpr double ElectronCharge = 1.602176565e-19 * Charge;
  static constexpr double qe = ElectronCharge;

  // Speed of light (CODATA 2018 exact value)
  static constexpr double SpeedOfLight = 2.99792458e8 * Length / Time;
  static constexpr double c = SpeedOfLight;

  // Gravitational constant (CODATA 2010 value)
  static constexpr double GravitationalConstant =
      6.67384e-11 * Length * Length * Length / (Mass * Time * Time);
  static constexpr double G = GravitationalConstant;

  // Standard acceleration of gravity (CODATA 2010 value)
  static constexpr double AccelerationFromGravity = 9.80665 * Length / (Time * Time);
  static constexpr double g = AccelerationFromGravity;

  // Electron rest mass (CODATA 2010 value)
  static constexpr double ElectronMass = 9.10938291e-31 * Mass;
  static constexpr double me = ElectronMass;

  // Proton rest mass (CODATA 2010 value)
  static constexpr double ProtonMass = 1.672621777e-27 * Mass;
  static constexpr double mp = ProtonMass;

  // Stefan-Boltzmann constant
  static constexpr double StefanBoltzmann = 2.0 * M_PI * M_PI * M_PI * M_PI * M_PI * kb *
                                            kb * kb * kb / (15.0 * h * h * h * c * c);
  static constexpr double sb = StefanBoltzmann;

  // Faraday constant
  static constexpr double FaradayConstant = 96485.33645957 * Capacitance;
  static constexpr double F = FaradayConstant;

  // Permeability of free space
  static constexpr double PermeabilityOfVacuum =
      4.0 * M_PI * 1.0e-7 * Force / (Current * Current);
  static constexpr double mu0 = PermeabilityOfVacuum;

  // Permittivity of free space
  static constexpr double PermittivityOfVacuum = 8.85418782e-12 * Capacitance / Length;
  static constexpr double eps0 = PermittivityOfVacuum;

  // Classical electron radius
  static constexpr double ClassicalElectronRadius = 2.81794033e-15 * Length;
  static constexpr double re = ClassicalElectronRadius;

  // Electron volt
  static constexpr double ElectronVolt = 1.602176565e-19 * Energy;
  static constexpr double eV = ElectronVolt;

  // Atomic mass unit (CODATA 2010 value)
  static constexpr double AtomicMassUnit = 1.660538921e-27 * Mass;
  static constexpr double amu = AtomicMassUnit;
};

// These can be removed for C++17 and up
template <typename T>
constexpr double PhysicalConstants<T>::Avogadro;
template <typename T>
constexpr double PhysicalConstants<T>::Na;
template <typename T>
constexpr double PhysicalConstants<T>::FineStructure;
template <typename T>
constexpr double PhysicalConstants<T>::alpha;
template <typename T>
constexpr double PhysicalConstants<T>::Planck;
template <typename T>
constexpr double PhysicalConstants<T>::h;
template <typename T>
constexpr double PhysicalConstants<T>::ReducedPlanck;
template <typename T>
constexpr double PhysicalConstants<T>::hbar;
template <typename T>
constexpr double PhysicalConstants<T>::GasConstant;
template <typename T>
constexpr double PhysicalConstants<T>::R;
template <typename T>
constexpr double PhysicalConstants<T>::Boltzmann;
template <typename T>
constexpr double PhysicalConstants<T>::kb;
template <typename T>
constexpr double PhysicalConstants<T>::ElectronCharge;
template <typename T>
constexpr double PhysicalConstants<T>::qe;
template <typename T>
constexpr double PhysicalConstants<T>::SpeedOfLight;
template <typename T>
constexpr double PhysicalConstants<T>::c;
template <typename T>
constexpr double PhysicalConstants<T>::GravitationalConstant;
template <typename T>
constexpr double PhysicalConstants<T>::G;
template <typename T>
constexpr double PhysicalConstants<T>::AccelerationFromGravity;
template <typename T>
constexpr double PhysicalConstants<T>::g;
template <typename T>
constexpr double PhysicalConstants<T>::ElectronMass;
template <typename T>
constexpr double PhysicalConstants<T>::me;
template <typename T>
constexpr double PhysicalConstants<T>::ProtonMass;
template <typename T>
constexpr double PhysicalConstants<T>::mp;
template <typename T>
constexpr double PhysicalConstants<T>::StefanBoltzmann;
template <typename T>
constexpr double PhysicalConstants<T>::sb;
template <typename T>
constexpr double PhysicalConstants<T>::FaradayConstant;
template <typename T>
constexpr double PhysicalConstants<T>::F;
template <typename T>
constexpr double PhysicalConstants<T>::PermeabilityOfVacuum;
template <typename T>
constexpr double PhysicalConstants<T>::mu0;
template <typename T>
constexpr double PhysicalConstants<T>::PermittivityOfVacuum;
template <typename T>
constexpr double PhysicalConstants<T>::eps0;
template <typename T>
constexpr double PhysicalConstants<T>::ClassicalElectronRadius;
template <typename T>
constexpr double PhysicalConstants<T>::re;
template <typename T>
constexpr double PhysicalConstants<T>::ElectronVolt;
template <typename T>
constexpr double PhysicalConstants<T>::eV;
template <typename T>
constexpr double PhysicalConstants<T>::AtomicMassUnit;
template <typename T>
constexpr double PhysicalConstants<T>::amu;

} // namespace constants

} // namespace parthenon

#endif // UTILS_CONSTANTS_HPP_
