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

struct BaseUnitSystem {
  const double length;
  const double mass;
  const double time;
};

struct SI : public BaseUnitSystem : BaseUnitSystem(1, 1, 1)  {
};

struct CGS : public BaseUnitSystem {};

// Conversion factors
/*struct SI {
  static const double length;
  static const double mass;
  static const double time;
  static const double temperature;
  static const double current;
  static const double charge;
  static const double capacitance;
};

struct CGS {
  static const double length;
  static const double mass;
  static const double time;
  static const double temperature;
  static const double current;
  static const double charge;
  static const double capacitance;
};*/

// Defines and encapsulates physical and mathematical constants with support for
// multiple unit systems.
class PhysicalConstants {
  /*protected:
   static double length;
   static double mass;
   static double time;
   static double Temperature;
   static double Current;
   static double charge;
   static double capacitance;
   static double force;
   static double energy;
   static double power;*/
  // protected:
  //  static double

 public:
  PhysicalConstants(BaseUnitSystem units_)
      : units(units_), atomicMassUnit(atomicMassUnit_SI_ * units.mass),
        amu(atomicMassUnit) {}

 private:
  double atomicMassUnit_SI_ = 1.660538921e-27;

  /*
  // Avogadro constant (CODATA 2010 value)
  static double Avogadro = 6.02214129e23;
  static double Na = Avogadro;

  // Fine structure constant (CODATA 2010 value)
  static double FineStructure = 7.2973525698e-3;
  static double alpha = FineStructure;

  // Planck constant (CODATA 2010 value)
  static double Planck = 6.62606957e-34 * Energy * Time;
  static double h = Planck;

  // Reduced Planck constant
  static double ReducedPlanck = Planck / (2.0 * M_PI);
  static double hbar = ReducedPlanck;

  // Molar gas constant (CODATA 2010 value)
  static double GasConstant = 8.3144621 * Energy / Temperature;
  static double R = GasConstant;

  // Boltzmann constant (CODATA 2010 value)
  static double Boltzmann = 1.380648800e-23 * Energy / Temperature;
  static double kb = Boltzmann;

  // Electron charge (CODATA 2018 exact value)
  static double ElectronCharge = 1.602176565e-19 * Charge;
  static double qe = ElectronCharge;

  // Speed of light (CODATA 2018 exact value)
  static double SpeedOfLight = 2.99792458e8 * Length / Time;
  static double c = SpeedOfLight;

  // Gravitational constant (CODATA 2010 value)
  static double GravitationalConstant =
      6.67384e-11 * Length * Length * Length / (Mass * Time * Time);
  static double G = GravitationalConstant;

  // Standard acceleration of gravity (CODATA 2010 value)
  static double AccelerationFromGravity = 9.80665 * Length / (Time * Time);
  static double g = AccelerationFromGravity;

  // Electron rest mass (CODATA 2010 value)
  static double ElectronMass = 9.10938291e-31 * Mass;
  static double me = ElectronMass;

  // Proton rest mass (CODATA 2010 value)
  static double ProtonMass = 1.672621777e-27 * Mass;
  static double mp = ProtonMass;

  // Stefan-Boltzmann constant
  static double StefanBoltzmann = 2.0 * M_PI * M_PI * M_PI * M_PI * M_PI * kb *
                                            kb * kb * kb / (15.0 * h * h * h * c * c);
  static double sb = StefanBoltzmann;

  // Faraday constant
  static double FaradayConstant = 96485.33645957 * Capacitance;
  static double F = FaradayConstant;

  // Permeability of free space
  static double PermeabilityOfVacuum =
      4.0 * M_PI * 1.0e-7 * Force / (Current * Current);
  static double mu0 = PermeabilityOfVacuum;

  // Permittivity of free space
  static double PermittivityOfVacuum = 8.85418782e-12 * Capacitance / Length;
  static double eps0 = PermittivityOfVacuum;

  // Classical electron radius
  static double ClassicalElectronRadius = 2.81794033e-15 * Length;
  static double re = ClassicalElectronRadius;
*/
  // Electron volt
  // const double electronVolt;// = 1.602176565e-19 * Energy;
  // const double eV;// = ElectronVolt;

  // Atomic mass unit (CODATA 2010 value)
  const double atomicMassUnit; // = 1.660538921e-27 * Mass;
  const double amu;            // = AtomicMassUnit;

 private:
  BaseUnitSystem units;
};

// template <typename T>
// const double PhysicalConstants<T>::atomicMassUnit = 1.660538921e-27 * T::mass;

// These can be removed for C++17 and up
/*template <typename T>
double PhysicalConstants<T>::Avogadro;
template <typename T>
double PhysicalConstants<T>::Na;
template <typename T>
double PhysicalConstants<T>::FineStructure;
template <typename T>
double PhysicalConstants<T>::alpha;
template <typename T>
double PhysicalConstants<T>::Planck;
template <typename T>
double PhysicalConstants<T>::h;
template <typename T>
double PhysicalConstants<T>::ReducedPlanck;
template <typename T>
double PhysicalConstants<T>::hbar;
template <typename T>
double PhysicalConstants<T>::GasConstant;
template <typename T>
double PhysicalConstants<T>::R;
template <typename T>
double PhysicalConstants<T>::Boltzmann;
template <typename T>
double PhysicalConstants<T>::kb;
template <typename T>
double PhysicalConstants<T>::ElectronCharge;
template <typename T>
double PhysicalConstants<T>::qe;
template <typename T>
double PhysicalConstants<T>::SpeedOfLight;
template <typename T>
double PhysicalConstants<T>::c;
template <typename T>
double PhysicalConstants<T>::GravitationalConstant;
template <typename T>
double PhysicalConstants<T>::G;
template <typename T>
double PhysicalConstants<T>::AccelerationFromGravity;
template <typename T>
double PhysicalConstants<T>::g;
template <typename T>
double PhysicalConstants<T>::ElectronMass;
template <typename T>
double PhysicalConstants<T>::me;
template <typename T>
double PhysicalConstants<T>::ProtonMass;
template <typename T>
double PhysicalConstants<T>::mp;
template <typename T>
double PhysicalConstants<T>::StefanBoltzmann;
template <typename T>
double PhysicalConstants<T>::sb;
template <typename T>
double PhysicalConstants<T>::FaradayConstant;
template <typename T>
double PhysicalConstants<T>::F;
template <typename T>
double PhysicalConstants<T>::PermeabilityOfVacuum;
template <typename T>
double PhysicalConstants<T>::mu0;
template <typename T>
double PhysicalConstants<T>::PermittivityOfVacuum;
template <typename T>
double PhysicalConstants<T>::eps0;
template <typename T>
double PhysicalConstants<T>::ClassicalElectronRadius;
template <typename T>
double PhysicalConstants<T>::re;
template <typename T>
double PhysicalConstants<T>::ElectronVolt;
template <typename T>
double PhysicalConstants<T>::eV;*/
// template <typename T>
// double PhysicalConstants<T>::AtomicMassUnit;
// template <typename T>
// double PhysicalConstants<T>::amu = PhysicalConstants<T>::atomicMassUnit;

} // namespace constants

} // namespace parthenon

#endif // UTILS_CONSTANTS_HPP_
