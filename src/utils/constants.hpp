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

#include <string>

namespace constants {

  struct SI {
    static constexpr double Length = 1.;      // meter
    static constexpr double Mass = 1.;        // kilogram
    static constexpr double Time = 1.;        // second
    static constexpr double Temperature = 1.; // Kelvin
    static constexpr double Current = 1.;     // Amp
    static constexpr double Charge = 1.;      // Coulomb
    static constexpr double Capacitance = 1.; // Farad
    static constexpr double Angle = 1.;       // Radian
    static constexpr double Quantity = 1.;    // Mole
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
    static constexpr double Quantity = 1.;                    // Mole
  };

//==============================================================================
/*!
 * \class PhysicalConstants
 *
 * \brief Defines and encapsulates physical and mathematical constants in a
 *        purely constexpr way with support for multiple unit systems.
 */
//==============================================================================
template <typename UNITSYSTEM> class PhysicalConstants {
  private:
    static constexpr double Length = UNITSYSTEM::Length;
    static constexpr double Mass = UNITSYSTEM::Mass;
    static constexpr double Time = UNITSYSTEM::Time;
    static constexpr double Temperature = UNITSYSTEM::Temperature;
    static constexpr double Current = UNITSYSTEM::Current;
    static constexpr double Charge = UNITSYSTEM::Charge;
    static constexpr double Capacitance = UNITSYSTEM::Capacitance;
    static constexpr double Angle = UNITSYSTEM::Angle;
    static constexpr double Quantity = UNITSYSTEM::Quantity;
    static constexpr double Force = Mass * Length / (Time * Time);
    static constexpr double Energy = Force * Length;
    static constexpr double Power = Energy / Time;

  public:
    //! Default constructor
    static constexpr PhysicalConstants() {}

    //! Avogadro constant (CODATA 2010 value)
    static constexpr double Avogadro = 6.02214129e23*Quantity;
    static constexpr double Na = Avogadro;

    //! Fine structure constant (CODATA 2010 value)
    static constexpr double FineStructure = 7.2973525698e-3;
    static constexpr double alpha = FineStructure;

    //! Planck constant (CODATA 2010 value)
    static constexpr double Planck = 6.62606957e-34;
    static constexpr double h = Planck;

    //! Reduced Planck constant
    static constexpr double ReducedPlanck  = Planck/(2.*M_PI);
    static constexpr double hbar = ReducedPlanck;

    //! Molar gas constant (CODATA 2010 value)
    static constexpr double GasConstant = 8.3144621;

    //! Boltzmann constant (CODATA 2010 value)
    static constexpr double Boltzmann = 1.380648800e-23;


}

} // namespace constants

#endif //UTILS_CONSTANTS_HPP_
