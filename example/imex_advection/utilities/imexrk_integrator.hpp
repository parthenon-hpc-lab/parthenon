//========================================================================================
// (C) (or copyright) 2020-2025. Triad National Security, LLC. All rights
// reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001
// for Los Alamos National Laboratory (LANL), which is operated by Triad
// National Security, LLC for the U.S. Department of Energy/National Nuclear
// Security Administration. All rights in the program are reserved by Triad
// National Security, LLC, and the U.S. Department of Energy/National Nuclear
// Security Administration. The Government is granted for itself and others
// acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license
// in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do
// so.
//========================================================================================

#ifndef SRC_IMEXRK_INTEGRATOR_HPP_
#define SRC_IMEXRK_INTEGRATOR_HPP_

#include <memory>
#include <vector>

#include <parthenon/parthenon.hpp>

namespace scalar_imex {
using namespace parthenon::package::prelude;

class IMEXRKIntegrator {
 public:
  explicit IMEXRKIntegrator(const std::string &name);
  IMEXRKIntegrator() : IMEXRKIntegrator("SSP2-(2,2,2)") {}
  explicit IMEXRKIntegrator(ParameterInput *pin)
      : IMEXRKIntegrator(
            pin->GetOrAddString("parthenon/time", "integrator", "SSP2-(2,2,2)")) {}

  // To conform with other integrators
  int nstages;
  Real dt;

  Real at(int i, int j) const { return at_[i - 1][j - 1]; }
  Real a(int i, int j) const { return a_[i - 1][j - 1]; }

  Real bt(int i) const { return bt_[i - 1]; }
  Real b(int i) const { return b_[i - 1]; }

  Real ct(int i) const { return ct_[i - 1]; }
  Real c(int i) const { return c_[i - 1]; }

  std::string GetStageName(int i) const { return stage_names_[i - 1]; }

 private:
  std::vector<std::vector<Real>> at_, a_;
  std::vector<Real> bt_, b_;
  std::vector<Real> ct_, c_;
  std::vector<std::string> stage_names_;
};

} // namespace scalar_imex

#endif // SRC_IMEXRK_INTEGRATOR_HPP_
