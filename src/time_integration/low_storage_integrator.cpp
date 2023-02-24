//========================================================================================
// (C) (or copyright) 2020-2023. Triad National Security, LLC. All rights reserved.
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

#include <string>
#include <vector>

#include "basic_types.hpp"
#include "parameter_input.hpp"
#include "staged_integrator.hpp"

namespace parthenon {

/*
 * These integrators are of the 2S form as described in
 * Ketchson, Jcomp 229 (2010) 1763-1773
 * See Equation 14.
 * They can be generalized to support more general methods with the
 * introduction of a delta term for a first averaging.
 * The form is also described in Section 3.2.3 of the Athena++ paper:
 * Stone et al., ApJS (2020) 249:4
 * See equations 11 through 15.
 */
LowStorageIntegrator::LowStorageIntegrator(ParameterInput *pin)
  : StagedIntegrator(pin->GetOrAddString("parthenon/time", "integrator", "rk2")) {
  if (name_ == "rk1") {
    nstages = 1;
    nbuffers = 1;
    delta.resize(nstages);
    beta.resize(nstages);
    gam0.resize(nstages);
    gam1.resize(nstages);

    delta[0] = 1.0;
    beta[0] = 1.0;
    gam0[0] = 0.0;
    gam1[0] = 1.0;
  } else if (name_ == "rk2") {
    nstages = 2;
    nbuffers = 2;
    delta.resize(nstages);
    beta.resize(nstages);
    gam0.resize(nstages);
    gam1.resize(nstages);

    delta[0] = 1.0;
    beta[0] = 1.0;
    gam0[0] = 0.0;
    gam1[0] = 1.0;

    delta[1] = 0.0;
    beta[1] = 0.5;
    gam0[1] = 0.5;
    gam1[1] = 0.5;
  } else if (name_ == "vl2") {
    nstages = 2;
    nbuffers = 2;
    delta.resize(nstages);
    beta.resize(nstages);
    gam0.resize(nstages);
    gam1.resize(nstages);

    delta[0] = 1.0;
    beta[0] = 0.5;
    gam0[0] = 0.0;
    gam1[0] = 1.0;

    delta[1] = 0.0;
    beta[1] = 1.0;
    gam0[1] = 0.0;
    gam1[1] = 1.0;
  } else if (name_ == "rk3") {
    nstages = 3;
    nbuffers = 2;
    delta.resize(nstages);
    beta.resize(nstages);
    gam0.resize(nstages);
    gam1.resize(nstages);

    delta[0] = 1.0;
    beta[0] = 1.0;
    gam0[0] = 0.0;
    gam1[0] = 1.0;

    delta[1] = 0.0;
    beta[1] = 0.25;
    gam0[1] = 0.25;
    gam1[1] = 0.75;

    delta[2] = 0.0;
    beta[2] = 2.0 / 3.0;
    gam0[2] = 2.0 / 3.0;
    gam1[2] = 1.0 / 3.0;
  } else if (name_ == "rk4") {
    // From Table 4 of Ketchson, Jcomp 229 (2010) 1763-1773
    nstages = 5;
    nbuffers = 2;
    delta.resize(nstages);
    beta.resize(nstages);
    gam0.resize(nstages);
    gam1.resize(nstages);

    delta[0] = 1.0;
    beta[0] = 0.357534921136978;
    gam0[0] = 0.0;
    gam1[0] = 1.0;

    delta[1] = 0.0;
    beta[1] = 2.364680399061355;
    gam0[1] = -3.666545952121251;
    gam1[1] = 4.666545952121251;

    delta[2] = 0.0;
    beta[2] = 0.016239790859612;
    gam0[2] = 0.035802535958088;
    gam1[2] = 0.964197464041912;

    delta[3] = 0.0;
    beta[3] = 0.498173799587251;
    gam0[3] = 4.398279365655791;
    gam1[3] = -3.398279365655790;

    delta[4] = 0.0;
    beta[4] = 0.433334235669763;
    gam0[4] = 0.770411587328417;
    gam1[4] = 0.229588412671583;
  } else {
    throw std::invalid_argument("Invalid selection for the time integrator: " +
                                name_);
  }
  MakePeriodicNames_(buffer_name, nbuffers);
  MakePeriodicNames_(stage_name, nstages);
}

} // namespace parthenon
