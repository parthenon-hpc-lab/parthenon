//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2026 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================

// This file was made in part with generative AI.

#ifndef UTILS_CALC_SPECTRUM_HPP_
#define UTILS_CALC_SPECTRUM_HPP_

#include <string>
#include <vector>

#include "parthenon_arrays.hpp"

namespace parthenon {

class Mesh;

// Computes the shell-averaged power spectrum of the requested components of
// var_name on a uniform mesh.  Returns a device-side array shaped [num_bins, 3]:
//   col 0: power sum, col 1: wavenumber sum, col 2: bin count
// num_bins = ceil(k_max) + 1.  An MPI_Reduce to rank 0 is performed internally,
// so only rank 0 holds meaningful data on return.
parthenon::ParArray2D<Real> CalcSpectrum(Mesh *pm, const std::string &var_name,
                                         const std::vector<int> &components);

} // namespace parthenon

#endif // UTILS_CALC_SPECTRUM_HPP_
