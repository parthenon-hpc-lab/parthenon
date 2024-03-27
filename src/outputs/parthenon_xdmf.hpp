//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020-2023 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2023. Triad National Security, LLC. All rights reserved.
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

#ifndef OUTPUTS_PARTHENON_XDMF_HPP_
#define OUTPUTS_PARTHENON_XDMF_HPP_

// C++ includes
#include <string>
#include <vector>

#include "basic_types.hpp"
#include "outputs/output_utils.hpp"

namespace parthenon {
// forward declarations
namespace XDMF {
void genXDMF(std::string hdfFile, Mesh *pm, SimTime *tm, IndexDomain domain, int nx1,
             int nx2, int nx3, const std::vector<OutputUtils::VarInfo> &var_list,
             const OutputUtils::AllSwarmInfo &all_swarm_info);
} // namespace XDMF
} // namespace parthenon

#endif // OUTPUTS_PARTHENON_XDMF_HPP_
