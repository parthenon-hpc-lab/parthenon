//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020-2023 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
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

#ifndef OUTPUTS_PARTHENON_XDMF_HPP_
#define OUTPUTS_PARTHENON_XDMF_HPP_

// C++ includes
#include <string>
#include <vector>

namespace parthenon {
// forward declarations
namespace HDF5 {
  struct SimTime;
  struct VarInfo;
  struct SwarmVarInfo;
} // namespace HDF5

namespace XDMF {
void genXDMF(std::string hdfFile, Mesh *pm, SimTime *tm, int nx1, int nx2, int nx3,
             const std::vector<VarInfo> &var_list);
} // namespace XDMF
} // namespace parthenon

#endif // OUTPUTS_PARTHENON_XDMF_HPP_
