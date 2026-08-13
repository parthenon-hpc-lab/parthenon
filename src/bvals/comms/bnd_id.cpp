//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2024 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2024. Triad National Security, LLC. All rights reserved.
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

#include <algorithm>
#include <cstdio>
#include <iostream> // debug
#include <memory>
#include <string>
#include <vector>

#include "basic_types.hpp"
#include "bvals/comms/bnd_id.hpp"
#include "bvals/comms/bvals_utils.hpp"
#include "bvals/neighbor_block.hpp"
#include "config.hpp"
#include "globals.hpp"
#include "interface/state_descriptor.hpp"
#include "interface/variable.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/domain.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "mesh/meshblock.hpp"
#include "prolong_restrict/prolong_restrict.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

BndId BndId::GetSend(MeshBlock *pmb, const NeighborBlock &nb,
                     std::shared_ptr<Variable<Real>> v, BoundaryType b_type,
                     int partition, int start_idx) {
  // TODO(LFR): This needs to be fixed for unique buffer ids
  auto [send_gid, recv_gid, vlabel, loc, extra_id] = SendKey(pmb, nb, v, b_type, 0);
  BndId out;
  out.send_gid() = send_gid;
  out.recv_gid() = recv_gid;
  out.loc_idx() = loc;
  out.var_id() = v->GetUniqueID();
  out.extra_id() = extra_id;
  out.rank_send() = Globals::my_rank;
  out.rank_recv() = nb.rank;
  out.partition() = partition;
  out.size() = BndInfo::GetSendBndInfo(pmb, nb, v, nullptr).size();
  out.start_idx() = start_idx;
  return out;
}

void BndId::PrintInfo(const std::string &start) {
  printf("%s var %s (%i -> %i) starting at %i with size %i (Total combined buffer size = "
         "%li, buffer size = %li, buf_allocated = %i) [rank = %i]\n",
         start.c_str(), Variable<Real>::GetLabel(var_id()).c_str(), send_gid(),
         recv_gid(), start_idx(), size(), coalesced_buf.size(), buf.size(), buf_allocated,
         Globals::my_rank);
}

} // namespace parthenon
