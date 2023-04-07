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

#include <map>
#include <string>
#include <vector>

#include "globals.hpp"
#include "interface/metadata.hpp"
#include "interface/swarm.hpp"
#include "interface/swarm_container.hpp"
#include "interface/variable.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "outputs/output_utils.hpp"

namespace parthenon {
namespace OutputUtils {

// TODO(JMM): Could probably reduce boiler plate/code duplication
// below with some more clever templating, but I don't care.
AllSwarmInfo::AllSwarmInfo(BlockList_t &block_list,
                           const std::vector<std::string> &swarmnames,
                           const std::vector<std::string> &varnames,
                           bool is_restart) {
  for (auto &pmb : block_list) {
    auto &swarm_container = pmb->swarm_data.Get();
    swarm_container->DefragAll();
    if (is_restart) {
      using FC = parthenon::Metadata::FlagCollection;
      auto flags =
        FC({parthenon::Metadata::Independent, parthenon::Metadata::Restart}, true);
      auto swarms = swarm_container->GetSwarmsByFlag(flags);
      for (auto &swarm : swarms) {
        auto swarmname = swarm->label();
        auto &info = all_info[swarmname];
        std::size_t count = swarm->GetNumActive();
        std::size_t offset = (info.offsets.size() > 0) ? info.offsets.back() : 0;
        offset += (info.counts.size() > 0) ? info.counts.back() : 0;
        info.counts.push_back(count);
        info.offsets.push_back(offset);
        info.count_on_rank += count;
        for (const auto &var : swarm->GetVariableVector<int>()) {
          const auto &varname = var->label();
          info.int_vars[varname] = var;
          info.var_info[varname] = SwarmVarInfo(var->GetDim(2));
        }
        for (const auto &var : swarm->GetVariableVector<Real>()) {
          const auto &varname = var->label();
          info.real_vars[varname] = var;
          info.var_info[varname] = SwarmVarInfo(var->GetDim(2));
        }
      }
    } else {
      for (const auto &swarmname : swarmnames) {
        if (swarm_container->Contains(swarmname)) {
          auto &swarm = swarm_container->Get(swarmname);
          auto &info = all_info[swarmname];
          std::size_t count = swarm->GetNumActive();
          std::size_t offset = (info.offsets.size() > 0) ? info.offsets.back() : 0;
          offset += (info.counts.size() > 0) ? info.counts.back() : 0;
          info.counts.push_back(count);
          info.offsets.push_back(offset);
          info.count_on_rank += count;
          for (const auto &varname : varnames) {
            if (swarm->Contains<int>(varname)) {
              auto var = swarm->GetP<int>(varname);
              info.int_vars[varname] = var;
              info.var_info[varname] = SwarmVarInfo(var->GetDim(2));
            } else if (swarm->Contains<Real>(varname)) {
              auto var = swarm->GetP<Real>(varname);
              info.real_vars[varname] = var;
              info.var_info[varname] = SwarmVarInfo(var->GetDim(2));
            } // else nothing
          }
        }
      }
    }
  }
  for (auto &[name,info] : all_info) {
    // TODO(JMM): Implies a bunch of blocking MPIAllReduces... but
    // we're just doing I/O right now, so probably ok?
    MPI_SIZE_t tot_count;
    info.global_offset = MPIPrefixSum(info.count_on_rank, tot_count);
    for (int i = 0; i < info.offsets.size(); ++i) {
      info.offsets[i] += info.global_offset;
    }
    info.global_count = tot_count;
  }
}

// TODO(JMM): may need to generalize this
MPI_SIZE_t MPIPrefixSum(MPI_SIZE_t local, MPI_SIZE_t &tot_count) {
  MPI_SIZE_t out = 0;
  tot_count = 0;
#ifdef MPI_PARALLEL
  std::vector<MPI_SIZE_t> buffer(Globals::nranks);
  MPI_Allgather(&local, 1, MPI_UNSIGNED_LONG_LONG, buffer.data(), 1,
                MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
  for (int i = 0; i < Globals::my_rank; ++i) {
    out += buffer[i];
  }
  for (int i = 0; i < buffer.size(); ++i) {
    tot_count += buffer[i];
  }
#else
  tot_count = local;
#endif // MPI_PARALLEL
  return out;
}


} // namespace OutputUtils
} // namespace parthenon
