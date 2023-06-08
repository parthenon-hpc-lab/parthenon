//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2023 The Parthenon collaboration
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
#include <set>
#include <string>
#include <type_traits>
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

void SwarmInfo::AddOffsets(const SP_Swarm &swarm) {
  std::size_t count = swarm->GetNumActive();
  std::size_t offset = (offsets.size() > 0) ? offsets.back() : 0;
  offset += (counts.size() > 0) ? counts.back() : 0;
  counts.push_back(count);
  offsets.push_back(offset);
  count_on_rank += count;
  max_indices.push_back(swarm->GetMaxActiveIndex());
  // JMM: If we defrag, we don't need these
  // masks.push_back(swarm->GetMask());
}

AllSwarmInfo::AllSwarmInfo(BlockList_t &block_list,
                           const std::map<std::string, std::set<std::string>> &swarmnames,
                           bool is_restart) {
  for (auto &pmb : block_list) {
    auto &swarm_container = pmb->swarm_data.Get();
    swarm_container->DefragAll(); // JMM: If we defrag, we don't need to mask?
    if (is_restart) {
      using FC = parthenon::Metadata::FlagCollection;
      auto flags =
          FC({parthenon::Metadata::Independent, parthenon::Metadata::Restart}, true);
      auto swarms = swarm_container->GetSwarmsByFlag(flags);
      for (auto &swarm : swarms) {
        auto swarmname = swarm->label();
        auto &info = all_info[swarmname];
        info.AddOffsets(swarm);
        for (const auto &var : swarm->GetVariableVector<int>()) {
          const auto &varname = var->label();
          info.Add(varname, var);
        }
        for (const auto &var : swarm->GetVariableVector<Real>()) {
          const auto &varname = var->label();
          info.Add(varname, var);
        }
      }
    } else {
      for (const auto &[swarmname, varnames] : swarmnames) {
        if (swarm_container->Contains(swarmname)) {
          auto &swarm = swarm_container->Get(swarmname);
          auto &info = all_info[swarmname];
          info.AddOffsets(swarm);
          for (const auto &varname : varnames) {
            if (swarm->Contains<int>(varname)) {
              auto var = swarm->GetP<int>(varname);
              info.Add(varname, var);
            } else if (swarm->Contains<Real>(varname)) {
              auto var = swarm->GetP<Real>(varname);
              info.Add(varname, var);
            } // else nothing
          }
        }
      }
    }
  }
  for (auto &[name, info] : all_info) {
    // TODO(JMM): Implies a bunch of blocking MPIAllReduces... but
    // we're just doing I/O right now, so probably ok?
    std::size_t tot_count;
    info.global_offset = MPIPrefixSum(info.count_on_rank, tot_count);
    for (int i = 0; i < info.offsets.size(); ++i) {
      info.offsets[i] += info.global_offset;
    }
    info.global_count = tot_count;
  }
}

// TODO(JMM): may need to generalize this
std::size_t MPIPrefixSum(std::size_t local, std::size_t &tot_count) {
  std::size_t out = 0;
  tot_count = 0;
#ifdef MPI_PARALLEL
  // Need to use sizeof here because unsigned long long and unsigned
  // long are identical under the hood but registered as different
  // types
  static_assert(std::is_integral<std::size_t>::value &&
                    !std::is_signed<std::size_t>::value,
                "size_t is unsigned and integral");
  static_assert(sizeof(std::size_t) == sizeof(unsigned long long int),
                "MPI_UNSIGNED_LONG_LONG same as size_t");
  std::vector<std::size_t> buffer(Globals::nranks);
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
std::size_t MPISum(std::size_t val) {
#ifdef MPI_PARALLEL
  // Need to use sizeof here because unsigned long long and unsigned
  // long are identical under the hood but registered as different
  // types
  static_assert(std::is_integral<std::size_t>::value &&
                    !std::is_signed<std::size_t>::value,
                "size_t is unsigned and integral");
  static_assert(sizeof(std::size_t) == sizeof(unsigned long long int),
                "MPI_UNSIGNED_LONG_LONG same as size_t");
  PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &val, 1, MPI_UNSIGNED_LONG_LONG,
                                    MPI_SUM, MPI_COMM_WORLD));
#endif
  return val;
}

} // namespace OutputUtils
} // namespace parthenon
