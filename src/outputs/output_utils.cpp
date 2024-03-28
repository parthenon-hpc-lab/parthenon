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
    auto &swarm_container = pmb->meshblock_data.Get()->swarm_data.Get();
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

// Tools that can be shared accross Output types

std::vector<Real> ComputeXminBlocks(Mesh *pm) {
  return FlattenBlockInfo<Real>(pm, pm->ndim,
                                [=](MeshBlock *pmb, std::vector<Real> &data, int &i) {
                                  auto xmin = pmb->coords.GetXmin();
                                  data[i++] = xmin[0];
                                  if (pm->ndim > 1) {
                                    data[i++] = xmin[1];
                                  }
                                  if (pm->ndim > 2) {
                                    data[i++] = xmin[2];
                                  }
                                });
}

std::vector<int64_t> ComputeLocs(Mesh *pm) {
  return FlattenBlockInfo<int64_t>(
      pm, 3, [=](MeshBlock *pmb, std::vector<int64_t> &locs, int &i) {
        locs[i++] = pmb->loc.lx1();
        locs[i++] = pmb->loc.lx2();
        locs[i++] = pmb->loc.lx3();
      });
}

std::vector<int> ComputeIDsAndFlags(Mesh *pm) {
  return FlattenBlockInfo<int>(pm, 5,
                               [=](MeshBlock *pmb, std::vector<int> &data, int &i) {
                                 data[i++] = pmb->loc.level();
                                 data[i++] = pmb->gid;
                                 data[i++] = pmb->lid;
                                 data[i++] = pmb->cnghost;
                                 data[i++] = pmb->gflag;
                               });
}

// TODO(JMM): I could make this use the other loop
// functionality/high-order functions.  but it was more code than this
// for, I think, little benefit.
void ComputeCoords(Mesh *pm, bool face, const IndexRange &ib, const IndexRange &jb,
                   const IndexRange &kb, std::vector<Real> &x, std::vector<Real> &y,
                   std::vector<Real> &z) {
  const int nx1 = ib.e - ib.s + 1;
  const int nx2 = jb.e - jb.s + 1;
  const int nx3 = kb.e - kb.s + 1;
  const int num_blocks = pm->block_list.size();
  x.resize((nx1 + face) * num_blocks);
  y.resize((nx2 + face) * num_blocks);
  z.resize((nx3 + face) * num_blocks);
  std::size_t idx_x = 0, idx_y = 0, idx_z = 0;

  // note relies on casting of bool to int
  for (auto &pmb : pm->block_list) {
    for (int i = ib.s; i <= ib.e + face; ++i) {
      x[idx_x++] = face ? pmb->coords.Xf<1>(i) : pmb->coords.Xc<1>(i);
    }
    for (int j = jb.s; j <= jb.e + face; ++j) {
      y[idx_y++] = face ? pmb->coords.Xf<2>(j) : pmb->coords.Xc<2>(j);
    }
    for (int k = kb.s; k <= kb.e + face; ++k) {
      z[idx_z++] = face ? pmb->coords.Xf<3>(k) : pmb->coords.Xc<3>(k);
    }
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
