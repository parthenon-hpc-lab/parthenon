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

#ifndef OUTPUTS_OUTPUT_UTILS_HPP_
#define OUTPUTS_OUTPUT_UTILS_HPP_

// C++
#include <array>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

// Parthenon
#include "interface/metadata.hpp"
#include "interface/variable.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {
namespace OutputUtils {
// Helper struct containing some information about a variable
struct VarInfo {
  std::string label;
  int num_components;
  int nx6;
  int nx5;
  int nx4;
  int nx3;
  int nx2;
  int nx1;
  int tensor_rank; // 0- to 3-D for cell-centered variables, 0- to 6-D for arbitrary shape
                   // variables
  MetadataFlag where;
  bool is_sparse;
  bool is_vector;
  std::vector<std::string> component_labels;

  VarInfo() = delete;

  // TODO(JMM): Separate this into an implementation file again?
  VarInfo(const std::string &label, const std::vector<std::string> &component_labels_,
          int num_components, int nx6, int nx5, int nx4, int nx3, int nx2, int nx1,
          Metadata metadata, bool is_sparse, bool is_vector)
      : label(label), num_components(num_components), nx6(nx6), nx5(nx5), nx4(nx4),
        nx3(nx3), nx2(nx2), nx1(nx1), tensor_rank(metadata.Shape().size()),
        where(metadata.Where()), is_sparse(is_sparse), is_vector(is_vector) {
    if (num_components <= 0) {
      std::stringstream msg;
      msg << "### ERROR: Got variable " << label << " with " << num_components
          << " components."
          << " num_components must be greater than 0" << std::endl;
      PARTHENON_FAIL(msg);
    }

    // Note that this logic does not subscript components without component_labels if
    // there is only one component. Component names will be e.g.
    //   my_scalar
    // or
    //   my_non-vector_set_0
    //   my_non-vector_set_1
    // Note that this means the subscript will be dropped for multidim quantities if their
    // Nx6, Nx5, Nx4 are set to 1 at runtime e.g.
    //   my_non-vector_set
    // Similarly, if component labels are given for all components, those will be used
    // without the prefixed label.
    component_labels = {};
    if (num_components == 1 || is_vector) {
      component_labels = component_labels_.size() > 0 ? component_labels_
                                                      : std::vector<std::string>({label});
    } else if (component_labels_.size() == num_components) {
      component_labels = component_labels_;
    } else {
      for (int i = 0; i < num_components; i++) {
        component_labels.push_back(label + "_" + std::to_string(i));
      }
    }
  }

  explicit VarInfo(const std::shared_ptr<CellVariable<Real>> &var)
      : VarInfo(var->label(), var->metadata().getComponentLabels(), var->NumComponents(),
                var->GetDim(6), var->GetDim(5), var->GetDim(4), var->GetDim(3),
                var->GetDim(2), var->GetDim(1), var->metadata(), var->IsSparse(),
                var->IsSet(Metadata::Vector)) {}
};

struct SwarmVarInfo {
  std::array<int, 5> n;
  int nvar, tensor_rank;
  bool vector;
  SwarmVarInfo() = default;
  SwarmVarInfo(int n6, int n5, int n4, int n3, int n2, int rank, bool vector)
    : n({n2, n3, n4, n5, n6}), nvar(n6 * n5 * n4 * n3 * n2),
        tensor_rank(rank), vector((tensor_rank == 1) && (nvar == 3) && vector) {}
  int GetN(int d) const {
    PARTHENON_DEBUG_REQUIRE_THROWS(1 < d && d <= 6, "allowed dim");
    return n[d - 2];
  }
};
// Contains information about a particle swarm spanning
// meshblocks. Everything needed for output
struct SwarmInfo {
  SwarmInfo() = default;
  template<typename T>
  using MapToVarVec = std::map<std::string, ParticleVariableVector<T>>;
  std::tuple<MapToVarVec<int>, MapToVarVec<Real>> vars; // SwarmVars on each meshblock
  std::map<std::string, SwarmVarInfo> var_info; // size of each swarmvar
  std::size_t count_on_rank = 0; // per-meshblock
  std::size_t global_offset; // global
  std::size_t global_count; // global
  std::vector<std::size_t> counts;  // per-meshblock
  std::vector<std::size_t> offsets; // global
  std::vector<ParArray1D<bool>> masks; // used for reading swarms without defrag
  std::vector<std::size_t> max_indices; // JMM: If we defrag, unneeded?
  void AddOffsets(const SP_Swarm &swarm); // sets above metadata
  template<typename T>
  MapToVarVec<T> &Vars() {
    return std::get<MapToVarVec<T>>(vars);
  }
  template<typename T>
  void Add(const std::string &varname, const ParticleVarPtr<T> &var) {
    Vars<T>()[varname].push_back(var);
    auto m = var->metadata();
    bool vector = m.IsSet(Metadata::Vector);
    auto shape = m.Shape();
    int rank = shape.size();
    std::cout << "tensor_rank = " << rank << std::endl; // DEBUG
    var_info[varname] = SwarmVarInfo(var->GetDim(6), var->GetDim(5), var->GetDim(4),
                                     var->GetDim(3), var->GetDim(2), rank, vector);
  }
  // Copies swarmvar to host in prep for output
  template <typename T>
  std::vector<T> FillHostBuffer(const std::string vname,
                                ParticleVariableVector<T> &swmvarvec) {
    const auto &vinfo = var_info.at(vname);
    std::vector<T> host_data(count_on_rank * vinfo.nvar);
    std::size_t ivec = 0;
    std::size_t block_idx = 0;
    for (int n6 = 0; n6 < vinfo.GetN(6); ++n6) {
      for (int n5 = 0; n5 < vinfo.GetN(5); ++n5) {
        for (int n4 = 0; n4 < vinfo.GetN(4); ++n4) {
          for (int n3 = 0; n3 < vinfo.GetN(3); ++n3) {
            for (int n2 = 0; n2 < vinfo.GetN(2); ++n2) {
              for (auto &swmvar : swmvarvec) {
                // Copied extra times. JMM: If we defrag, unneeded?
                auto mask_h = masks[block_idx].GetHostMirrorAndCopy();
                // Prevents us from having to copy extra data for swarm vars
                // with multiple components
                auto v_h = swmvar->GetHostMirrorAndCopy(n6, n5, n4, n3, n2);
                // DO NOT use GetDim, as it does not reflect particle count
                std::size_t max_index = max_indices[block_idx];
                std::size_t particles_added = 0;
                for (std::size_t i = 0; i <= max_index; ++i) {
                  if (mask_h(i)) {
//                  if (true) {
                    host_data[ivec++] = v_h(i);
                    particles_added++;
                  }
                }
                std::cout << "particles added, counts = " << particles_added
                          << ", " << counts[block_idx]
                          << std::endl;
                PARTHENON_REQUIRE_THROWS(particles_added == counts[block_idx],
                                         "All active particles set for output");
                ++block_idx;
              }
            }
          }
        }
      }
    }
    return host_data; // move semantics
  }
};
struct AllSwarmInfo {
  std::map<std::string, SwarmInfo> all_info;
  AllSwarmInfo(BlockList_t &block_list,
               const std::vector<std::string> &swarmnames,
               const std::vector<std::string> &varnames,
               bool is_restart);
};

// TODO(JMM): Potentially unsafe if these types aren't compatible
// TODO(JMM): If we ever need non-int need to generalize
using MPI_SIZE_t = unsigned long long int;
MPI_SIZE_t MPIPrefixSum(MPI_SIZE_t local, MPI_SIZE_t &tot_count);

} // namespace OutputUtils
} // namespace parthenon

#endif // OUTPUTS_OUTPUT_UTILS_HPP_
