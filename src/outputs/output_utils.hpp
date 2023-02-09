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
#include <memory>
#include <sstream>
#include <string>
#include <vector>

// Parthenon
#include "interface/metadata.hpp"
#include "interface/variable.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {
namespace OutputUtils {
// Helper struct containing some information about a variable
struct VarInfo {
  std::string label;
  int vlen;
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

  VarInfo(const std::string &label, const std::vector<std::string> &component_labels_,
          int vlen, int nx6, int nx5, int nx4, int nx3, int nx2, int nx1,
          Metadata metadata, bool is_sparse, bool is_vector)
      : label(label), vlen(vlen), nx6(nx6), nx5(nx5), nx4(nx4), nx3(nx3), nx2(nx2),
        nx1(nx1), tensor_rank(metadata.Shape().size()), where(metadata.Where()),
        is_sparse(is_sparse), is_vector(is_vector) {
    if (vlen <= 0) {
      std::stringstream msg;
      msg << "### ERROR: Got variable " << label << " with length " << vlen
          << ". vlen must be greater than 0" << std::endl;
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
    if (vlen == 1 || is_vector) {
      component_labels = component_labels_.size() > 0 ? component_labels_
                                                      : std::vector<std::string>({label});
    } else if (component_labels_.size() == vlen) {
      component_labels = component_labels_;
    } else {
      for (int i = 0; i < vlen; i++) {
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

// TODO(JMM): If higher tensorial rank swarms are ever added this will
// need to be changed
struct SwarmVarInfo {
  std::string label;
  int npart; // assumes swarm has been defragmented.
  int nvar;
  int tensor_rank;
  SwarmVarInfo() = delete;
  explicit SwarmVarInfo(const std::shared_ptr<ParticleVariable<Real>> &var)
      : label(var->label()), npart(var->GetDim(1)), nvar(var->GetDim(2)),
        tensor_rank(var->GetDim(2) > 1 ? 1 : 0) {}
};
} // namespace OutputUtils
} // namespace parthenon

#endif // OUTPUTS_OUTPUT_UTILS_HPP_
