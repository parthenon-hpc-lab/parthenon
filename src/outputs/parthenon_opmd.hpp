//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2024 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
#ifndef OUTPUTS_PARTHENON_OPMD_HPP_
#define OUTPUTS_PARTHENON_OPMD_HPP_
//! \file restart_opmd.hpp
//  \brief Provides support for restarting from OpenPMD output

#include <memory>
#include <string>
#include <tuple>

#include "mesh/meshblock.hpp"
#include "openPMD/Dataset.hpp"
#include "openPMD/Iteration.hpp"
#include "outputs/output_utils.hpp"

namespace parthenon {

namespace OpenPMDUtils {

void WriteAllParams(const Params &params, const std::string &prefix,
                    openPMD::Iteration *it);

// Deliminter to separate packages and parameters in attributes.
// More or less a workaround as the OpenPMD API does currently not expose
// access to non-standard groups (such as "Params" versus the standard "meshes").
inline static const std::string delim = "~";

// Construct OpenPMD Mesh "record" name and comonnent identifier.
// - comp_idx is a flattended index over all components of the vectors and tensors, i.e.,
// the typical v,u,t indices.
// - level is the current effective level of the Mesh record
std::tuple<std::string, std::string>
GetMeshRecordAndComponentNames(const OutputUtils::VarInfo &vinfo, const int comp_idx,
                               const int level);

// Calculate logical location on effective mesh (i.e., a mesh with size that matches full
// coverage at given resolution on a particular level)
// TODO(pgrete) needs to be updated to properly work with Forests
std::tuple<openPMD::Offset, openPMD::Extent>
GetChunkOffsetAndExtent(Mesh *pm, std::shared_ptr<MeshBlock> pmb);

} // namespace OpenPMDUtils
} // namespace parthenon
#endif // OUTPUTS_PARTHENON_OPMD_HPP_
