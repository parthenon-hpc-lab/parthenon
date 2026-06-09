//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2024-2026 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
#ifndef OUTPUTS_PARTHENON_OPMD_HPP_
#define OUTPUTS_PARTHENON_OPMD_HPP_
//! \file restart_opmd.hpp
//  \brief Provides support for restarting from OpenPMD output

// C++ stdlib
#include <memory>
#include <string>
#include <tuple>

// OpenPMD headers
#include <openPMD/openPMD.hpp>

#include "basic_types.hpp"
#include "mesh/meshblock.hpp"
#include "outputs/output_parameters.hpp"
#include "outputs/output_utils.hpp"

namespace parthenon {

namespace OpenPMDUtils {

template <typename T>
void RestoreViewAttribute(const std::string &full_path, T &view, openPMD::Iteration *it);

void WriteAllParams(const Params &params, const std::string &prefix,
                    openPMD::Iteration *it);

// Deliminter to separate packages and parameters in attributes.
// More or less a workaround as the OpenPMD API does currently not expose
// access to non-standard groups (such as "Params" versus the standard "meshes").
inline static const std::string delim = "~";

// OpenPMD standard: names of records and their components are only allowed to contain
// the characters a-Z, the numbers 0-9 and the underscore _
void CheckValidName(const std::string &name);

// Construct OpenPMD Mesh "record" name and comonnent identifier.
// - te is the TopologicalElement (which is used as part of the variable name record)
// - comp_idx is a flattened index over all components of the vectors and tensors, i.e.,
// the typical v,u,t indices.
// - level is the current effective level of the Mesh record
std::tuple<std::string, std::string>
GetMeshRecordAndComponentNames(const OutputUtils::VarInfo &vinfo,
                               const TopologicalElement te, const int comp_idx,
                               const int level);

// Calculate logical location on effective mesh (i.e., a mesh with size that matches full
// coverage at given resolution on a particular level)
// TODO(pgrete) needs to be updated to properly work with Forests
std::tuple<openPMD::Offset, openPMD::Extent>
GetChunkOffsetAndExtent(Mesh *pm, std::shared_ptr<MeshBlock> pmb,
                        const TopologicalElement te, const int coarsening_factor,
                        const DumpOutputMode outupt_type);

// Construct OpenPMD Particle "record" name and comonnent identifier.
// - vname is the variable name
// - rank is the variable rank (i.e., 0 is scalar etc)
// - comp_idx is a flattened index over all components of the vectors and tensors, i.e.,
// the typical v,u,t indices.
std::tuple<std::string, std::string>
GetParticleRecordAndComponentNames(const std::string &vname, const int rank,
                                   const int flat_comp_idx);

} // namespace OpenPMDUtils
} // namespace parthenon

#endif // OUTPUTS_PARTHENON_OPMD_HPP_
