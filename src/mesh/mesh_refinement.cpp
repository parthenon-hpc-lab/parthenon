//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2022. Triad National Security, LLC. All rights reserved.
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
//! \file mesh_refinement.cpp
//  \brief implements functions for static/adaptive mesh refinement

// TODO(JMM): The MeshRefinement can likely be simplified and/or
// removed entirely as we clean up our machinery and move to
// refinement-in-one everywhere in the code. I leave it in the `mesh`
// directory since it hooks into `Mesh` and `BoundaryValues` but in
// the long term this should be cleaned up.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>

#include "amr_criteria/refinement_package.hpp"
#include "coordinates/coordinates.hpp"
#include "defs.hpp"
#include "globals.hpp"
#include "interface/variable.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "mesh/meshblock.hpp"
#include "parameter_input.hpp"
#include "parthenon_arrays.hpp"
#include "prolong_restrict/pr_loops.hpp"
#include "prolong_restrict/prolong_restrict.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

//----------------------------------------------------------------------------------------
//! \fn MeshRefinement::MeshRefinement(MeshBlock *pmb, ParameterInput *pin)
//  \brief constructor

MeshRefinement::MeshRefinement(std::weak_ptr<MeshBlock> pmb, ParameterInput *pin)
    : pmy_block_(pmb), deref_count_(0),
      deref_threshold_(pin->GetOrAddInteger("parthenon/mesh", "derefine_count", 10)) {
  // Create coarse mesh object for parent grid
  coarse_coords = Coordinates_t(pmb.lock()->coords, 2);

  if ((Globals::nghost % 2) != 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in MeshRefinement constructor" << std::endl
        << "Selected --nghost=" << Globals::nghost
        << " is incompatible with mesh refinement because it is not a multiple of 2.\n"
        << "Rerun with an even number of ghost cells " << std::endl;
    PARTHENON_FAIL(msg);
  }
}

//----------------------------------------------------------------------------------------
//  \brief restrict cell centered values

void MeshRefinement::RestrictCellCenteredValues(Variable<Real> *var, int csi, int cei,
                                                int csj, int cej, int csk, int cek) {
  const auto &metadata = var->metadata();
  PARTHENON_DEBUG_REQUIRE(metadata.IsRefined(), "Variable " + var->base_name() +
                                                    " must be registered for refinement");
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();
  const auto &refinement_funcs = metadata.GetRefinementFunctions();
  const auto &restrictor = refinement_funcs.restrictor_host;
  int b = 0;
  int nbuffers = 1;
  // TODO(JMM): We're allocating on the heap here... we could move to
  // the stack by giving these functions pointers to underlying data?
  // Probably not worth it, as these functions will be completely removed soon.
  BufferCacheHost_t info_h("refinement info", nbuffers);
  refinement::loops::IdxHost_t idxs_h("host data", nbuffers);
  idxs_h(b) = b;
  // buff and var unused.
  block_ownership_t owns(true);
  info_h(b).prores_idxer[0] = SpatiallyMaskedIndexer6D(
      owns, {0, var->GetDim(6) - 1}, {0, var->GetDim(5) - 1}, {0, var->GetDim(4) - 1},
      {csk, cek}, {csj, cej}, {csi, cei});
  info_h(b).refinement_op = RefinementOp_t::Restriction;
  info_h(b).coords = pmb->coords;
  info_h(b).coarse_coords = this->coarse_coords;
  info_h(b).fine = (var->data).Get();
  info_h(b).coarse = (var->coarse_s).Get();
  restrictor(info_h, idxs_h, pmb->cellbounds, pmb->c_cellbounds, nbuffers);
}

//----------------------------------------------------------------------------------------
//  \brief Prolongate cell centered values

void MeshRefinement::ProlongateCellCenteredValues(Variable<Real> *var, int si, int ei,
                                                  int sj, int ej, int sk, int ek) {
  const auto &metadata = var->metadata();
  PARTHENON_DEBUG_REQUIRE(metadata.IsRefined(), "Variable " + var->base_name() +
                                                    " must be registered for refinement");
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();
  const auto &refinement_funcs = metadata.GetRefinementFunctions();
  const auto &prolongator = refinement_funcs.prolongator_host;
  int b = 0;
  int nbuffers = 1;
  // TODO(JMM): We're allocating on the heap here... we could move to
  // the stack by giving these functions pointers to underlying data?
  // Probably not worth it, as these functions will be completely removed soon.
  BufferCacheHost_t info_h("refinement info", nbuffers);
  refinement::loops::IdxHost_t idxs_h("host data", nbuffers);
  idxs_h(b) = b;
  // buff and var unused
  block_ownership_t owns(true);
  info_h(b).prores_idxer[0] =
      SpatiallyMaskedIndexer6D(owns, {0, var->GetDim(6) - 1}, {0, var->GetDim(5) - 1},
                               {0, var->GetDim(4) - 1}, {sk, ek}, {sj, ej}, {si, ei});
  info_h(b).refinement_op = RefinementOp_t::Prolongation;
  info_h(b).coords = pmb->coords;
  info_h(b).coarse_coords = this->coarse_coords;
  info_h(b).fine = (var->data).Get();
  info_h(b).coarse = (var->coarse_s).Get();
  prolongator(info_h, idxs_h, pmb->cellbounds, pmb->c_cellbounds, nbuffers);
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::CheckRefinementCondition()
//  \brief Check refinement criteria

void MeshRefinement::CheckRefinementCondition() {
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();
  auto &rc = pmb->meshblock_data.Get();
  AmrTag ret = Refinement::CheckAllRefinement(rc.get());
  SetRefinement(ret);
}

void MeshRefinement::SetRefinement(AmrTag flag) {
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();
  int aret = std::max(-1, static_cast<int>(flag));

  if (aret == 0) refine_flag_ = 0;

  if (aret >= 0) deref_count_ = 0;
  if (aret > 0) {
    if (pmb->loc.level() == pmb->pmy_mesh->max_level) {
      refine_flag_ = 0;
    } else {
      refine_flag_ = 1;
    }
  } else if (aret < 0) {
    if (pmb->loc.level() == pmb->pmy_mesh->root_level) {
      refine_flag_ = 0;
      deref_count_ = 0;
    } else {
      deref_count_++;
      int ec = 0, js, je, ks, ke;
      if (pmb->block_size.nx2 > 1) {
        js = -1;
        je = 1;
      } else {
        js = 0;
        je = 0;
      }
      if (pmb->block_size.nx3 > 1) {
        ks = -1;
        ke = 1;
      } else {
        ks = 0;
        ke = 0;
      }
      for (int k = ks; k <= ke; k++) {
        for (int j = js; j <= je; j++) {
          for (int i = -1; i <= 1; i++)
            if (pmb->pbval->nblevel[k + 1][j + 1][i + 1] > pmb->loc.level()) ec++;
        }
      }
      if (ec > 0) {
        refine_flag_ = 0;
      } else {
        if (deref_count_ >= deref_threshold_) {
          refine_flag_ = -1;
        } else {
          refine_flag_ = 0;
        }
      }
    }
  }

  return;
}

// TODO(felker): consider merging w/ MeshBlock::pvars_cc, etc. See meshblock.cpp

int MeshRefinement::AddToRefinement(std::shared_ptr<Variable<Real>> pvar) {
  pvars_cc_.insert(pvar);
  return static_cast<int>(pvars_cc_.size() - 1);
}

} // namespace parthenon
