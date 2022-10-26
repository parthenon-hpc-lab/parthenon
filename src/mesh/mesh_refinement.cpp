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

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>

#include "coordinates/coordinates.hpp"
#include "defs.hpp"
#include "globals.hpp"
#include "interface/variable.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "mesh/mesh_refinement_loops.hpp"
#include "mesh/meshblock.hpp"
#include "mesh/refinement_in_one.hpp"
#include "parameter_input.hpp"
#include "parthenon_arrays.hpp"
#include "refinement/refinement.hpp"
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
//! \fn void MeshRefinement::RestrictCellCenteredValues(const ParArrayND<Real> &fine,
//                           ParArrayND<Real> &coarse, int sn, int en,
//                           int csi, int cei, int csj, int cej, int csk, int cek)
//  \brief restrict cell centered values

void MeshRefinement::RestrictCellCenteredValues(const CellVariable<Real> *fine,
                                                CellVariable<Real> *coarse, int sn,
                                                int en, int csi, int cei, int csj,
                                                int cej, int csk, int cek) {
  PARTHENON_DEBUG_REQUIRE(fine->label() == coarse->label(),
                          "Variable " + fine->label() + "== " + coarse->label());
  const auto &metadata = fine->metadata();
  PARTHENON_DEBUG_REQUIRE(metadata.IsRefined(), "Variable " + fine->base_name() +
                                                    " must be registered for refinement");
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();
  const auto &refinement_funcs = metadata.GetRefinementFunctions();
  const auto &restrictor = refinement_funcs.restrictor_host;
  int b = 0;
  int nbuffers = 1;
  // TODO(JMM): We're allocating on the heap here... we could move to
  // the stack by giving these functions pointers to underlying data?
  // Probably not worth it, as these functions will be completely removed soon.
  cell_centered_bvars::BufferCacheHost_t info_h("refinement info", nbuffers);
  refinement::loops::IdxHost_t idxs_h("host data", nbuffers);
  idxs_h(b) = b;
  // buff and var unused.
  info_h(b).si = csi;
  info_h(b).ei = cei;
  info_h(b).sj = csj;
  info_h(b).ej = cej;
  info_h(b).sk = csk;
  info_h(b).ek = cek;
  info_h(b).Nt = fine->GetDim(6);
  info_h(b).Nu = fine->GetDim(5);
  info_h(b).Nv = fine->GetDim(4);
  info_h(b).refinement_op = RefinementOp_t::Restriction;
  info_h(b).coords = pmb->coords;
  info_h(b).coarse_coords = this->coarse_coords;
  info_h(b).fine = (fine->data).Get();
  info_h(b).coarse = (coarse->coarse_s).Get();
  restrictor(info_h, idxs_h, pmb->cellbounds, pmb->c_cellbounds, nbuffers);
}

//----------------------------------------------------------------------------------------
//! \fn void MeshRefinement::ProlongateCellCenteredValues(
//        const ParArrayND<Real> &coarse,ParArrayND<Real> &fine, int sn, int en,,
//        int si, int ei, int sj, int ej, int sk, int ek)
//  \brief Prolongate cell centered values

void MeshRefinement::ProlongateCellCenteredValues(const CellVariable<Real> *coarse,
                                                  CellVariable<Real> *fine, int sn,
                                                  int en, int si, int ei, int sj, int ej,
                                                  int sk, int ek) {
  PARTHENON_DEBUG_REQUIRE(fine->label() == coarse->label(),
                          "Variable " + fine->label() + "== " + coarse->label());
  const auto &metadata = coarse->metadata();
  PARTHENON_DEBUG_REQUIRE(metadata.IsRefined(), "Variable " + coarse->base_name() +
                                                    " must be registered for refinement");
  std::shared_ptr<MeshBlock> pmb = GetBlockPointer();
  const auto &refinement_funcs = metadata.GetRefinementFunctions();
  const auto &prolongator = refinement_funcs.prolongator_host;
  int b = 0;
  int nbuffers = 1;
  // TODO(JMM): We're allocating on the heap here... we could move to
  // the stack by giving these functions pointers to underlying data?
  // Probably not worth it, as these functions will be completely removed soon.
  cell_centered_bvars::BufferCacheHost_t info_h("refinement info", nbuffers);
  refinement::loops::IdxHost_t idxs_h("host data", nbuffers);
  idxs_h(b) = b;
  // buff and var unused
  info_h(b).si = si;
  info_h(b).ei = ei;
  info_h(b).sj = sj;
  info_h(b).ej = ej;
  info_h(b).sk = sk;
  info_h(b).ek = ek;
  info_h(b).Nt = fine->GetDim(6);
  info_h(b).Nu = fine->GetDim(5);
  info_h(b).Nv = fine->GetDim(4);
  info_h(b).refinement_op = RefinementOp_t::Prolongation;
  info_h(b).coords = pmb->coords;
  info_h(b).coarse_coords = this->coarse_coords;
  info_h(b).fine = (fine->data).Get();
  info_h(b).coarse = (coarse->coarse_s).Get();
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
    if (pmb->loc.level == pmb->pmy_mesh->max_level) {
      refine_flag_ = 0;
    } else {
      refine_flag_ = 1;
    }
  } else if (aret < 0) {
    if (pmb->loc.level == pmb->pmy_mesh->root_level) {
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
            if (pmb->pbval->nblevel[k + 1][j + 1][i + 1] > pmb->loc.level) ec++;
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

int MeshRefinement::AddToRefinement(std::shared_ptr<CellVariable<Real>> pvar) {
  pvars_cc_.push_back(pvar);
  return static_cast<int>(pvars_cc_.size() - 1);
}

} // namespace parthenon
