//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
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
#ifndef MESH_MESH_REFINEMENT_HPP_
#define MESH_MESH_REFINEMENT_HPP_
//! \file mesh_refinement.hpp
//  \brief defines MeshRefinement class used for static/adaptive mesh refinement

// TODO(JMM): The MeshRefinement can likely be simplified and/or
// removed entirely as we clean up our machinery and move to
// refinement-in-one everywhere in the code. I leave it in the `mesh`
// directory since it hooks into `Mesh` and `BoundaryValues` but in
// the long term this should be cleaned up.

#include <memory>
#include <set>
#include <tuple>
#include <vector>

#include "interface/variable.hpp"
#include "parthenon_mpi.hpp"

#include "coordinates/coordinates.hpp"
#include "defs.hpp"
#include "parthenon_arrays.hpp"

namespace parthenon {

class MeshBlock;
class ParameterInput;

//----------------------------------------------------------------------------------------
//! \class MeshRefinement
//  \brief

class MeshRefinement {
  // needs to access refine_flag_ in Mesh::AdaptiveMeshRefinement(). Make var public?
  friend class Mesh;

 public:
  MeshRefinement(std::weak_ptr<MeshBlock> pmb, ParameterInput *pin);

  // JMM: fine and coarse may be on different meshblocks and thus
  // different variable objects.
  void RestrictCellCenteredValues(Variable<Real> *var, int csi, int cei, int csj, int cej,
                                  int csk, int cek);
  void ProlongateCellCenteredValues(Variable<Real> *var, int si, int ei, int sj, int ej,
                                    int sk, int ek);
  void CheckRefinementCondition();
  void SetRefinement(AmrTag flag);

  // setter functions for "enrolling" variable arrays in refinement via Mesh::AMR()
  int AddToRefinement(std::shared_ptr<Variable<Real>> pvar);

  // TODO(JMM): coarse-coords maybe should move out of this code, or
  // be made public
  Coordinates_t GetCoarseCoords() const { return coarse_coords; }

  int &DereferenceCount() { return deref_count_; }

 private:
  // data
  std::weak_ptr<MeshBlock> pmy_block_;
  Coordinates_t coarse_coords;

  int refine_flag_, neighbor_rflag_, deref_count_, deref_threshold_;

  // tuples of references to AMR-enrolled arrays (quantity, coarse_quantity)
  VariableSet<Real> pvars_cc_;

  // Returns shared pointer to a block
  std::shared_ptr<MeshBlock> GetBlockPointer() {
    if (pmy_block_.expired()) {
      PARTHENON_THROW("Invalid pointer to MeshBlock!");
    }
    return pmy_block_.lock();
  }
};

} // namespace parthenon

#endif // MESH_MESH_REFINEMENT_HPP_
