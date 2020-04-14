//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
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

// C headers

// C++ headers
#include <tuple>
#include <vector>

// Athena++ headers
#include "athena.hpp"           // Real
#include "parthenon_arrays.hpp" // ParArrayND

// MPI headers
#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

namespace parthenon {
class MeshBlock;
class ParameterInput;
class Coordinates;
class BoundaryValues;

//----------------------------------------------------------------------------------------
//! \class MeshRefinement
//  \brief

class MeshRefinement {
  // needs to access pcoarsec in ProlongateBoundaries() for passing to BoundaryFunc()
  friend class BoundaryValues;
  // needs to access refine_flag_ in Mesh::AdaptiveMeshRefinement(). Make var public?
  friend class Mesh;

 public:
  MeshRefinement(MeshBlock *pmb, ParameterInput *pin);
  ~MeshRefinement();

  // functions
  void RestrictCellCenteredValues(const ParArrayND<Real> &fine, ParArrayND<Real> &coarse,
                                  int sn, int en, int csi, int cei, int csj, int cej,
                                  int csk, int cek);
  void RestrictFieldX1(const ParArrayND<Real> &fine, ParArrayND<Real> &coarse, int csi,
                       int cei, int csj, int cej, int csk, int cek);
  void RestrictFieldX2(const ParArrayND<Real> &fine, ParArrayND<Real> &coarse, int csi,
                       int cei, int csj, int cej, int csk, int cek);
  void RestrictFieldX3(const ParArrayND<Real> &fine, ParArrayND<Real> &coarse, int csi,
                       int cei, int csj, int cej, int csk, int cek);
  void ProlongateCellCenteredValues(const ParArrayND<Real> &coarse,
                                    ParArrayND<Real> &fine, int sn, int en, int si,
                                    int ei, int sj, int ej, int sk, int ek);
  void ProlongateSharedFieldX1(const ParArrayND<Real> &coarse, ParArrayND<Real> &fine,
                               int si, int ei, int sj, int ej, int sk, int ek);
  void ProlongateSharedFieldX2(const ParArrayND<Real> &coarse, ParArrayND<Real> &fine,
                               int si, int ei, int sj, int ej, int sk, int ek);
  void ProlongateSharedFieldX3(const ParArrayND<Real> &coarse, ParArrayND<Real> &fine,
                               int si, int ei, int sj, int ej, int sk, int ek);
  void ProlongateInternalField(FaceField &fine, int si, int ei, int sj, int ej, int sk,
                               int ek);
  void CheckRefinementCondition();
  void SetRefinement(AmrTag flag);

  // setter functions for "enrolling" variable arrays in refinement via Mesh::AMR()
  // and/or in BoundaryValues::ProlongateBoundaries() (for SMR and AMR)
  int AddToRefinement(ParArrayND<Real> pvar_cc, ParArrayND<Real> pcoarse_cc);
  int AddToRefinement(FaceField *pvar_fc, FaceField *pcoarse_fc);

 private:
  // data
  MeshBlock *pmy_block_;
  Coordinates *pcoarsec;

  ParArrayND<Real> fvol_[2][2], sarea_x1_[2][2], sarea_x2_[2][3], sarea_x3_[3][2];
  int refine_flag_, neighbor_rflag_, deref_count_, deref_threshold_;

  // functions
  AMRFlagFunc AMRFlag_; // duplicate of Mesh class member

  // tuples of references to AMR-enrolled arrays (quantity, coarse_quantity)
  std::vector<std::tuple<ParArrayND<Real>, ParArrayND<Real>>> pvars_cc_;
  std::vector<std::tuple<FaceField *, FaceField *>> pvars_fc_;
};
} // namespace parthenon
#endif // MESH_MESH_REFINEMENT_HPP_
