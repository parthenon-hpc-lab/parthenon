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

#ifndef AMR_CRITERIA_REFINEMENT_PACKAGE_HPP_
#define AMR_CRITERIA_REFINEMENT_PACKAGE_HPP_

#include <memory>
#include <string>

#include "defs.hpp"
#include "parthenon_arrays.hpp"

namespace parthenon {

class ParameterInput;
class MeshBlock;
template <typename T>
class MeshBlockData;
template <typename T>
class MeshData;
class StateDescriptor;
class AMRBounds;

namespace Refinement {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);
template <typename T>
TaskStatus Tag(T *rc);

AmrTag CheckAllRefinement(MeshBlockData<Real> *rc, const AmrTag &level);
ParArray1D<AmrTag> CheckAllRefinement(MeshData<Real> *md);

AmrTag FirstDerivative(const AMRBounds &bnds, const ParArray3D<Real> &q,
                       const Real refine_criteria, const Real derefine_criteria);

void FirstDerivative(const AMRBounds &bnds, MeshData<Real> *md, const std::string &field,
                     const int &idx, ParArray1D<AmrTag> &amr_tags,
                     const Real refine_criteria_, const Real derefine_criteria_,
                     const int max_level_);

AmrTag SecondDerivative(const AMRBounds &bnds, const ParArray3D<Real> &q,
                        const Real refine_criteria, const Real derefine_criteria);

void SecondDerivative(const AMRBounds &bnds, MeshData<Real> *md, const std::string &field,
                      const int &idx, ParArray1D<AmrTag> &amr_tags,
                      const Real refine_criteria_, const Real derefine_criteria_,
                      const int max_level_);

} // namespace Refinement

} // namespace parthenon

#endif // AMR_CRITERIA_REFINEMENT_PACKAGE_HPP_
