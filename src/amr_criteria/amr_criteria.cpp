//========================================================================================
// (C) (or copyright) 2020-2025. Triad National Security, LLC. All rights reserved.
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
#include "amr_criteria/amr_criteria.hpp"

#include <iostream>
#include <memory>
#include <string>

#include "amr_criteria/refinement_package.hpp"
#include "interface/meshblock_data.hpp"
#include "interface/variable.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"

namespace parthenon {

AMRCriteria::AMRCriteria(ParameterInput *pin, std::string &block_name)
    : comp6(0), comp5(0), comp4(0) {
  field =
      pin->GetOrAddString(block_name, "field", "NO FIELD WAS SET", "Field to refine on");
  if (field == "NO FIELD WAS SET") {
    std::cerr << "Error in " << block_name << ": no field set" << std::endl;
    exit(1);
  }
  if (pin->DoesParameterExist(block_name, "tensor_ijk")) {
    auto index = pin->GetVector<int>(block_name, "tensor_ijk");
    PARTHENON_REQUIRE_THROWS(
        index.size() == 3, "tensor_ijk requires three values, e.g. tensor_ijk = 2, 1, 3");
    comp6 = index[0];
    comp5 = index[1];
    comp4 = index[2];
  } else if (pin->DoesParameterExist(block_name, "tensor_ij")) {
    auto index = pin->GetVector<int>(block_name, "tensor_ij");
    PARTHENON_REQUIRE_THROWS(index.size() == 2,
                             "tensor_ij requires two values, e.g. tensor_ij = 2, 1");
    comp5 = index[0];
    comp4 = index[1];
  } else if (pin->DoesParameterExist(block_name, "vector_i")) {
    auto index = pin->GetVector<int>(block_name, "vector_i");
    PARTHENON_REQUIRE_THROWS(index.size() == 1,
                             "vector_i requires one value, e.g. vector_i = 2");
    comp4 = index[0];
  }
  refine_criteria = pin->GetOrAddReal(block_name, "refine_tol", 0.5,
                                      "magnitude that triggers refinement");
  derefine_criteria = pin->GetOrAddReal(block_name, "derefine_tol", 0.05,
                                        "magnitude that triggers de-refinement");
  int global_max_level = pin->GetOrAddInteger("parthenon/mesh", "numlevel", 1,
                                              "maximum level of refinement globally");
  max_level =
      pin->GetOrAddInteger(block_name, "max_level", global_max_level,
                           "maximum level this refinement criterion will achieve");
  if (max_level > global_max_level) {
    std::cerr << "WARNING: max_level in " << block_name
              << " exceeds numlevel (the global maximum number of levels) set in "
                 "<parthenon/mesh>."
              << std::endl
              << std::endl
              << "Setting max_level = numlevel, but this may not be what you want."
              << std::endl
              << std::endl;
    max_level = global_max_level;
  }
}

std::shared_ptr<AMRCriteria> AMRCriteria::MakeAMRCriteria(std::string &criteria,
                                                          ParameterInput *pin,
                                                          std::string &block_name) {
  if (criteria == "derivative_order_1")
    return std::make_shared<AMRFirstDerivative>(pin, block_name);
  if (criteria == "derivative_order_2")
    return std::make_shared<AMRSecondDerivative>(pin, block_name);
  if (criteria == "magnitude") return std::make_shared<AMRMagnitude>(pin, block_name);
  throw std::invalid_argument("\n  Invalid selection for refinment method in " +
                              block_name + ": " + criteria);
}

void AMRFirstDerivative::operator()(MeshData<Real> *md,
                                    ParArray1D<AmrTag> &amr_tags) const {
  auto ib = md->GetBoundsI(IndexDomain::interior);
  auto jb = md->GetBoundsJ(IndexDomain::interior);
  auto kb = md->GetBoundsK(IndexDomain::interior);
  auto bnds = AMRBounds(ib, jb, kb);
  auto dims = md->GetMeshPointer()->resolved_packages->FieldMetadata(field).Shape();
  int n5(0), n4(0);
  if (dims.size() > 2) {
    n5 = dims[1];
    n4 = dims[2];
  } else if (dims.size() > 1) {
    n5 = dims[0];
    n4 = dims[1];
  }
  const int idx = comp4 + n4 * (comp5 + n5 * comp6);
  Refinement::FirstDerivative(bnds, md, field, idx, amr_tags, refine_criteria,
                              derefine_criteria, max_level);
}

void AMRSecondDerivative::operator()(MeshData<Real> *md,
                                     ParArray1D<AmrTag> &amr_tags) const {
  auto ib = md->GetBoundsI(IndexDomain::interior);
  auto jb = md->GetBoundsJ(IndexDomain::interior);
  auto kb = md->GetBoundsK(IndexDomain::interior);
  auto bnds = AMRBounds(ib, jb, kb);
  auto dims = md->GetMeshPointer()->resolved_packages->FieldMetadata(field).Shape();
  int n5(0), n4(0);
  if (dims.size() > 2) {
    n5 = dims[1];
    n4 = dims[2];
  } else if (dims.size() > 1) {
    n5 = dims[0];
    n4 = dims[1];
  }
  const int idx = comp4 + n4 * (comp5 + n5 * comp6);
  Refinement::SecondDerivative(bnds, md, field, idx, amr_tags, refine_criteria,
                               derefine_criteria, max_level);
}

void AMRMagnitude::operator()(MeshData<Real> *md, ParArray1D<AmrTag> &amr_tags) const {
  auto ib = md->GetBoundsI(IndexDomain::interior);
  auto jb = md->GetBoundsJ(IndexDomain::interior);
  auto kb = md->GetBoundsK(IndexDomain::interior);
  auto bnds = AMRBounds(ib, jb, kb);
  auto dims = md->GetMeshPointer()->resolved_packages->FieldMetadata(field).Shape();
  int n5(0), n4(0);
  if (dims.size() > 2) {
    n5 = dims[1];
    n4 = dims[2];
  } else if (dims.size() > 1) {
    n5 = dims[0];
    n4 = dims[1];
  }
  const int idx = comp4 + n4 * (comp5 + n5 * comp6);
  Refinement::Magnitude(bnds, md, field, idx, amr_tags, sign, refine_criteria,
                        derefine_criteria, max_level);
}

} // namespace parthenon
