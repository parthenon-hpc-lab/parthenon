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

// Standard Includes
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// Parthenon Includes
#include <coordinates/coordinates.hpp>
#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>

// Local Includes
#include "calculate_pi.hpp"

using namespace parthenon::package::prelude;

// This defines a "physics" package
// In this case, calculate_pi provides the functions required to set up
// an indicator function in_or_out(x,y) = (r < r0 ? 1 : 0), and compute the area
// of a circle of radius r0 as A = \int d^x in_or_out(x,y) over the domain. Then
// pi \approx A/r0^2
namespace calculate_pi {

void SetInOrOut(MeshBlockData<Real> *rc) {
  auto pmb = rc->GetBlockPointer();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  ParArrayND<Real> &v = rc->Get("in_or_out").data;
  const auto &radius = pmb->packages.Get("calculate_pi")->Param<Real>("radius");
  auto &coords = pmb->coords;
  // Set an indicator function that indicates whether the cell center
  // is inside or outside of the circle we're interating the area of.
  // Loop bounds are set to catch the case where the edge is between the
  // cell centers of the first/last real cell and the first ghost cell
  pmb->par_for(
      "SetInOrOut", kb.s, kb.e, jb.s - 1, jb.e + 1, ib.s - 1, ib.e + 1,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        Real rsq = std::pow(coords.x1v(i), 2) + std::pow(coords.x2v(j), 2);
        if (rsq < radius * radius) {
          v(k, j, i) = 1.0;
        } else {
          v(k, j, i) = 0.0;
        }
      });
}

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto package = std::make_shared<StateDescriptor>("calculate_pi");
  Params &params = package->AllParams();

  Real radius = pin->GetOrAddReal("Pi", "radius", 1.0);
  params.Add("radius", radius);

  // add a variable called in_or_out that will hold the value of the indicator function
  std::string field_name("in_or_out");
  Metadata m({Metadata::Cell, Metadata::Derived});
  package->AddField(field_name, m);

  // All the package FillDerived and CheckRefinement functions are called by parthenon
  package->FillDerivedBlock = SetInOrOut;
  // could use package specific refinement tagging routine (see advection example), but
  // instead this example will make use of the parthenon shipped first derivative
  // criteria, as invoked in the input file
  // package->CheckRefinementBlock = CheckRefinement;

  return package;
}

TaskStatus ComputeArea(std::shared_ptr<MeshData<Real>> &md, ParArrayHost<Real> areas,
                       int i) {
  auto pack = md->PackVariables(std::vector<std::string>({"in_or_out"}));
  const IndexRange ib = pack.cellbounds.GetBoundsI(IndexDomain::interior);
  const IndexRange jb = pack.cellbounds.GetBoundsJ(IndexDomain::interior);
  const IndexRange kb = pack.cellbounds.GetBoundsK(IndexDomain::interior);

  Real area = 0.0;
  par_reduce(
      parthenon::loop_pattern_mdrange_tag, "calculate_pi compute area",
      parthenon::DevExecSpace(), 0, pack.GetDim(5) - 1, 0, pack.GetDim(4) - 1, kb.s, kb.e,
      jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(int b, int v, int k, int j, int i, Real &larea) {
        larea += pack(b, v, k, j, i) * pack.coords(b).Area(parthenon::X3DIR, k, j, i);
      },
      area);

  areas(i) = area;
  return TaskStatus::complete;
}

TaskStatus AccumulateAreas(ParArrayHost<Real> areas, Packages_t &packages) {
  const auto &radius = packages.Get("calculate_pi")->Param<Real>("radius");

  Real area = 0.0;
  for (int i = 0; i < areas.GetSize(); i++) {
    area += areas(i);
  }
  area /= (radius * radius);

#ifdef MPI_PARALLEL
  Real pi_val;
  PARTHENON_MPI_CHECK(
      MPI_Reduce(&area, &pi_val, 1, MPI_PARTHENON_REAL, MPI_SUM, 0, MPI_COMM_WORLD));
#else
  Real pi_val = area;
#endif

  packages.Get("calculate_pi")->AddParam("pi_val", pi_val);
  return TaskStatus::complete;
}

} // namespace calculate_pi
