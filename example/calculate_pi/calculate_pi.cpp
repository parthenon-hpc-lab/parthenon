//========================================================================================
// (C) (or copyright) 2020-2021. Triad National Security, LLC. All rights reserved.
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
#include <mesh/domain.hpp>
#include <parthenon/package.hpp>

// Local Includes
#include "calculate_pi.hpp"

using namespace parthenon::package::prelude;
using parthenon::IndexShape;

// This defines a "physics" package
// In this case, calculate_pi provides the functions required to set up
// an indicator function in_or_out(x,y) = (r < r0 ? 1 : 0), and compute the area
// of a circle of radius r0 as A = \int d^x in_or_out(x,y) over the domain. Then
// pi \approx A/r0^2
namespace calculate_pi {

void SetInOrOut(MeshBlockData<Real> *rc) {
  auto pmb = rc->GetBlockPointer();

  ParArrayND<Real> v;
  const auto &radius = pmb->packages.Get("calculate_pi")->Param<Real>("radius");

  // If we're using sparse variables, we only allocate in_or_out on blocks that have at
  // least some part inside the circle, otherwise it would be all 0's
  bool const use_sparse = pmb->packages.Get("calculate_pi")->Param<bool>("use_sparse");
  if (use_sparse) {
    auto &bs = pmb->block_size;
    // check if block falls on radius.
    Real coords[4][2] = {{bs.xmin(X1DIR), bs.xmin(X2DIR)},
                         {bs.xmin(X1DIR), bs.xmax(X2DIR)},
                         {bs.xmax(X1DIR), bs.xmin(X2DIR)},
                         {bs.xmax(X1DIR), bs.xmax(X2DIR)}};

    bool fully_outside = true;

    for (auto i = 0; i < 4; i++) {
      Real const rsq = coords[i][0] * coords[i][0] + coords[i][1] * coords[i][1];
      if (rsq < radius * radius) {
        fully_outside = false;
        break;
      }
    }

    // If any part of the block falls inside, then we need to allocate the sparse id
    // before computing on it. If it's fully outside, then it would be all 0's and we
    // don't need to allocate in_or_out on this block
    if (fully_outside) {
      // block is fully outside of circle, do nothing
      return;
    }

    pmb->AllocSparseID("in_or_out", 0);
    v = rc->Get("in_or_out", 0).data;
  } else {
    v = rc->Get("in_or_out").data;
  }

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  auto &coords = pmb->coords;
  // Set an indicator function that indicates whether the cell center
  // is inside or outside of the circle we're interating the area of.
  // Loop bounds are set to catch the case where the edge is between the
  // cell centers of the first/last real cell and the first ghost cell
  pmb->par_for(
      PARTHENON_AUTO_LABEL, kb.s, kb.e, jb.s - 1, jb.e + 1, ib.s - 1, ib.e + 1,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        Real rsq = std::pow(coords.Xc<1>(i), 2) + std::pow(coords.Xc<2>(j), 2);
        if (rsq < radius * radius) {
          v(k, j, i) = 1.0;
        } else {
          v(k, j, i) = 0.0;
        }
      });
}

void SetInOrOutBlock(MeshBlock *pmb, ParameterInput *pin) {
  MeshBlockData<Real> *rc = pmb->meshblock_data.Get().get();
  SetInOrOut(rc);
}

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto package = std::make_shared<StateDescriptor>("calculate_pi");
  Params &params = package->AllParams();

  Real radius = pin->GetOrAddReal("Pi", "radius", 1.0);
  params.Add("radius", radius);

  bool use_sparse = pin->GetOrAddBoolean("Pi", "use_sparse", false);
  params.Add("use_sparse", use_sparse);

  if (use_sparse) {
    // rename "in_or_out" field referenced in input file to "in_or_out_0"
    pin->SetString("parthenon/refinement0", "field", "in_or_out_0");
    pin->SetString("parthenon/output0", "variables", "in_or_out_0");
  }

  // add a variable called in_or_out that will hold the value of the indicator function
  std::string field_name("in_or_out");
  Metadata m({Metadata::Cell, Metadata::Derived, Metadata::OneCopy});
  if (use_sparse) {
    m.Set(Metadata::Sparse);
    package->AddSparsePool(field_name, m, std::vector<int>{0});
  } else {
    package->AddField(field_name, m);
  }

  // All the package FillDerived and CheckRefinement functions are called by parthenon
  // We could use the package FillDerived, which is called every "cycle" as below.
  // Instead in this example, we use the InitMeshBlockUserData, which happens
  // only when the mesh is created or changes.
  // package->FillDerivedBlock = SetInOrOut;
  // could use package specific refinement tagging routine (see advection example), but
  // instead this example will make use of the parthenon shipped first derivative
  // criteria, as invoked in the input file
  // package->CheckRefinementBlock = CheckRefinement;

  return package;
}

template <typename CheckAllocated>
Real ComputeAreaInternal(MeshBlockPack<VariablePack<Real>> pack, ParArrayHost<Real> areas,
                         const IndexShape &cellbounds, CheckAllocated &&check_allocated) {
  const IndexRange ib = cellbounds.GetBoundsI(IndexDomain::interior);
  const IndexRange jb = cellbounds.GetBoundsJ(IndexDomain::interior);
  const IndexRange kb = cellbounds.GetBoundsK(IndexDomain::interior);

  Real area = 0.0;
  par_reduce(
      parthenon::loop_pattern_mdrange_tag, "calculate_pi compute area",
      parthenon::DevExecSpace(), 0, pack.GetDim(5) - 1, 0, pack.GetDim(4) - 1, kb.s, kb.e,
      jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(int b, int v, int k, int j, int i, Real &larea) {
        // Must check if in_or_out is allocated for sparse variables
        if (check_allocated(b, v)) {
          larea +=
              pack(b, v, k, j, i) * pack.GetCoords(b).FaceArea<parthenon::X3DIR>(k, j, i);
        }
      },
      area);
  return area;
}

TaskStatus ComputeArea(std::shared_ptr<MeshData<Real>> &md, ParArrayHost<Real> areas,
                       int i) {
  bool const use_sparse =
      md->GetMeshPointer()->packages.Get("calculate_pi")->Param<bool>("use_sparse");

  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  const auto &cellbounds = pmb->cellbounds;

  PackIndexMap imap; // PackIndex map can be used to get the index in
                     // a pack of a specific variable
  // This call signature works
  const auto &pack = use_sparse
                         ? md->PackVariables(std::vector<std::string>({"in_or_out"}),
                                             std::vector<int>{0}, imap)

                         // and so does this one
                         : md->PackVariables(std::vector<std::string>({"in_or_out"}));

  areas(i) = use_sparse
                 ? ComputeAreaInternal(
                       pack, areas, cellbounds,
                       KOKKOS_LAMBDA(int const b, int const v) {
                         return pack.IsAllocated(b, v);
                       })
                 : ComputeAreaInternal(
                       pack, areas, cellbounds, KOKKOS_LAMBDA(int, int) { return true; });
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
