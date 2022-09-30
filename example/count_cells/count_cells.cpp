//========================================================================================
// (C) (or copyright) 2022. Triad National Security, LLC. All rights reserved.
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

// C++ includes
#include <array>

// Parthenon Includes
#include <interface/state_descriptor.hpp>
#include <parthenon/package.hpp>

#include "count_cells.hpp"

using parthenon::ParameterInput;
using parthenon::Params;
using parthenon::Real;
using parthenon::StateDescriptor;

namespace count_cells {
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto package = std::make_shared<StateDescriptor>("count_cells");
  Params &params = package->AllParams();

  Real radius = pin->GetOrAddReal("count_cells", "sphere_radius", 1.0);
  params.Add("radius", radius);

  Real dx1_target = pin->GetOrAddReal("count_cells", "dx1_target", 0.1);
  Real dx2_target = pin->GetOrAddReal("count_cells", "dx2_target", 0.1);
  Real dx3_target = pin->GetOrAddReal("count_cells", "dx3_target", 0.1);
  std::array<Real, 3> dx_target{dx1_target, dx2_target, dx3_target};
  params.Add("dx_target", dx_target);

  package->CheckRefinementBlock = CheckRefinement;

  return package;
}

AmrTag CheckRefinement(MeshBlockData<Real> *rc) {
  auto pmb = rc->GetBlockPointer();
  auto pkg = pmb->packages.Get("count_cells");
  const auto &coords = pmb->coords;
  if (BlockInRegion(pkg.get(), pmb.get()) && !SufficientlyRefined(pkg.get(), coords)) {
    return AmrTag::refine;
  }
  return AmrTag::same;
}

bool BlockInRegion(const StateDescriptor *pkg, MeshBlock *pmb) {
  const auto radius = pkg->Param<Real>("radius");
  const auto coords = pmb->coords;

  auto ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  auto jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  auto kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  // Loop through all cell centers in the block to determine if block
  // intersects with region. Technically this could still miss some.
  // Could improve fidelity by also looping through cell face and nodes.
  // Note this doesn't require actually allocating any cells, but you
  // do have to loop through them.
  int num_intersections = 0;
  pmb->par_reduce(
      "Check intersections of cells with regions", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(int k, int j, int i, int &n) {
        Real x1 = coords.x1v(k, j, i);
        Real x2 = coords.x2v(k, j, i);
        Real x3 = coords.x3v(k, j, i);
        if (std::sqrt(x1 * x1 + x2 * x2 + x3 * x3) <= radius) {
          n += 1;
        }
      },
      num_intersections);

  return num_intersections > 0;
}

bool SufficientlyRefined(const StateDescriptor *pkg, const Coordinates_t &coords) {
  const auto &dx_target = pkg->Param<std::array<Real, 3>>("dx_target");
  for (int d = 1; d <= 3; ++d) {
    // assumes uniform cartesian coordinates
    // which have constant Dx accross whole meshblock
    if (coords.Dx(d) > dx_target[d - 1]) return false;
  }
  return true;
}

void CountCells(Mesh *pmesh) {
  // a representative meshblock
  auto pmb = pmesh->block_list[0];

  const size_t mb_ncells_interior = pmb->cellbounds.GetTotal(IndexDomain::interior);
  const size_t mb_ncells_total = pmb->cellbounds.GetTotal(IndexDomain::entire);
  const size_t mb_ncells_ghost = mb_ncells_total - mb_ncells_interior;

  // includes 3 flux buffers + coarse buffer + comm buffers
  const size_t mb_ncells_with_extra_buffs = 5 * mb_ncells_total + mb_ncells_ghost;

  const size_t num_blocks = pmesh->block_list.size();

  Real ncells_interior = num_blocks * mb_ncells_interior;
  Real ncells_total = num_blocks * mb_ncells_total;
  Real ncells_ghost = num_blocks * mb_ncells_ghost;
  Real ncells_with_extra_buffs = num_blocks * mb_ncells_with_extra_buffs;

  std::cout << std::scientific
	    << "num blocks         = " << std::setw(14) << num_blocks << "\n"
            << "num cells interior = " << std::setw(14) << ncells_interior << "\n"
            << "num cells total    = " << std::setw(14) << ncells_total << "\n"
            << "num ghosts         = " << std::setw(14) << ncells_ghost << "\n"
            << "num with comms etc = " << std::setw(14) << ncells_with_extra_buffs
	    << std::endl;
}
} // namespace count_cells
