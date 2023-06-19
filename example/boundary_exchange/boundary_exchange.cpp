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
#include "boundary_exchange.hpp"

using namespace parthenon::package::prelude;
using parthenon::IndexShape;

namespace boundary_exchange {

TaskStatus SetBlockValues(MeshData<Real> *md) {
  auto pmesh = md->GetMeshPointer(); 
  auto desc = parthenon::MakePackDescriptor<morton_num>(pmesh->resolved_packages.get()); 
  auto pack = desc.GetPack(md); 
  {
    IndexRange ib = md->GetBoundsI(IndexDomain::entire);
    IndexRange jb = md->GetBoundsJ(IndexDomain::entire);
    IndexRange kb = md->GetBoundsK(IndexDomain::entire);
    parthenon::par_for(parthenon::loop_pattern_mdrange_tag, "SetNaN", DevExecSpace(), 
        0, pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          pack(b, morton_num(0), k, j, i) = std::numeric_limits<Real>::quiet_NaN(); 
          pack(b, morton_num(1), k, j, i) = std::numeric_limits<Real>::quiet_NaN(); 
          pack(b, morton_num(2), k, j, i) = std::numeric_limits<Real>::quiet_NaN(); 
        });
  }

  // Get the morton numbers of the blocks in the pack onto device
  parthenon::ParArray1D<int> x_morton("x Morton number", pack.GetNBlocks());  
  parthenon::ParArray1D<int> y_morton("y Morton number", pack.GetNBlocks());  
  parthenon::ParArray1D<int> z_morton("z Morton number", pack.GetNBlocks());  
  auto x_morton_h = Kokkos::create_mirror_view(x_morton); 
  auto y_morton_h = Kokkos::create_mirror_view(y_morton); 
  auto z_morton_h = Kokkos::create_mirror_view(z_morton); 
  for (int b; b < md->NumBlocks(); ++b) { 
    auto cpmb = md->GetBlockData(b)->GetBlockPointer();
    x_morton_h(b) = cpmb->loc.lx1; 
    y_morton_h(b) = cpmb->loc.lx2; 
    z_morton_h(b) = cpmb->loc.lx3; 
  }
  Kokkos::deep_copy(x_morton, x_morton_h);
  Kokkos::deep_copy(y_morton, y_morton_h);
  Kokkos::deep_copy(z_morton, z_morton_h);
  
  {
    IndexRange ib = md->GetBoundsI(IndexDomain::interior);
    IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
    IndexRange kb = md->GetBoundsK(IndexDomain::interior);
    parthenon::par_for(parthenon::loop_pattern_mdrange_tag, "SetMorton", DevExecSpace(), 
        0, pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          pack(b, morton_num(0), k, j, i) = x_morton(b); 
          pack(b, morton_num(1), k, j, i) = y_morton(b); 
          pack(b, morton_num(2), k, j, i) = z_morton(b); 
        });
  }
  return TaskStatus::complete; 
}

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto package = std::make_shared<StateDescriptor>("boundary_exchange");
  Params &params = package->AllParams();
  
  Metadata m({Metadata::Cell, Metadata::Independent, Metadata::FillGhost}, morton_num::shape()); 
  package->AddField(morton_num::name(), m);

  return package;
}

} // namespace boundary_exchange
