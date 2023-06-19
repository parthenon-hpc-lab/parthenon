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

template <int NDIM = 3>
constexpr uint64_t GetInterleaveConstant(int power) { 
    // For power = 2, NDIM = 3, this should return 
    // ...000011000011
    // For power = 1, NDIM = 3, this should return 
    // ...001001001001
    // For power = 2, NDIM = 2, this should return 
    // ...001100110011
    // etc.
    uint64_t i_const = ~((~static_cast<uint64_t>(0)) << power); //std::pow(2, power) - 1; 
    int cur_shift = sizeof(uint64_t) * 8 * NDIM; // Works for anything that will fit in uint64_t
    while (cur_shift >= NDIM * power) {
      i_const = (i_const << cur_shift) | i_const;
      cur_shift /= 2;
    }
    return i_const;
}

template <int NDIM = 3, int N_VALID_BITS = 21> 
uint64_t InterleaveZeros(uint64_t x) {
  // This is a standard bithack for interleaving zeros in binary numbers to make a Morton number
  if constexpr (N_VALID_BITS >= 64) x = (x | x << 64 * (NDIM - 1)) & GetInterleaveConstant<NDIM>(64); 
  if constexpr (N_VALID_BITS >= 32) x = (x | x << 32 * (NDIM - 1)) & GetInterleaveConstant<NDIM>(32); 
  if constexpr (N_VALID_BITS >= 16) x = (x | x << 16 * (NDIM - 1)) & GetInterleaveConstant<NDIM>(16); 
  if constexpr (N_VALID_BITS >= 8)  x = (x | x << 8  * (NDIM - 1)) & GetInterleaveConstant<NDIM>(8);
  if constexpr (N_VALID_BITS >= 4)  x = (x | x << 4  * (NDIM - 1)) & GetInterleaveConstant<NDIM>(4);
  if constexpr (N_VALID_BITS >= 2)  x = (x | x << 2  * (NDIM - 1)) & GetInterleaveConstant<NDIM>(2);
  if constexpr (N_VALID_BITS >= 1)  x = (x | x << 1  * (NDIM - 1)) & GetInterleaveConstant<NDIM>(1);
  return x;
}

uint64_t GetMortonNumber(uint64_t x, uint64_t y, uint64_t z) { 
  return InterleaveZeros<3>(z) << 2 | InterleaveZeros<3>(y) << 1 | InterleaveZeros<3>(x);  
}

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
  parthenon::ParArray1D<int> morton("Morton number", pack.GetNBlocks());  
  auto x_morton_h = Kokkos::create_mirror_view(x_morton); 
  auto y_morton_h = Kokkos::create_mirror_view(y_morton); 
  auto z_morton_h = Kokkos::create_mirror_view(z_morton); 
  auto morton_h = Kokkos::create_mirror_view(morton); 
  for (int b = 0; b < md->NumBlocks(); ++b) { 
    auto cpmb = md->GetBlockData(b)->GetBlockPointer();
    auto level = cpmb->loc.level;
    auto mx = cpmb->loc.lx1 << (2 - level); 
    auto my = cpmb->loc.lx2 << (2 - level); 
    auto mz = cpmb->loc.lx3 << (2 - level); 
    auto mort_tot = GetMortonNumber(mx, my, mz);
    printf("gid = %i (%i, %i, %i) %i [%i: %i %i]\n", cpmb->gid, mx, my, mz, mort_tot, level, cpmb->loc.lx1, cpmb->loc.lx2);
    // Here we are assuming the maximum level is one 
    morton_h(b) = mort_tot;
    x_morton_h(b) = mx; 
    y_morton_h(b) = my; 
    z_morton_h(b) = mz; 
  }
  Kokkos::deep_copy(x_morton, x_morton_h);
  Kokkos::deep_copy(y_morton, y_morton_h);
  Kokkos::deep_copy(z_morton, z_morton_h);
  Kokkos::deep_copy(morton, morton_h);
  
  {
    IndexRange ib = md->GetBoundsI(IndexDomain::interior);
    IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
    IndexRange kb = md->GetBoundsK(IndexDomain::interior);
    parthenon::par_for(parthenon::loop_pattern_mdrange_tag, "SetMorton", DevExecSpace(), 
        0, pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          pack(b, morton_num(0), k, j, i) = morton(b); 
          pack(b, morton_num(1), k, j, i) = x_morton(b); 
          pack(b, morton_num(2), k, j, i) = y_morton(b); 
          pack(b, morton_num(3), k, j, i) = z_morton(b); 
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
