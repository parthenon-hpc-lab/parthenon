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

#include "sparse/sparse_management.hpp"

namespace parthenon {

template <typename T>
void SparseDeallocOnCount(T *rc, std::size_t count) {
  auto [pack, packIdx, control_vars, is_zero_h] = SparseCheckIsZero(rc);
  if (!Globals::sparse_config.enabled || (pack.GetNBlocks() < 1)) {
    return;
  }

  for (int b = 0; b < pack.GetNBlocks(); ++b) {
    auto pmbdata = GetBlockDataPointer(rc, b);
    auto pmb = pmbdata->GetBlockPointer();
    for (auto &control_var : control_vars) {
      int lo = pack.GetLowerBoundHost(b, PackIdx(packIdx[control_var]));
      int hi = pack.GetUpperBoundHost(b, PackIdx(packIdx[control_var]));
      if (lo <= hi) { // Check that this control variable is actually in the pack
        auto &counter = pmbdata->Get(control_var).dealloc_count;
        bool all_zero = true;
        for (int iv = lo; iv <= hi; ++iv)
          all_zero = all_zero && is_zero_h(b, iv);
        if (all_zero) {
          counter++;
        } else {
          counter = 0;
        }
        if (counter > count) {
          // this variable has been flagged for deallocation deallocation_count times in
          // a row, now deallocate it
          counter = 0;
          pmb->DeallocateSparse(control_var);
        }
      }
    }
  }
}

template void SparseDeallocOnCount<MeshData<Real>>(MeshData<Real> *, std::size_t);
template void SparseDeallocOnCount<MeshBlockData<Real>>(MeshBlockData<Real> *,
                                                        std::size_t);

} // namespace parthenon
