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

#include <optional>
#include <string>
#include <unordered_set>
#include <utility>

#include "basic_types.hpp"
#include "defs.hpp"
#include "interface/metadata.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/domain.hpp"
#include "pack/make_pack_descriptor.hpp"
#include "pack/sparse_pack.hpp"

#ifndef SPARSE_SPARSE_MANAGEMENT_HPP_
#define SPARSE_SPARSE_MANAGEMENT_HPP_

namespace parthenon {

template <typename T>
TaskStatus InitNewlyAllocatedVars(T *rc);

TaskStatus SparseDealloc(MeshData<Real> *md);

template <typename T>
void SparseDeallocOnCount(T *rc, std::size_t count,
                          const std::unordered_set<std::string> &exclude = {});

extern template TaskStatus
InitNewlyAllocatedVars<MeshBlockData<Real>>(MeshBlockData<Real> *rc);
extern template TaskStatus InitNewlyAllocatedVars<MeshData<Real>>(MeshData<Real> *rc);

extern template void
SparseDeallocOnCount<MeshData<Real>>(MeshData<Real> *, std::size_t,
                                     const std::unordered_set<std::string> &);
extern template void
SparseDeallocOnCount<MeshBlockData<Real>>(MeshBlockData<Real> *, std::size_t,
                                          const std::unordered_set<std::string> &);

} // namespace parthenon
#endif // SPARSE_SPARSE_MANAGEMENT_HPP_
