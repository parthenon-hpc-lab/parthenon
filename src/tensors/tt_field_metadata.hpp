//========================================================================================
// (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
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

#ifndef TENSORS_TT_FIELD_METADATA_HPP
#define TENSORS_TT_FIELD_METADATA_HPP

#include <memory>
#include <vector>

#include "interface/metadata.hpp"
#include "utils/indexer.hpp"

namespace parthenon {

class MeshBlock;

// Descriptor for a tensor-train field registered with a package. Unlike a
// regular field, a tensor train has dynamic ranks that change every timestep,
// so its data cannot live in fixed-size Variable<T>/ParArrayND storage. This
// struct captures the *static* structure of a TT field: the number of cores and
// the physical dimension of each core.
//
// The first (spatial) core's physical dimension is NOT stored here. It is
// derived per-block from the mesh geometry and the topological type carried by
// `metadata` (Cell/Face/Node), because it depends on the block's cell count
// (including ghost zones). Only the trailing, non-spatial core dimensions --
// e.g. the angular dimensions {NTHETA, NPHI} for radiation transport -- are
// registered here.
struct TTFieldMetadata {
  // Physical dimensions of the non-spatial cores only. size() == NCores() - 1.
  std::vector<int> phys_dims;
  // Flags/queries, package-ownership flag, and the topological type used to
  // derive the spatial core dimension.
  Metadata metadata;

  TTFieldMetadata() = default;
  TTFieldMetadata(std::vector<int> phys_dims_in, Metadata metadata_in)
      : phys_dims(std::move(phys_dims_in)), metadata(std::move(metadata_in)) {}

  // Total number of cores in the train: one spatial core plus the registered
  // non-spatial cores.
  int NCores() const { return static_cast<int>(phys_dims.size()) + 1; }

  std::vector<Indexer6D> CoreIndexers(std::weak_ptr<MeshBlock> wpmb,
                                      bool coarse = false) const {
    const auto dims = metadata.GetArrayDims(wpmb, coarse);
    std::vector<Indexer6D> out;
    out.reserve(NCores());
    out.push_back(Indexer6D({0, dims[5] - 1}, {0, dims[4] - 1}, {0, dims[3] - 1},
                            {0, dims[2] - 1}, {0, dims[1] - 1}, {0, dims[0] - 1}));
    for (int dd : phys_dims)
      out.push_back(Indexer6D({0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, dd - 1}));
    return out;
  }
};

} // namespace parthenon

#endif // TENSORS_TT_FIELD_METADATA_HPP
