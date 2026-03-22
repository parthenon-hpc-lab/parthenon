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

// This file was made in part with generative AI

#ifndef PACK_SWARM_PACK_BASE_HPP_
#define PACK_SWARM_PACK_BASE_HPP_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "interface/swarm_device_context.hpp"
#include "kokkos_abstraction.hpp"
#include "pack/swarm_pack/swarm_pack_descriptor.hpp"
#include "utils/utils.hpp"

namespace parthenon {

template <typename TYPE>
class SwarmPackBase;

namespace impl {
template <typename TYPE, class T>
SwarmPackBase<TYPE> GetSwarmPack(T *pmd, const SwarmPackDescriptor<TYPE> &desc);
} // namespace impl

template <typename TYPE>
class SwarmPackCache;

template <typename TYPE>
class SwarmPackBase {
 public:
  SwarmPackBase() = default;
  virtual ~SwarmPackBase() = default;

  using pack_t = ParArray3DRaw<ParArray1D<TYPE>>;
  using bounds_t = ParArray3D<int>;
  using contexts_t = ParArray1DRaw<SwarmDeviceContext>;
  using contexts_h_t = typename contexts_t::HostMirror;
  using max_active_indices_t = ParArray1D<int>;
  using desc_t = impl::SwarmPackDescriptor<TYPE>;
  using idx_map_t = std::unordered_map<std::string, std::size_t>;

 protected:
  friend class SwarmPackCache<TYPE>;
  template <typename TOUT, class T>
  friend SwarmPackBase<TOUT> impl::GetSwarmPack(T *pmd,
                                                const SwarmPackDescriptor<TOUT> &desc);

  // Build supplemental entries to SwarmPack that change on a cadence faster than the
  // persistent cache contents. This mirrors the sparse-pack split where the cache owns
  // reuse and the pack base owns construction details.
  template <class T>
  static void BuildSupplemental(T *pmd, const SwarmPackDescriptor<TYPE> &desc,
                                SwarmPackBase<TYPE> &pack);

  // Actually build a `SwarmPackBase` (i.e. create a view of views, fill on host, and
  // deep copy the view of views to device) from the variables specified in desc contained
  // from the blocks contained in pmd (which can either be MeshBlockData/MeshData).
  template <class T>
  static SwarmPackBase<TYPE> Build(T *pmd, const SwarmPackDescriptor<TYPE> &desc);

  template <class T>
  static SwarmPackBase<TYPE> GetPack(T *pmd,
                                     const impl::SwarmPackDescriptor<TYPE> &desc);

  static idx_map_t GetIdxMap(const desc_t &desc);

  pack_t pack_;
  bounds_t bounds_;
  contexts_t contexts_;
  contexts_h_t contexts_h_;
  max_active_indices_t max_active_indices_;
  max_active_indices_t flat_index_map_;

  int nblocks_;
  int nvar_;
  int max_flat_index_;
};

} // namespace parthenon

#endif // PACK_SWARM_PACK_BASE_HPP_
