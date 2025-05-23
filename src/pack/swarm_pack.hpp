//========================================================================================
// (C) (or copyright) 2020-2024. Triad National Security, LLC. All rights reserved.
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
#ifndef PACK_SWARM_PACK_HPP_
#define PACK_SWARM_PACK_HPP_

#include <algorithm>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "interface/mesh_data.hpp"
#include "interface/meshblock_data.hpp"
#include "interface/swarm.hpp"
#include "interface/variable.hpp"
#include "pack/pack_utils.hpp"
#include "pack/swarm_pack_base.hpp"
#include "utils/concepts_lite.hpp"
#include "utils/utils.hpp"

namespace parthenon {

template <typename TYPE, class... Ts>
class SwarmPack : public SwarmPackBase<TYPE> {
 public:
  using SwarmPackBase<TYPE>::pack_;
  using SwarmPackBase<TYPE>::bounds_;
  using SwarmPackBase<TYPE>::contexts_;
  using SwarmPackBase<TYPE>::max_active_indices_;
  using SwarmPackBase<TYPE>::flat_index_map_;
  using SwarmPackBase<TYPE>::nvar_;
  using SwarmPackBase<TYPE>::nblocks_;
  using SwarmPackBase<TYPE>::max_flat_index_;

  explicit SwarmPack(const SwarmPackBase<TYPE> &spb) : SwarmPackBase<TYPE>(spb) {
    if constexpr (sizeof...(Ts) != 0) {
      static_assert(std::is_same<TYPE, typename GetDataType<Ts...>::value>::value,
                    "Type mismatch in SwarmPack! When passing type-based variables as "
                    "template argument to SwarmPack, ensure that the first template "
                    "parameter is a data type (e.g., Real or int) that matches the "
                    "data type of subsequent variable types!");
    }
  }

  class Descriptor : public impl::SwarmPackDescriptor<TYPE> {
   public:
    Descriptor() = default;
    explicit Descriptor(const impl::SwarmPackDescriptor<TYPE> &desc_in)
        : impl::SwarmPackDescriptor<TYPE>(desc_in) {}

    template <class T>
    SwarmPack GetPack(T *pmd) const {
      return SwarmPack(SwarmPackBase<TYPE>::GetPack(pmd, *this));
    }

    SparsePackIdxMap GetMap() const {
      PARTHENON_REQUIRE(sizeof...(Ts) == 0,
                        "Should not be getting an IdxMap for a type based pack");
      return SwarmPackBase<TYPE>::GetIdxMap(*this);
    }
  };

  // Methods for getting parts of the shape of the pack
  KOKKOS_FORCEINLINE_FUNCTION
  int GetNBlocks() const { return nblocks_; }

  // inclusive max index of active particles at block level.
  KOKKOS_FORCEINLINE_FUNCTION
  const int &GetMaxActiveIndex(const int b = 0) const { return max_active_indices_(b); }

  // inclusive max index in the space flattened over blocks and particles.
  KOKKOS_FORCEINLINE_FUNCTION
  const auto &GetMaxFlatIndex() const { return max_flat_index_; }

  // map from flat index to block and particle indices
  KOKKOS_FORCEINLINE_FUNCTION
  auto GetBlockParticleIndices(const int idx) const {
    PARTHENON_REQUIRE(idx >= 0 && idx <= max_flat_index_,
                      "Requesting an out-of-bounds index");
    // binary search to figure out what block we're on
    int b = 0;
    int r = nblocks_;
    while (r - b > 1) {
      auto c = static_cast<int>(0.5 * (b + r));
      if (flat_index_map_(c) > idx)
        r = c;
      else
        b = c;
    }
    return std::make_tuple(b, idx - flat_index_map_(b));
  }

  KOKKOS_FORCEINLINE_FUNCTION
  const SwarmDeviceContext &GetContext(const int b = 0) const { return contexts_(b); }

  // Bound overloads
  KOKKOS_INLINE_FUNCTION int GetLowerBound(const int b) const { return bounds_(0, b, 0); }

  KOKKOS_INLINE_FUNCTION int GetUpperBound(const int b) const {
    return bounds_(1, b, nvar_ - 1);
  }

  KOKKOS_INLINE_FUNCTION int GetLowerBound(const int b, PackIdx idx) const {
    static_assert(sizeof...(Ts) == 0, "Cannot create a string/type hybrid pack");
    return bounds_(0, b, idx.VariableIdx());
  }

  KOKKOS_INLINE_FUNCTION int GetUpperBound(const int b, PackIdx idx) const {
    static_assert(sizeof...(Ts) == 0, "Cannot create a string/type hybrid pack");
    return bounds_(1, b, idx.VariableIdx());
  }

  template <class TIn, REQUIRES(IncludesType<TIn, Ts...>::value)>
  KOKKOS_INLINE_FUNCTION int GetLowerBound(const int b, const TIn &) const {
    const int vidx = GetTypeIdx<TIn, Ts...>::value;
    return bounds_(0, b, vidx);
  }

  template <class TIn, REQUIRES(IncludesType<TIn, Ts...>::value)>
  KOKKOS_INLINE_FUNCTION int GetUpperBound(const int b, const TIn &) const {
    const int vidx = GetTypeIdx<TIn, Ts...>::value;
    return bounds_(1, b, vidx);
  }

  // operator() overloads
  KOKKOS_INLINE_FUNCTION
  auto &operator()(const int b, const int idx) const { return pack_(0, b, idx); }

  KOKKOS_INLINE_FUNCTION
  auto &operator()(const int b, const int idx, const int n) const {
    return pack_(0, b, idx)(n);
  }

  template <class TIn, REQUIRES(IncludesType<TIn, Ts...>::value)>
  KOKKOS_INLINE_FUNCTION auto &operator()(const int b, const TIn &t, const int n) const {
    const int vidx = GetLowerBound(b, t) + t.idx;
    return pack_(0, b, vidx)(n);
  }
};

} // namespace parthenon

#endif // PACK_SWARM_PACK_HPP_
