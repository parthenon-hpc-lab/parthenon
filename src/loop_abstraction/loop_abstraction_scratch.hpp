//========================================================================================
// (C) (or copyright) 2024-2026. Triad National Security, LLC. All rights reserved.
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
#ifndef LOOP_ABSTRACTION_LOOP_ABSTRACTION_SCRATCH_HPP_
#define LOOP_ABSTRACTION_LOOP_ABSTRACTION_SCRATCH_HPP_

// This file was made in part with generative AI.

#include <algorithm>
#include <array>
#include <concepts>
#include <optional>
#include <typeindex>
#include <unordered_map>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "basic_types.hpp"
#include "interface/mesh_data.hpp"
#include "kokkos_types.hpp"
#include "mesh/mesh.hpp"
#include "pack/pack_utils.hpp"
#include "utils/bump_arena.hpp"
#include "utils/indexer.hpp"

namespace parthenon::loop_abstraction {

template <class Halo>
struct HaloBox {
  static constexpr int min_k = [] {
    int out = Halo::dk(0);
    for (int n = 1; n < Halo::npoints; ++n) out = std::min(out, Halo::dk(n));
    return out;
  }();
  static constexpr int max_k = [] {
    int out = Halo::dk(0);
    for (int n = 1; n < Halo::npoints; ++n) out = std::max(out, Halo::dk(n));
    return out;
  }();
  static constexpr int min_j = [] {
    int out = Halo::dj(0);
    for (int n = 1; n < Halo::npoints; ++n) out = std::min(out, Halo::dj(n));
    return out;
  }();
  static constexpr int max_j = [] {
    int out = Halo::dj(0);
    for (int n = 1; n < Halo::npoints; ++n) out = std::max(out, Halo::dj(n));
    return out;
  }();
  static constexpr int min_i = [] {
    int out = Halo::di(0);
    for (int n = 1; n < Halo::npoints; ++n) out = std::min(out, Halo::di(n));
    return out;
  }();
  static constexpr int max_i = [] {
    int out = Halo::di(0);
    for (int n = 1; n < Halo::npoints; ++n) out = std::max(out, Halo::di(n));
    return out;
  }();
  static constexpr int nk = max_k - min_k + 1;
  static constexpr int nj = max_j - min_j + 1;
  static constexpr int ni = max_i - min_i + 1;
  static constexpr int size = nk * nj * ni;
};

template <std::size_t... Dims>
struct ctime_flat_indexer {
  static constexpr std::size_t ndim = sizeof...(Dims);
  static constexpr std::size_t size = (Dims * ... * std::size_t{1});
  static constexpr std::array<std::size_t, ndim> dim_sizes{Dims...};

  template <class... Args>
    requires(sizeof...(Args) >= ndim) // We allow for unused trailing arguments to simplify template code
  KOKKOS_FORCEINLINE_FUNCTION
  static constexpr std::size_t GetFlat(Args&&... args) {
    auto tup = std::forward_as_tuple(std::forward<Args>(args)...);
    return GetFlatImpl(std::make_index_sequence<ndim>{}, tup);
  }

 private:
  template <std::size_t... I, class Tuple>
  KOKKOS_FORCEINLINE_FUNCTION
  static constexpr std::size_t GetFlatImpl(std::index_sequence<I...>, Tuple &&tup) {
    std::size_t flat_idx{0};
    ([&]{
      KOKKOS_ASSERT(static_cast<std::size_t>(std::get<I>(tup)) < dim_sizes[I]);
      flat_idx += std::get<I>(tup);
      if constexpr (I + 1 < ndim) {
        flat_idx *= dim_sizes[I + 1];
      }
    }(), ...);
    return flat_idx;
  }
};

template <class T, class IndexRange, std::size_t... Dims>
struct StackScratch1D {
  using halo_t = typename IndexRange::halo_t;
  using box_t = HaloBox<halo_t>;
  using idxer_t = ctime_flat_indexer<Dims...>;

  mutable std::array<T, box_t::size * idxer_t::size> data{};
  int ks = 0;
  int js = 0;
  int is = 0;

  KOKKOS_INLINE_FUNCTION
  explicit StackScratch1D(const IndexRange &idx_range)
      : ks(idx_range.ks), js(idx_range.js), is(idx_range.is) {}

  // Version called with component indices first and point index last.
  template <class... Args>
    requires(sizeof...(Args) == idxer_t::ndim + 1 ||
             sizeof...(Args) == idxer_t::ndim + 3)
  KOKKOS_FORCEINLINE_FUNCTION T &operator()(Args&&... all) const {
    auto tup = std::forward_as_tuple(std::forward<Args>(all)...);
    const auto dense_index = [&]{
      if constexpr (sizeof...(Args) == idxer_t::ndim + 1) {
        return GetDenseIndex(std::get<idxer_t::ndim>(tup));
      } else if constexpr (sizeof...(Args) == idxer_t::ndim + 3) {
        return GetDenseIndex(std::get<idxer_t::ndim>(tup),
                             std::get<idxer_t::ndim + 1>(tup),
                             std::get<idxer_t::ndim + 2>(tup));
      }
    }();
    return data[dense_index + box_t::size * idxer_t::GetFlat(all...)];
  }

  KOKKOS_FORCEINLINE_FUNCTION
  constexpr std::size_t size() const {
    return box_t::size * idxer_t::size;
  }

  // Zero the entire buffer. Prefer this over a zero-initializing constructor so a
  // buffer can be reused without reallocating. No barrier is issued (matches
  // inner()); the caller barriers before any cross-thread read.
  KOKKOS_FORCEINLINE_FUNCTION void Zero() const {
#pragma omp simd
    for (std::size_t i = 0; i < data.size(); ++i) data[i] = T{};
  }

  KOKKOS_FORCEINLINE_FUNCTION
  constexpr std::size_t shmem_size() const {
    return 0;
  }

 private:
  KOKKOS_FORCEINLINE_FUNCTION static bool ContainsDeclaredOffset(const int dk,
                                                                 const int dj,
                                                                 const int di) {
    for (int n = 0; n < halo_t::npoints; ++n) {
      if (dk == halo_t::dk(n) && dj == halo_t::dj(n) && di == halo_t::di(n)) {
        return true;
      }
    }
    return false;
  }

  KOKKOS_FORCEINLINE_FUNCTION int GetDenseIndex(int k, int j, int i) const {
    return DenseIndex(k - ks, j - js, i - is);
  }

  KOKKOS_FORCEINLINE_FUNCTION int GetDenseIndex(Index3 idx) const {
    return DenseIndex(idx.k - ks, idx.j - js, idx.i - is);
  }

  KOKKOS_FORCEINLINE_FUNCTION int GetDenseIndex(MemoryOffset idx) const {
    return DenseIndex(idx.dk, idx.dj, idx.di);
  }

  KOKKOS_FORCEINLINE_FUNCTION static int DenseIndex(const int dk, const int dj,
                                                    const int di) {
    KOKKOS_ASSERT(dk >= box_t::min_k && dk <= box_t::max_k);
    KOKKOS_ASSERT(dj >= box_t::min_j && dj <= box_t::max_j);
    KOKKOS_ASSERT(di >= box_t::min_i && di <= box_t::max_i);
    KOKKOS_ASSERT(ContainsDeclaredOffset(dk, dj, di));
    return ((dk - box_t::min_k) * box_t::nj + (dj - box_t::min_j)) * box_t::ni +
           (di - box_t::min_i);
  }
};

template <class IndexRange, class T, std::size_t... Dims>
struct HostScratch1D {
  using idxer_t = ctime_flat_indexer<Dims...>;

  IndexRange idx_range;
  // Non-owning view into the per-thread ThreadLocalBumpArena. Storage is
  // bump-allocated (no init) and reclaimed wholesale when outer_raw_for resets the
  // arena at the start of the next outer iteration. Callers that += into this
  // scratch must zero it first, just like the Kokkos team_scratch path.
  // Declaration order matters: `n` must precede `data` because `data`'s
  // initializer reads `n` (members initialize in declaration order).
  std::size_t n;
  T *data;

  HostScratch1D(const IndexRange &idx_range)
      : idx_range(idx_range), n(idx_range.ScratchSize() * idxer_t::size),
        data(static_cast<T *>(
            parthenon::GetThreadLocalBumpArena().allocate(n * sizeof(T)))) {}

  template <class... Args>
    requires(sizeof...(Args) == idxer_t::ndim + 1 ||
             sizeof...(Args) == idxer_t::ndim + 3)
  KOKKOS_FORCEINLINE_FUNCTION T &operator()(Args&&... all) const {
    auto tup = std::forward_as_tuple(std::forward<Args>(all)...);
    const auto dense_index = [&]{
      if constexpr (sizeof...(Args) == idxer_t::ndim + 1) {
        return idx_range.ScratchIndex(std::get<idxer_t::ndim>(tup));
      } else if constexpr (sizeof...(Args) == idxer_t::ndim + 3) {
        return idx_range.ScratchIndex(std::get<idxer_t::ndim>(tup),
                                      std::get<idxer_t::ndim + 1>(tup),
                                      std::get<idxer_t::ndim + 2>(tup));
      }
    }();
    return data[dense_index + idx_range.ScratchSize() * idxer_t::GetFlat(all...)];
  }

  KOKKOS_FORCEINLINE_FUNCTION
  std::size_t size() const {
    return n;
  }

  // Zero the entire buffer. Prefer this over a zero-initializing allocation so the
  // arena can hand out uninitialized memory and callers zero only what they += into.
  // No barrier is issued (matches inner()); the caller barriers before cross-thread
  // reads. Single-threaded on the raw backend.
  KOKKOS_FORCEINLINE_FUNCTION void Zero() const {
#pragma omp simd
    for (std::size_t i = 0; i < n; ++i) data[i] = T{};
  }

  KOKKOS_FORCEINLINE_FUNCTION
  constexpr std::size_t shmem_size() const {
    return 0;
  }
};

template <class IndexRange, class T, std::size_t... Dims>
struct TeamScratch1D {
  using idxer_t = ctime_flat_indexer<Dims...>;

  static constexpr int scratch_level = 1;
  IndexRange idx_range;
  parthenon::ScratchPad1D<T> data;
  TeamScratch1D(const IndexRange &idx_range) 
      : idx_range(idx_range),
        data(idx_range.team_member->team_scratch(scratch_level),
             idx_range.ScratchSize() * idxer_t::size) {}

  template <class... Args>
    requires(sizeof...(Args) == idxer_t::ndim + 1 ||
             sizeof...(Args) == idxer_t::ndim + 3)
  KOKKOS_FORCEINLINE_FUNCTION T &operator()(Args&&... all) const {
    auto tup = std::forward_as_tuple(std::forward<Args>(all)...);
    const auto dense_index = [&]{
      if constexpr (sizeof...(Args) == idxer_t::ndim + 1) {
        return idx_range.ScratchIndex(std::get<idxer_t::ndim>(tup));
      } else if constexpr (sizeof...(Args) == idxer_t::ndim + 3) {
        return idx_range.ScratchIndex(std::get<idxer_t::ndim>(tup),
                                      std::get<idxer_t::ndim + 1>(tup),
                                      std::get<idxer_t::ndim + 2>(tup));
      }
    }();
    return data(dense_index + idx_range.ScratchSize() * idxer_t::GetFlat(all...));
  }

  KOKKOS_FORCEINLINE_FUNCTION
  std::size_t size() const {
    return idx_range.ScratchSize() * idxer_t::size;
  }

  // Zero the entire buffer, team-parallel over the scratch span. No barrier is
  // issued (matches inner()); the caller barriers before any cross-thread read.
  KOKKOS_FORCEINLINE_FUNCTION void Zero() const {
    const std::size_t n = size();
    Kokkos::parallel_for(Kokkos::TeamVectorRange(*idx_range.team_member, 0, n),
                         [&](const std::size_t i) { data(i) = T{}; });
  }

  KOKKOS_FORCEINLINE_FUNCTION
  std::size_t shmem_size() const {
    return parthenon::ScratchPad1D<T>::shmem_size(size());
  }
};

template <class T, std::size_t... Dims, class IndexRange>
KOKKOS_INLINE_FUNCTION
auto GetPerPointScratch(const IndexRange &idx_range) {
  if constexpr (IndexRange::index_space_t::loop_tag_v == loop_tag::boiv) {
    return StackScratch1D<T, IndexRange, Dims...>(idx_range);
  } else if constexpr (IndexRange::index_space_t::backend_v == loop_backend::raw) {
    return HostScratch1D<IndexRange, T, Dims...>(idx_range);
  } else if constexpr (IndexRange::index_space_t::backend_v == loop_backend::kokkos) {
    return TeamScratch1D<IndexRange, T, Dims...>(idx_range);
  } else {
    static_assert(always_false<IndexRange>,
                  "Unsupported loop backend for per-point scratch.");
  }
}

template <class T, std::size_t... Dims, class IndexRange>
KOKKOS_INLINE_FUNCTION std::size_t GetPerPointScratchSize(const IndexRange &idx_range) {
  using idxer_t = ctime_flat_indexer<Dims...>;
  if constexpr (IndexRange::index_space_t::loop_tag_v == loop_tag::boiv ||
                IndexRange::index_space_t::backend_v == loop_backend::raw) {
    return 0;
  } else if constexpr (IndexRange::index_space_t::backend_v == loop_backend::kokkos) {
    return parthenon::ScratchPad1D<T>::shmem_size(idx_range.ScratchSize() *
                                                 idxer_t::size);
  } else {
    static_assert(always_false<IndexRange>,
                  "Unsupported loop backend for per-point scratch size.");
  }
}

template <class T, class Halo, std::size_t... Dims, class IndexSpaceType>
inline std::size_t GetPerTeamScratchSize(const IndexSpaceType &idx_space) {
  using idxer_t = ctime_flat_indexer<Dims...>;
  if constexpr (IndexSpaceType::loop_tag_v == loop_tag::boiv ||
                IndexSpaceType::backend_v == loop_backend::raw) {
    return 0;
  } else if constexpr (IndexSpaceType::backend_v == loop_backend::kokkos) {
    const std::size_t key = reinterpret_cast<std::size_t>(&idx_space) ^
                            (std::type_index(typeid(T)).hash_code() << 1) ^
                            (std::type_index(typeid(Halo)).hash_code() << 2) ^
                            (std::type_index(typeid(idxer_t)).hash_code() << 3);
    static thread_local std::unordered_map<std::size_t, std::size_t> cache;
    if (const auto it = cache.find(key); it != cache.end()) {
      return it->second;
    }
    std::size_t scratch_size = 0;
    using BaseRangeType = InnerIndexRange<IndexSpaceType>;
    const auto &logical_kji = idx_space.GetLogicalIndexer();
    
    // Lambda for calculating the amount of scratch required for a given inner IndexRange
    auto update_scratch_size = [&](const auto &base_range) {
      const auto halo_range = base_range.template AddHalo<Halo>();
      scratch_size =
          std::max(scratch_size,
                   parthenon::ScratchPad1D<T>::shmem_size(halo_range.ScratchSize() *
                                                          idxer_t::size));
    };

    if constexpr (IndexSpaceType::loop_tag_v == loop_tag::bvoi) {
      for (int b = 0; b < idx_space.GetNBlocks(); ++b) {
        const BaseRangeType idx_range(idx_space, logical_kji, b);
        update_scratch_size(idx_range);
      }
    } else {
      const int nouter = impl::GetNOuter(idx_space);
      for (int b = 0; b < idx_space.GetNBlocks(); ++b) {
        for (int o = 0; o < nouter; ++o) {
          const int logical_start = o * idx_space.GetNInner();
          const int logical_end =
              std::min((o + 1) * idx_space.GetNInner() - 1,
                       static_cast<int>(logical_kji.size()) - 1);
          const BaseRangeType idx_range(idx_space, logical_kji, b, logical_start,
                                        logical_end);
          update_scratch_size(idx_range);
        }
      }
    }
    cache.emplace(key, scratch_size);
    return scratch_size;
  } else {
    static_assert(always_false<IndexSpaceType>,
                  "Unsupported loop backend for per-team scratch size.");
  }
}

template <class T, std::size_t... Dims, class IndexSpaceType>
inline std::size_t GetPerTeamScratchSize(const IndexSpaceType &idx_space) {
  return GetPerTeamScratchSize<T, halo::none_t, Dims...>(idx_space);
}

//----------------------------------------------------------------------------------------
//! \struct IndexedVarTypeList
//! \brief  A compile-time list of variable types that indexes per-point scratch by
//!         variable type (and, optionally, material). Each variable occupies a
//!         contiguous block of components sized by that variable's size(); StartIdx
//!         gives the offset of a given variable's block within the flat scratch.
template <class... Var_Types>
struct IndexedVarTypeList {
  using var_types = parthenon::TypeList<Var_Types...>;

  template <class VarT>
  KOKKOS_INLINE_FUNCTION static constexpr auto StartIdx() {
    return SumSizesBefore<var_types, VarT>();
  }

  template <class VarT>
  KOKKOS_INLINE_FUNCTION static constexpr auto StartIdx(VarT) {
    return SumSizesBefore<var_types, VarT>();
  }

  template <class VarT>
    requires(VarT::size() == 1)
  KOKKOS_INLINE_FUNCTION static constexpr auto StartIdx(VarT var, int mat) {
    return SumSizesBefore<var_types, VarT>() + size() * mat;
  }

  KOKKOS_INLINE_FUNCTION static constexpr auto size() {
    return SumSizesBefore<var_types>();
  }
};

//----------------------------------------------------------------------------------------
//! \class  TypeIndexedPerPointScratch
//! \brief  Wraps a flat per-point scratch buffer so it can be indexed by variable type
//!         (field_tag), component (field_tag.idx), and (optionally) a sparse/material
//!         index, using the layout defined by VarTL.
template <class Scratch, class VarTL, int NSPARSE = 1>
class TypeIndexedPerPointScratch {
 public:
  KOKKOS_INLINE_FUNCTION
  explicit TypeIndexedPerPointScratch(Scratch scratch) : scratch_(std::move(scratch)) {}

  template <class Var, class Index>
    requires(NSPARSE == 1)
  KOKKOS_INLINE_FUNCTION decltype(auto) operator()(Var &&field_tag, Index &&index) const {
    return scratch_(VarTL::StartIdx(field_tag) + field_tag.idx,
                    std::forward<Index>(index));
  }

  template <class Var, class Index>
  KOKKOS_INLINE_FUNCTION decltype(auto) operator()(Var field_tag, int sparse_idx,
                                                   Index &&index) const {
    return scratch_(VarTL::StartIdx(field_tag) + field_tag.idx +
                        VarTL::size() * sparse_idx,
                    std::forward<Index>(index));
  }

  Scratch &raw() { return scratch_; }
  const Scratch &raw() const { return scratch_; }

  KOKKOS_FORCEINLINE_FUNCTION void Zero() const { scratch_.Zero(); }

 private:
  Scratch scratch_;
};

//----------------------------------------------------------------------------------------
//! \fn     GetTypeIndexedPerPointScratch
//! \brief  Hand out a type-indexed per-point scratch buffer sized for ReconTypes
//!         (times NSPARSE materials).
template <class Real, class ReconTypes, int NSPARSE = 1, class HaloRange>
KOKKOS_INLINE_FUNCTION auto GetTypeIndexedPerPointScratch(HaloRange &&halo_range) {
  auto scratch = GetPerPointScratch<Real, ReconTypes::size() * NSPARSE>(
      std::forward<HaloRange>(halo_range));
  using Scratch = decltype(scratch);
  return TypeIndexedPerPointScratch<Scratch, ReconTypes, NSPARSE>{std::move(scratch)};
}

template <class T, class Halo, class VarTL, int NSPARSE = 1, class IdxSpace>
void AddTypeIndexedPerPointScratch(IdxSpace &idx_space, int ncopies = 1) {
  idx_space.template AddPerPointScratch<T, Halo, VarTL::size() * NSPARSE>(ncopies);
}

template <class T, class Halo, int... Shape, class IdxSpace>
void AddPerPointScratch(IdxSpace &idx_space, int ncopies = 1) {
  idx_space.template AddPerPointScratch<T, Halo, Shape...>(ncopies);
}

template <class T, class VarTL, int NSPARSE = 1, class IdxSpace>
void AddTypeIndexedPerPointScratch(IdxSpace &idx_space, int ncopies = 1) {
  idx_space.template AddPerPointScratch<T, VarTL::size() * NSPARSE>(ncopies);
}

template <class T, int... Shape, class IdxSpace>
void AddPerPointScratch(IdxSpace &idx_space, int ncopies = 1) {
  idx_space.template AddPerPointScratch<T, Shape...>(ncopies);
}

} // namespace parthenon::loop_abstraction

#endif // LOOP_ABSTRACTION_LOOP_ABSTRACTION_SCRATCH_HPP_
