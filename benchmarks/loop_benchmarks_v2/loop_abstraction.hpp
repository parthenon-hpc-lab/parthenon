#pragma once

#include <algorithm>
#include <concepts>

#include <Kokkos_Core.hpp>

#include "utils/indexer.hpp"
#include "basic_types.hpp"

namespace plb2 {

namespace loop_abstraction {

template<class...>
inline constexpr bool always_false_v = false;

enum class loop_tag {bvoi, bovi, boiv};
enum class inner_tag {logical, memory};

struct Index3 {
  int k, j, i;
};

template <class idx_space_t>
struct var_view_t {
  parthenon::Real* data = nullptr;
  int offset;
  idx_space_t const * pidx_space = nullptr;

  KOKKOS_FUNCTION
  parthenon::Real &operator()(int idx) const {
    return data[idx - offset];
  }

  KOKKOS_FUNCTION
  parthenon::Real &operator()(Index3 in) const {
    return data[pidx_space->memory_kji.GetFlatIdx(in.k, in.j, in.i) - offset];
  }
};

template <loop_tag LOOP_TAG, inner_tag INNER_TAG>
struct index_space_t {
  static constexpr loop_tag loop_tag = LOOP_TAG;
  static constexpr inner_tag inner_tag = INNER_TAG;
  
  index_space_t(int nblocks, int nx, int ny, int nz, int nghost) : nblocks(nblocks) {
    logical_kji = parthenon::Indexer3D({nghost, nghost + nz - 1},
                                       {nghost, nghost + ny - 1},
                                       {nghost, nghost + nx - 1}); 
    memory_kji = parthenon::Indexer3D({0, 2 * nghost + nz - 1},
                                      {0, 2 * nghost + ny - 1},
                                      {0, 2 * nghost + nx - 1});
    ninner = nx * ny; 
  }

  parthenon::Indexer3D logical_kji, memory_kji;
  int nblocks;
  int ninner;

  template <class view_t>
  KOKKOS_INLINE_FUNCTION
  auto GetInnerView(view_t& in, int block, int var) const { 
    return var_view_t<index_space_t>{&in(block, var, 0, 0, 0), 0, this};
  }
};


template <class IndexSpace>
struct inner_index_range_t {
  using idx_space_t = IndexSpace;

  KOKKOS_FUNCTION
  static inner_index_range_t flat_range(const IndexSpace &idx_space, int b, int logical_start, int logical_end) {
    inner_index_range_t out;
    out.pidx_space = &idx_space; 
    const auto [ks, js, is] = idx_space.logical_kji(logical_start);
    out.block = b;
    out.ks = ks;
    out.js = js;
    out.is = is;
    const auto [ke, je, ie] = idx_space.logical_kji(logical_end);
    if constexpr (idx_space_t::inner_tag == inner_tag::memory) {
      out.flat_start = idx_space.memory_kji.GetFlatIdx(ks, js, is);
      out.flat_end = idx_space.memory_kji.GetFlatIdx(ke, je, ie);
    } else if constexpr (idx_space_t::inner_tag == inner_tag::logical) {
      out.flat_start = logical_start;
      out.flat_end = logical_end;
    } 
    return out;
  } 
  
  KOKKOS_INLINE_FUNCTION
  auto GetSpatialIndices(Index3 in) const {
    return std::make_tuple(in.k, in.j, in.i);
  }

  KOKKOS_INLINE_FUNCTION
  auto GetSpatialIndices(int flat) const {
    if constexpr (IndexSpace::inner_tag == inner_tag::memory) {
      return pidx_space->memory_kji(flat);
    } else if constexpr (IndexSpace::inner_tag == inner_tag::logical) {
      return pidx_space->logical_kji(flat);
    } else {
      static_assert(always_false_v<IndexSpace>, "Unsupported inner_tag");
    }
  }

  IndexSpace const * pidx_space = nullptr;
  int flat_start, flat_end;
  int block;
  int ks, js, is;
};



template <class idx_space_t, class F> 
void outer(idx_space_t idx_space, F&& f) {
  using inner_idx_range_t = inner_index_range_t<idx_space_t>;
  if constexpr (idx_space.loop_tag == loop_tag::bvoi) {
    for (int b = 0; b < idx_space.nblocks; ++b) { 
      inner_idx_range_t idx_range;
      idx_range.pidx_space = &idx_space;
      idx_range.block = b;
      f(idx_space, idx_range, b);
    }   
  } else if constexpr (idx_space.loop_tag == loop_tag::bovi) {
    const int nouter = idx_space.logical_kji.size() / idx_space.ninner
                     + (idx_space.logical_kji.size() % idx_space.ninner != 0);
    for (int b = 0; b < idx_space.nblocks; ++b) { 
      for (int o = 0; o < nouter; ++o) {
        const int logical_start = o * idx_space.ninner; 
        const int logical_end = std::min((o + 1) * idx_space.ninner - 1, static_cast<int>(idx_space.logical_kji.size()) - 1);
        const auto idx_range = inner_idx_range_t::flat_range(idx_space, b, logical_start, logical_end);
        f(idx_space, idx_range, b);
      }
    }    
  } else if constexpr (idx_space.loop_tag == loop_tag::boiv) {
    static_assert(idx_space.inner_tag == inner_tag::logical, "Probably don't want to do boiv over interior memory"); 
    const int ks = idx_space.logical_kji.template StartIdx<0>();
    const int ke = idx_space.logical_kji.template EndIdx<0>();
    const int js = idx_space.logical_kji.template StartIdx<1>();
    const int je = idx_space.logical_kji.template EndIdx<1>();
    const int is = idx_space.logical_kji.template StartIdx<2>();
    const int ie = idx_space.logical_kji.template EndIdx<2>();
    inner_idx_range_t idx_range;
    for (idx_range.block = 0; idx_range.block < idx_space.nblocks; ++idx_range.block) {
      for (idx_range.ks = ks; idx_range.ks <= ke; ++idx_range.ks) {
        for (idx_range.js = js; idx_range.js <= je; ++idx_range.js) {
#pragma omp simd
          for (idx_range.is = is; idx_range.is <= ie; ++idx_range.is) {
            f(idx_space, idx_range, idx_range.block);
          }
        }
      }
    }  
  }
}

template <class idx_space_t, class inner_idx_range_t, class F> 
KOKKOS_INLINE_FUNCTION
void inner(const idx_space_t &idx_space, const inner_idx_range_t &idx_range, F &&f) {
  if constexpr (idx_space_t::loop_tag == loop_tag::bvoi) {
    if constexpr (idx_space_t::inner_tag == inner_tag::logical) { 
      const int ks = idx_space.logical_kji.template StartIdx<0>();
      const int ke = idx_space.logical_kji.template EndIdx<0>();
      const int js = idx_space.logical_kji.template StartIdx<1>();
      const int je = idx_space.logical_kji.template EndIdx<1>();
      const int is = idx_space.logical_kji.template StartIdx<2>();
      const int ie = idx_space.logical_kji.template EndIdx<2>();
      for (int k = ks; k <= ke; ++k) {
        for (int j = js; j <= je; ++j) {
#pragma omp simd
          for (int i = is; i <= ie; ++i) {
            f(Index3{k, j, i});
          }
        }
      }  
    } else if constexpr (idx_space_t::inner_tag == inner_tag::memory) {
      const int nouter = idx_space.logical_kji.size() / idx_space.ninner
                     + (idx_space.logical_kji.size() % idx_space.ninner != 0);
      for (int o = 0; o < nouter; ++o) {
        const int logical_start = o * idx_space.ninner; 
        const int logical_end = std::min((o + 1) * idx_space.ninner - 1, static_cast<int>(idx_space.logical_kji.size()) - 1);
        const auto inner_range = inner_idx_range_t::flat_range(idx_space, idx_range.block, logical_start, logical_end);
#pragma omp simd  
        for (int idx = inner_range.flat_start; idx <= inner_range.flat_end; ++idx) {
          f(idx);
        }
      }
    } 
  } else if constexpr (idx_space_t::loop_tag == loop_tag::bovi) {
#pragma omp simd  
    for (int idx = idx_range.flat_start; idx <= idx_range.flat_end; ++idx) {
      if constexpr(idx_space_t::inner_tag == inner_tag::memory) {
        f(idx);
      } else { 
        const auto [k, j, i] = idx_space.logical_kji(idx);
        f(Index3{k, j, i});
      }
    }
  } else if constexpr (idx_space_t::loop_tag == loop_tag::boiv) {
    f(Index3{idx_range.ks, idx_range.js, idx_range.is});
  }

}

  
} // namespace loop_abstraction

}  // namespace plb2
