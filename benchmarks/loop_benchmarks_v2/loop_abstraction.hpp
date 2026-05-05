#pragma once

#include <algorithm>
#include <concepts>
#include <optional>
#include <type_traits>
#include <utility>

#include <Kokkos_Core.hpp>

#include "kokkos_types.hpp"
#include "utils/indexer.hpp"
#include "basic_types.hpp"

namespace plb2 {

namespace loop_abstraction {

using device_team_member_t = typename Kokkos::TeamPolicy<parthenon::DevExecSpace>::member_type;

namespace impl {

template <class idx_space_t, class F>
KOKKOS_INLINE_FUNCTION
void outer_raw_for(idx_space_t idx_space, F &&f);

template <class inner_idx_range_t, class F>
KOKKOS_FORCEINLINE_FUNCTION
void inner_raw_for(const inner_idx_range_t &idx_range, F &&f);

template <class idx_space_t, class F>
void outer_kokkos(idx_space_t idx_space, F &&f);

template <class inner_idx_range_t, class F>
KOKKOS_FORCEINLINE_FUNCTION
void inner_kokkos(const inner_idx_range_t &idx_range, F &&f);

inline constexpr bool use_raw_for_v =
    std::is_same_v<parthenon::DevExecSpace, parthenon::HostExecSpace>;

} // namespace impl

template <class idx_space_t, class F>
void outer(idx_space_t idx_space, F &&f);

template <class inner_idx_range_t, class F>
KOKKOS_FORCEINLINE_FUNCTION
void inner(const inner_idx_range_t &idx_range, F &&f);

template <class IndexSpace>
class inner_index_range_t;

// Selection tags that define how a loop is partitioned:
// loop_tag chooses the outer/inner ordering, and inner_tag chooses whether the
// inner traversal is expressed in logical or memory coordinates.
enum class loop_tag {bvoi, bovi, boiv};
enum class inner_tag {logical, memory};

// Internal storage policy for the per-iteration range descriptor.
// The primary template is the span-style contract; boiv fully specializes it
// so the hot-path inner range stays as small as possible when iteration is
// point-wise instead of span-based.
template <loop_tag LOOP_TAG>
struct inner_index_range_payload_t;

// Span-style inner ranges carry flat bounds and, on device, the team member
// used to execute the inner team-parallel loop.
template <loop_tag LOOP_TAG>
struct inner_index_range_payload_t {
  int flat_start = 0;
  int flat_end = -1;
  const device_team_member_t *team_member = nullptr;
};

// Point-wise loop mode: the inner "range" is a single logical cell.
template <>
struct inner_index_range_payload_t<loop_tag::boiv> {
  int k = 0;
  int j = 0;
  int i = 0;
};

struct Index3 {
  int k, j, i;
};

template <class idx_space_t>
struct var_view_t {
 public:
  parthenon::Real* data = nullptr;
  int shift;
  idx_space_t const * pidx_space = nullptr;

  KOKKOS_FUNCTION
  parthenon::Real &operator()(int idx) const {
    return data[idx + shift];
  }

  KOKKOS_FUNCTION
  parthenon::Real &operator()(Index3 in) const {
    return data[pidx_space->memory_kji.GetFlatIdx(in.k, in.j, in.i) + shift];
  }
};

// index_space_t describes the full loop index space for one kernel:
// it owns the logical and memory coordinate systems, plus the span size used
// by bovi/bvoi-style inner loops. The compile-time tags select the strategy
// without changing the user-facing shape of the index space.
template <loop_tag LOOP_TAG, inner_tag INNER_TAG>
class index_space_t {
 public:
  template <class idx_space_t, class F>
  friend void impl::outer_raw_for(idx_space_t idx_space, F &&f);
  template <class inner_idx_range_t, class F>
  friend void impl::inner_raw_for(const inner_idx_range_t &idx_range, F &&f);
  template <class idx_space_t, class F>
  friend void impl::outer_kokkos(idx_space_t idx_space, F &&f);
  template <class inner_idx_range_t, class F>
  friend void impl::inner_kokkos(const inner_idx_range_t &idx_range, F &&f);
  template <class>
  friend struct var_view_t;
  template <class>
  friend class inner_index_range_t;

 public:
  static constexpr loop_tag loop_tag = LOOP_TAG;
  static constexpr inner_tag inner_tag = INNER_TAG;
  
  index_space_t(int nblocks, int nx, int ny, int nz, int nghost,
                std::optional<int> ninner = std::nullopt)
      : nblocks(nblocks), ninner(ninner.value_or(nx * ny)) {
    logical_kji = parthenon::Indexer3D({nghost, nghost + nz - 1},
                                       {nghost, nghost + ny - 1},
                                       {nghost, nghost + nx - 1}); 
    memory_kji = parthenon::Indexer3D({0, 2 * nghost + nz - 1},
                                      {0, 2 * nghost + ny - 1},
                                      {0, 2 * nghost + nx - 1});
  }
  
  template <class view_t>
  KOKKOS_INLINE_FUNCTION
  auto GetInnerView(view_t& in, int block, int var, std::array<int, 3> offset = {0, 0, 0}) const {
    return var_view_t<index_space_t>{&in(block, var, 0, 0, 0),
                                     static_cast<int>(memory_kji.GetFlatIdx(offset[0], offset[1], offset[2])),
                                     this};
  }

 private:
  parthenon::Indexer3D logical_kji, memory_kji;
  int nblocks;
  int ninner;
};

// inner_index_range_t is the user-facing per-iteration descriptor produced by
// outer(). It keeps the mutable iteration state private so kernel code can
// use view() and nested inner() calls without being able to corrupt the
// active loop indices directly.
template <class IndexSpace>
class inner_index_range_t {
 public:
  template <class view_t>
  KOKKOS_INLINE_FUNCTION
  auto view(view_t& in, int var, std::array<int, 3> offset = {0, 0, 0}) const {
    return pidx_space->GetInnerView(in, block, var, offset);
  }

 private:
  // Payload is specialized by loop_tag so each loop style carries only the
  // state it actually needs. This keeps the hot-path range descriptor as
  // small as possible.
  using payload_t = inner_index_range_payload_t<IndexSpace::loop_tag>;

  template <class idx_space_t, class F>
  friend void impl::outer_raw_for(idx_space_t idx_space, F &&f);
  template <class inner_idx_range_t, class F>
  friend void impl::inner_raw_for(const inner_idx_range_t &idx_range, F &&f);
  template <class idx_space_t, class F>
  friend void impl::outer_kokkos(idx_space_t idx_space, F &&f);
  template <class inner_idx_range_t, class F>
  friend void impl::inner_kokkos(const inner_idx_range_t &idx_range, F &&f);

  KOKKOS_FUNCTION
  static inner_index_range_t flat_range(const IndexSpace &idx_space, int b, int logical_start,
                                        int logical_end,
                                        const device_team_member_t *team_member = nullptr) {
    inner_index_range_t out;
    out.pidx_space = &idx_space; 
    const auto [ks, js, is] = idx_space.logical_kji(logical_start);
    out.block = b;
    const auto [ke, je, ie] = idx_space.logical_kji(logical_end);
    if constexpr (IndexSpace::inner_tag == inner_tag::memory) {
      out.payload_.flat_start = idx_space.memory_kji.GetFlatIdx(ks, js, is);
      out.payload_.flat_end = idx_space.memory_kji.GetFlatIdx(ke, je, ie);
    } else if constexpr (IndexSpace::inner_tag == inner_tag::logical) {
      out.payload_.flat_start = logical_start;
      out.payload_.flat_end = logical_end;
    }
    if constexpr (IndexSpace::loop_tag != loop_tag::boiv) {
      out.payload_.team_member = team_member;
    }
    return out;
  } 

  IndexSpace const * pidx_space = nullptr;
  int block = 0;
  payload_t payload_{};
};

namespace impl {

template <class idx_space_t, class F>
KOKKOS_INLINE_FUNCTION
void outer_raw_for(idx_space_t idx_space, F&& f) {
  using inner_idx_range_t = inner_index_range_t<idx_space_t>;
  // outer() is the user-visible loop entry point. It materializes the current
  // outer iteration into an inner range descriptor and hands it to the kernel body.
  if constexpr (idx_space.loop_tag == loop_tag::bvoi) {
    for (int b = 0; b < idx_space.nblocks; ++b) { 
      inner_idx_range_t idx_range;
      idx_range.pidx_space = &idx_space;
      idx_range.block = b;
      f(idx_range, b);
    }   
  } else if constexpr (idx_space.loop_tag == loop_tag::bovi) {
    const int nouter = idx_space.logical_kji.size() / idx_space.ninner
                     + (idx_space.logical_kji.size() % idx_space.ninner != 0);
    for (int b = 0; b < idx_space.nblocks; ++b) { 
      for (int o = 0; o < nouter; ++o) {
        const int logical_start = o * idx_space.ninner; 
        const int logical_end = std::min((o + 1) * idx_space.ninner - 1, static_cast<int>(idx_space.logical_kji.size()) - 1);
        const auto idx_range = inner_idx_range_t::flat_range(idx_space, b, logical_start, logical_end);
        f(idx_range, b);
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
    idx_range.pidx_space = &idx_space;
    for (idx_range.block = 0; idx_range.block < idx_space.nblocks; ++idx_range.block) {
      for (int k = ks; k <= ke; ++k) {
        for (int j = js; j <= je; ++j) {
#pragma omp simd
          for (int i = is; i <= ie; ++i) {
            idx_range.payload_.k = k;
            idx_range.payload_.j = j;
            idx_range.payload_.i = i;
            f(idx_range, idx_range.block);
          }
        }
      }  
    }  
  }
}

template <class idx_space_t, class F>
void outer_kokkos(idx_space_t idx_space, F&& f) {
  using inner_idx_range_t = inner_index_range_t<idx_space_t>;
  if constexpr (idx_space.loop_tag == loop_tag::boiv) {
    static_assert(idx_space.inner_tag == inner_tag::logical,
                  "boiv currently expects logical inner coordinates");
    const int cells_per_block = static_cast<int>(idx_space.logical_kji.size());
    const int total = idx_space.nblocks * cells_per_block;
    Kokkos::parallel_for(
        "loop_abstraction::outer_kokkos_boiv",
        Kokkos::RangePolicy<parthenon::DevExecSpace>(0, total),
        KOKKOS_LAMBDA(const int flat) {
          const int b = flat / cells_per_block;
          const int local = flat % cells_per_block;
          const auto [k, j, i] = idx_space.logical_kji(local);
          inner_idx_range_t idx_range;
          idx_range.pidx_space = &idx_space;
          idx_range.block = b;
          idx_range.payload_.k = k;
          idx_range.payload_.j = j;
          idx_range.payload_.i = i;
          f(idx_range, b);
        });
  } else if constexpr (idx_space.loop_tag == loop_tag::bovi) {
    const int nouter = idx_space.logical_kji.size() / idx_space.ninner
                     + (idx_space.logical_kji.size() % idx_space.ninner != 0);
    const int league_size = idx_space.nblocks * nouter;
    const Kokkos::TeamPolicy<parthenon::DevExecSpace> policy(league_size, Kokkos::AUTO);
    Kokkos::parallel_for(
        "loop_abstraction::outer_kokkos_team", policy,
        KOKKOS_LAMBDA(const device_team_member_t &member) {
          const int league = member.league_rank();
          const int b = league / nouter;
          const int o = league % nouter;
          const int logical_start = o * idx_space.ninner;
          const int logical_end =
              std::min((o + 1) * idx_space.ninner - 1,
                       static_cast<int>(idx_space.logical_kji.size()) - 1);
          const auto idx_range = inner_idx_range_t::flat_range(
              idx_space, b, logical_start, logical_end, &member);
          f(idx_range, b);
        });
  } else if constexpr (idx_space.loop_tag == loop_tag::bvoi) {
    const Kokkos::TeamPolicy<parthenon::DevExecSpace> policy(idx_space.nblocks, Kokkos::AUTO);
    Kokkos::parallel_for(
        "loop_abstraction::outer_kokkos_bvoi", policy,
        KOKKOS_LAMBDA(const device_team_member_t &member) {
          const int b = member.league_rank();
          inner_idx_range_t idx_range;
          idx_range.pidx_space = &idx_space;
          idx_range.block = b;
          idx_range.payload_.team_member = &member;
          f(idx_range, b);
        });
  }
}

template <class inner_idx_range_t, class F>
KOKKOS_FORCEINLINE_FUNCTION
void inner_raw_for(const inner_idx_range_t &idx_range, F &&f) {
  using idx_space_t = std::remove_cv_t<std::remove_reference_t<decltype(*idx_range.pidx_space)>>;
  const auto &idx_space = *(idx_range.pidx_space);
  // inner() is the portable execution contract: the same kernel body runs over
  // different backend-specific loop strategies without changing the call site.
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
            f(idx_space.memory_kji.GetFlatIdx(k, j, i));
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
        for (int idx = inner_range.payload_.flat_start; idx <= inner_range.payload_.flat_end; ++idx) {
          f(idx);
        }
      }
    } 
  } else if constexpr (idx_space_t::loop_tag == loop_tag::bovi) {
    const int start = idx_range.payload_.flat_start;
    const int end_exclusive = idx_range.payload_.flat_end + 1;
#pragma omp simd
    for (int idx = start; idx < end_exclusive; ++idx) {
      if constexpr(idx_space_t::inner_tag == inner_tag::memory) {
        f(idx);
      } else { 
        const auto [k, j, i] = idx_space.logical_kji(idx);
        f(Index3{k, j, i});
      }
    }
  } else if constexpr (idx_space_t::loop_tag == loop_tag::boiv) {
    f(Index3{idx_range.payload_.k, idx_range.payload_.j, idx_range.payload_.i});
  }
}

template <class inner_idx_range_t, class F>
KOKKOS_FORCEINLINE_FUNCTION
void inner_kokkos(const inner_idx_range_t &idx_range, F &&f) {
  using idx_space_t = std::remove_cv_t<std::remove_reference_t<decltype(*idx_range.pidx_space)>>;
  const auto &idx_space = *(idx_range.pidx_space);
  if constexpr (idx_space_t::loop_tag == loop_tag::boiv) {
    f(Index3{idx_range.payload_.k, idx_range.payload_.j, idx_range.payload_.i});
  } else if constexpr (idx_space_t::loop_tag == loop_tag::bovi) {
    const auto *team_member = idx_range.payload_.team_member;
    KOKKOS_ASSERT(team_member != nullptr);
    const auto &member = *team_member;
    const int start = idx_range.payload_.flat_start;
    const int end_exclusive = idx_range.payload_.flat_end + 1;
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(member, start, end_exclusive),
        KOKKOS_LAMBDA(const int idx) {
          if constexpr (idx_space_t::inner_tag == inner_tag::memory) {
            f(idx);
          } else {
            const auto [k, j, i] = idx_space.logical_kji(idx);
            f(Index3{k, j, i});
          }
        });
  } else if constexpr (idx_space_t::loop_tag == loop_tag::bvoi) {
    const auto *team_member = idx_range.payload_.team_member;
    KOKKOS_ASSERT(team_member != nullptr);
    const auto &member = *team_member;
    const int nouter = idx_space.logical_kji.size() / idx_space.ninner
                     + (idx_space.logical_kji.size() % idx_space.ninner != 0);
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(member, 0, nouter), KOKKOS_LAMBDA(const int o) {
          const int logical_start = o * idx_space.ninner;
          const int logical_end =
              std::min((o + 1) * idx_space.ninner - 1,
                       static_cast<int>(idx_space.logical_kji.size()) - 1);
          if constexpr (idx_space_t::inner_tag == inner_tag::memory) {
            const auto [ks, js, is] = idx_space.logical_kji(logical_start);
            const auto [ke, je, ie] = idx_space.logical_kji(logical_end);
            const int flat_start = idx_space.memory_kji.GetFlatIdx(ks, js, is);
            const int flat_end = idx_space.memory_kji.GetFlatIdx(ke, je, ie);
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(member, flat_start, flat_end + 1),
                KOKKOS_LAMBDA(const int idx) { f(idx); });
          } else {
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(member, logical_start, logical_end + 1),
                KOKKOS_LAMBDA(const int idx) {
                  const auto [k, j, i] = idx_space.logical_kji(idx);
                  f(idx_space.memory_kji.GetFlatIdx(k, j, i));
                });
          }
        });
  }
}

} // namespace impl

template <class idx_space_t, class F>
void outer(idx_space_t idx_space, F&& f) {
  if constexpr (impl::use_raw_for_v) {
    impl::outer_raw_for(idx_space, std::forward<F>(f));
  } else {
    impl::outer_kokkos(idx_space, std::forward<F>(f));
  }
}

template <class inner_idx_range_t, class F>
KOKKOS_FORCEINLINE_FUNCTION
void inner(const inner_idx_range_t &idx_range, F &&f) {
  if constexpr (impl::use_raw_for_v) {
    impl::inner_raw_for(idx_range, std::forward<F>(f));
  } else {
    impl::inner_kokkos(idx_range, std::forward<F>(f));
  }
}
  
} // namespace loop_abstraction

}  // namespace plb2
