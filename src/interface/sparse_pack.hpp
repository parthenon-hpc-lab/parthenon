//========================================================================================
// (C) (or copyright) 2020-2022. Triad National Security, LLC. All rights reserved.
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
#ifndef INTERFACE_SPARSE_PACK_HPP_
#define INTERFACE_SPARSE_PACK_HPP_

#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <regex>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "bvals/cc/bvals_cc_in_one.hpp"
#include "interface/mesh_data.hpp"
#include "interface/variable_pack.hpp"
#include "mesh/domain.hpp"
#include "mesh/meshblock.hpp"
#include "mesh/meshblock_pack.hpp"
#include "utils/error_checking.hpp"
#include "utils/utils.hpp"

namespace parthenon {
namespace impl {
template <int idx, class TIn, class T0, class... Tl>
struct GetIdxImpl {
  KOKKOS_FORCEINLINE_FUNCTION
  static int val() {
    if (std::is_same<TIn, T0>::value) return idx;
    return GetIdxImpl<idx + 1, TIn, Tl...>::val();
  }
};

template <int idx, class TIn, class T0>
struct GetIdxImpl<idx, TIn, T0> {
  KOKKOS_FORCEINLINE_FUNCTION
  static int val() {
    if (std::is_same<TIn, T0>::value) return idx;
    return -1;
  }
};
} // namespace impl

template <int... IN>
struct multiply;

template <int I0, int... IN>
struct multiply<I0, IN...> {
  constexpr static int val() { return I0 * multiply<IN...>::val(); }
};

template <>
struct multiply<> {
  constexpr static int val() { return 1; }
};

template <class TIn, class... Tl>
KOKKOS_FORCEINLINE_FUNCTION int GetTypeIdx() {
  return impl::GetIdxImpl<0, TIn, Tl...>::val();
}

// Struct that all variables types should inherit from
template <bool REGEX, int... NCOMP>
struct variable_t {
  KOKKOS_FUNCTION
  variable_t() : idx(0) { assert(0 == ndim); }
  KOKKOS_FUNCTION
  explicit variable_t(int idx1) : idx(idx1) { assert(1 == ndim); }

  virtual ~variable_t() = default;

  static std::string name() {
    PARTHENON_FAIL("Need to implement your own name method.");
    return "error";
  }
  static bool regex() { return REGEX; }
  static int ndim() { return sizeof...(NCOMP); }
  static int size() { return multiply<NCOMP...>::val(); }
  const int idx;
};

// Pack only allocated variables at the meshdata level
template <class... Ts>
class SparsePack {
 public:
  explicit SparsePack(MeshData<Real> *pmd) {
    const std::vector<std::string> names{Ts::name()...};
    const std::vector<std::regex> regexes{std::regex(Ts::name())...};
    const std::vector<bool> use_regex{Ts::regex()...};

    auto include_variable = [&](int vidx, const std::shared_ptr<CellVariable<Real>> &pv) {
      if (!pv->IsAllocated()) return false;
      // TODO(LFR): Check that the shapes agree
      if (use_regex[vidx]) {
        if (std::regex_match(std::string(pv->label()), regexes[vidx])) return true;
      } else {
        if (names[vidx] == pv->label()) return true;
      }
      return false;
    };

    // Count up the size of the array that is required
    int max_size = 0;
    for (int b = 0; b < pmd->NumBlocks(); ++b) {
      auto &pmb = pmd->GetBlockData(b);
      int size = 0;
      for (auto &pv : pmb->GetCellVariableVector()) {
        for (int i = 0; i < names.size(); ++i) {
          if (include_variable(i, pv)) {
            size += pv->GetDim(6) * pv->GetDim(5) * pv->GetDim(4);
          }
        }
      }
      max_size = std::max(size, max_size);
    }

    pack_ = ParArray2D<ParArray3D<Real>>("data_ptr", pmd->NumBlocks(), max_size);
    auto pack_h = Kokkos::create_mirror_view(pack_);

    for (int b = 0; b < pmd->NumBlocks(); ++b) {
      auto &pmb = pmd->GetBlockData(b);
      int idx = 0;
      for (int i = 0; i < names.size(); ++i) {
        hlo_[b][i] = idx;
        for (auto &pv : pmb->GetCellVariableVector()) {
          if (include_variable(i, pv)) {
            for (int t = 0; t < pv->GetDim(6); ++t) {
              for (int u = 0; u < pv->GetDim(5); ++u) {
                for (int v = 0; v < pv->GetDim(4); ++v) {
                  pack_h(b, idx) = pv->data.Get(t, u, v);
                  idx++;
                }
              }
            }
          }
        }

        hhi_[b][i] = idx - 1;

        if (hhi_[b][i] < hlo_[b][i]) {
          // Did not find any allocated variables meeting our criteria
          hlo_[b][i] = -1;
          hhi_[b][i] =
              -2; // Make the upper bound more negative so a for loop won't iterate once
        }
      }
    }

    Kokkos::deep_copy(pack_, pack_h);
  }

  // auto GetDim(int i) {
  //  return pack_.GetDim(i);
  //}

  template <class TIn>
  KOKKOS_INLINE_FUNCTION int GetLowerBound(const TIn &, const int b) const {
    return hlo_[b][GetTypeIdx<TIn, Ts...>()];
  }

  template <class TIn>
  KOKKOS_INLINE_FUNCTION int GetUpperBound(const TIn &, const int b) const {
    return hhi_[b][GetTypeIdx<TIn, Ts...>()];
  }

  KOKKOS_INLINE_FUNCTION
  Real &operator()(const int b, const int idx, const int k, const int j,
                   const int i) const {
    return pack_(b, idx)(k, j, i);
  }

  template <class TIn>
  KOKKOS_INLINE_FUNCTION Real &operator()(const int b, const TIn &t, const int k,
                                          const int j, const int i) const {
    return pack_(b, hlo_[b][GetTypeIdx<TIn, Ts...>()] + t.idx)(k, j, i);
  }

 private:
  int hlo_[20][sizeof...(Ts)];
  int hhi_[20][sizeof...(Ts)];

  ParArray2D<ParArray3D<Real>> pack_;
};

} // namespace parthenon

#endif // INTERFACE_SPARSE_PACK_HPP_
