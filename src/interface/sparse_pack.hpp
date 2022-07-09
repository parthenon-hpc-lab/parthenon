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
template <int... IN>
struct multiply;

template <>
struct multiply<> : std::integral_constant<std::size_t, 1> {};

template <int I0, int... IN>
struct multiply<I0, IN...> : std::integral_constant<int, I0 * multiply<IN...>::value> {};

// GetTypeIdx is taken from Stack Overflow 26169198, should cause compile time failure if type is not in list 
template <typename T, typename... Ts>
struct GetTypeIdx;

template <typename T, typename... Ts>
struct GetTypeIdx<T, T, Ts...> : std::integral_constant<std::size_t, 0> {};

template <typename T, typename U, typename... Ts>
struct GetTypeIdx<T, U, Ts...> : std::integral_constant<std::size_t, 1 + GetTypeIdx<T, Ts...>::value> {};

template<class T, class F> 
void IterateBlocks(T*, F func); 

template <class F> 
void IterateBlocks(MeshData<Real> * pmd, F func) {
  for (int b = 0; b < pmd->NumBlocks(); ++b) {
    auto &pmb = pmd->GetBlockData(b);
    func(b, pmb.get());
  } 
}

template <class F> 
void IterateBlocks(MeshBlockData<Real> * pmb, F func) {
  func(0, pmb);
}

} // namepace impl 

using namespace impl;

// Struct that all variables types should inherit from
template <bool REGEX, int... NCOMP>
struct variable_t {
  KOKKOS_FUNCTION
  variable_t() : idx(0) {}
  
  KOKKOS_FUNCTION
  explicit variable_t(int idx1) : idx(idx1) {}

  virtual ~variable_t() = default;
  
  // All of these are just static methods so that there is no 
  // extra storage in the struct
  static std::string name() {
    PARTHENON_FAIL("Need to implement your own name method.");
    return "error";
  }
  KOKKOS_INLINE_FUNCTION
  static bool regex() { return REGEX; }
  KOKKOS_INLINE_FUNCTION
  static int ndim() { return sizeof...(NCOMP); }
  KOKKOS_INLINE_FUNCTION
  static int size() { return multiply<NCOMP...>::value; }
  
  const int idx;
};

class SparsePackBase { 
 protected:

  friend class SparsePackCache;

  using pack_t = ParArray2D<ParArray3D<Real>>;
  using bounds_t = ParArray3D<int>;
  using alloc_t = ParArray1D<bool>::host_mirror_type; 
  
  pack_t pack_;
  bounds_t bounds_;
  alloc_t alloc_status_h_;
  
  template <class T, class F>
  static alloc_t GetAllocStatus(T *pmd, int nvar, const F& include_variable);
  
  template <class T, class F>
  static SparsePackBase Build(T *pmd, int nvar, const F& include_variable); 
  
 public: 
  SparsePackBase() = default;
  SparsePackBase(const pack_t& pack, const bounds_t& bounds) : pack_(pack), bounds_(bounds) {} 
  virtual ~SparsePackBase() = default; 

}; 

class SparsePackCache { 
 public: 
  template<class T> 
  bool Add(T* ppack) { 
    std::string ident = ppack->GetIdentifier(); 
    bool preexists = pack_map.count(ident); 
    pack_map[ident] = static_cast<SparsePackBase>(*ppack);
    return preexists;
  }

  template<class T> 
  bool TryLoad(T* ppack) {
    std::string ident = ppack->GetIdentifier();
    if (pack_map.count(ident) > 0) {
      if (ppack->alloc_status_h_.size() != pack_map[ident].alloc_status_h_.size()) {
        printf("Did not find a pre-existing cache that matches (%s).\n", ident.c_str()); 
        return false;
      }
      for (int i=0; i<ppack->alloc_status_h_.size(); ++i) {
        if (ppack->alloc_status_h_(i) != pack_map[ident].alloc_status_h_(i)) {
          printf("Did not find a pre-existing cache that matches (%s).\n", ident.c_str()); 
          return false;
        }
      }

      *ppack = T(pack_map[ident]);
      printf("Found a pre-existing cache that matches (%s).\n", ident.c_str());
      return true;
    }
    printf("Did not find a pre-existing cache that matches (%s).\n", ident.c_str()); 
    return false;
  }

 protected: 
  std::unordered_map<std::string, SparsePackBase> pack_map; 
};

// Pack only allocated variables at the meshdata level
template <class... Ts>
class SparsePack : public SparsePackBase {
 public:
  template <class T>
  explicit SparsePack(T *pmd) 
    : SparsePackBase(Build(pmd, sizeof...(Ts), GetTestFunction())) {}

  template <class T> 
  SparsePack(T *pmd, SparsePackCache* pcache) {
    auto include_variable = GetTestFunction();
    alloc_status_h_ = GetAllocStatus(pmd, sizeof...(Ts), include_variable); 
    if (!pcache->TryLoad(this)) {
      // Ok since there should be no data members of SparsePack itself
      static_cast<SparsePackBase&>(*this) = Build(pmd, sizeof...(Ts), include_variable);
      pcache->Add(this);
    }
  }

  explicit SparsePack(const SparsePackBase& spb) : SparsePackBase(spb) {}

  std::string GetIdentifier() const {
    const std::vector<std::string> idents{(Ts::name() + std::to_string(Ts::regex()))...}; 
    std::string identifier(""); 
    for (const auto &str : idents) identifier += str;
    return identifier;
  }

  template <class TIn>
  KOKKOS_INLINE_FUNCTION int GetLowerBound(const TIn &, const int b) const {
    const int vidx = GetTypeIdx<TIn, Ts...>::value;
    return bounds_(0, b, vidx);
  }

  template <class TIn>
  KOKKOS_INLINE_FUNCTION int GetUpperBound(const TIn &, const int b) const {
    const int vidx = GetTypeIdx<TIn, Ts...>::value;
    return bounds_(1, b, vidx);
  }
  
  KOKKOS_INLINE_FUNCTION
  Real &operator()(const int b, const int idx, const int k, const int j,
                   const int i) const {
    return pack_(b, idx)(k, j, i);
  }

  template <class TIn, class = typename std::enable_if<!std::is_integral<TIn>::value>::type>
  KOKKOS_INLINE_FUNCTION Real &operator()(const int b, const TIn &t, const int k,
                                          const int j, const int i) const {
    const int vidx = GetLowerBound(t, b) + t.idx;
    return pack_(b, vidx)(k, j, i);
  }
 protected:

  static auto GetTestFunction() { 
    const std::vector<std::string> names{Ts::name()...};
    const std::vector<std::regex> regexes{std::regex(Ts::name())...};
    const std::vector<bool> use_regex{Ts::regex()...};

    return [=](int vidx, const std::shared_ptr<CellVariable<Real>> &pv) {
      // TODO(LFR): Check that the shapes agree
      if (use_regex[vidx]) {
        if (std::regex_match(std::string(pv->label()), regexes[vidx])) return true;
      } else {
        if (names[vidx] == pv->label()) return true;
      }
      return false;
    };
  }
};

// Declarations below 
template <class T, class F> 
SparsePackBase::alloc_t SparsePackBase::GetAllocStatus(T *pmd, int nvar, const F &include_variable) {

  std::vector<bool> astat; 
  IterateBlocks(pmd, [&](int b, MeshBlockData<Real> *pmb) {
    for (int i = 0; i < nvar; ++i) {
      for (auto &pv : pmb->GetCellVariableVector()) {
        if (include_variable(i, pv)) {
          astat.push_back(pv->IsAllocated());
        }
      }
    }
  });

  alloc_t alloc_status_h("alloc", astat.size()); 
  for (int i=0; i<astat.size(); ++i) alloc_status_h(i) = astat[i];  
  return alloc_status_h;
} 

template <class T, class F> 
SparsePackBase SparsePackBase::Build(T *pmd, int nvar, const F &include_variable) {
  
  SparsePackBase pack; 
  pack.alloc_status_h_ = GetAllocStatus(pmd, nvar, include_variable);

  // Count up the size of the array that is required
  int max_size = 0;
  int nblocks = 0;
  IterateBlocks(pmd, [&](int b, MeshBlockData<Real> *pmb) {
    int size = 0;
    nblocks++;
    for (auto &pv : pmb->GetCellVariableVector()) {
      for (int i = 0; i < nvar; ++i) {
        if (include_variable(i, pv)) {
          if (pv->IsAllocated()) size += pv->GetDim(6) * pv->GetDim(5) * pv->GetDim(4);
        }
      }
    }
    max_size = std::max(size, max_size);
  });

  pack.pack_ = pack_t("data_ptr", nblocks, max_size);
  auto pack_h = Kokkos::create_mirror_view(pack.pack_);
  
  pack.bounds_ = bounds_t("bounds", 2, nblocks, nvar); 
  auto bounds_h = Kokkos::create_mirror_view(pack.bounds_);
   
  IterateBlocks(pmd, [&](int b, MeshBlockData<Real> *pmb) {
    int idx = 0;
    for (int i = 0; i < nvar; ++i) {
      bounds_h(0, b, i) = idx;
      for (auto &pv : pmb->GetCellVariableVector()) {
        if (include_variable(i, pv)) {
          if (pv->IsAllocated()) {
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
      }

      bounds_h(1, b, i) = idx - 1;

      if (bounds_h(1, b, i) < bounds_h(0, b, i)) {
        // Did not find any allocated variables meeting our criteria
        bounds_h(0, b, i) = -1;
        // Make the upper bound more negative so a for loop won't iterate once
        bounds_h(1, b, i) = -2;
      }
    }
  });

  Kokkos::deep_copy(pack.pack_, pack_h);
  Kokkos::deep_copy(pack.bounds_, bounds_h);
  
  return pack;
}

} // namespace parthenon

#endif // INTERFACE_SPARSE_PACK_HPP_
