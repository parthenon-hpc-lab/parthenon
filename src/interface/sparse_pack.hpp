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

#include "coordinates/coordinates.hpp"
#include "interface/variable.hpp"
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

// SFINAE for block iteration so that sparse packs can work for MeshBlockData and MeshData 
template <class T, class F> 
inline auto IterateBlocks(T *pmd, F func) -> decltype(T().GetBlockData(0), void()) {
  for (int b = 0; b < pmd->NumBlocks(); ++b) {
    auto &pmbd = pmd->GetBlockData(b);
    func(b, pmbd.get());
  } 
}

template <class T, class F> 
inline auto IterateBlocks(T *pmbd, F func) -> decltype(T().GetBlockPointer(), void()) {
  func(0, pmbd);
}

// This is some awful SFINAE so that we can get the MeshBlockData type using 
// decltype around this function without including meshblock_data.hpp
template <class T> 
auto MeshBlockDataTypeHack() -> typename std::remove_reference_t<decltype(T().GetBlockData(0))>::element_type {
  return decltype(std::declval<T>().GetBlockData(0))();  
}

template <class T> 
auto MeshBlockDataTypeHack() -> decltype(T().GetBlockPointer(), T()) {
  return T();  
}


} // namepace impl 

using namespace impl;

namespace variables { 
// Struct that all variables types should inherit from
template <bool REGEX, int... NCOMP>
struct base_t {
  KOKKOS_FUNCTION
  base_t() : idx(0) {}
  
  KOKKOS_FUNCTION
  explicit base_t(int idx1) : idx(idx1) {}

  virtual ~base_t() = default;
  
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

struct any : public base_t<true> {
  static std::string name() { return ".*"; }
};
}

template <class T>
class MeshBlockData;

struct PackDescriptor { 
  std::vector<std::string> vars; 
  std::vector<bool> use_regex; 
  std::vector<MetadataFlag> flags; 
  bool with_fluxes; 
}; 

class SparsePackBase { 
 protected:

  friend class SparsePackCache;

  using pack_t = ParArray3D<ParArray3D<Real>>;
  using bounds_t = ParArray3D<int>;
  using alloc_t = ParArray1D<bool>::host_mirror_type; 
  using coords_t = ParArray1D<ParArray0D<Coordinates_t>>;

  pack_t pack_;
  bounds_t bounds_;
  coords_t coords_;

  ParArray1D<std::string>::host_mirror_type names_h_;
  ParArray1D<bool>::host_mirror_type use_regex_h_; 
  alloc_t alloc_status_h_; 
  
  bool with_fluxes_;
  int nblocks_; 
  int ndim_;

  template <class T>
  static alloc_t GetAllocStatus(T *pmd, const PackDescriptor &desc);
  
  template <class T>
  static SparsePackBase Build(T *pmd, const PackDescriptor &desc); 

  static auto GetTestFunction(const PackDescriptor &desc) { 
    std::vector<std::regex> regexes; 
    for (const auto& var : desc.vars) regexes.push_back(std::regex(var)); 

    // Lambda for testing wether or not we want to include cellvariable pv 
    // in the index range of the type variable with index vidx 
    return [=](int vidx, const std::shared_ptr<CellVariable<Real>> &pv) {
      // TODO(LFR): Check that the shapes agree
      
      if (desc.flags.size() > 0) {
        for (const auto& flag : desc.flags) {
          if (!pv->IsSet(flag)) {
            return false;
          }
        }
      }
      
      if (desc.use_regex[vidx]) {
        if (std::regex_match(std::string(pv->label()), regexes[vidx])) return true;
      } else {
        if (desc.vars[vidx] == pv->label()) return true;
      }
      return false;
    };
  }

 public: 
  SparsePackBase() = default;
  virtual ~SparsePackBase() = default; 
  
  int GetNBlocks() const {return nblocks_;}
  int GetNDim() const {return ndim_;}

  KOKKOS_INLINE_FUNCTION
  const Coordinates_t& GetCoordinates(const int b) const { return coords_(b)(); }
  
  KOKKOS_INLINE_FUNCTION
  Real &operator()(const int b, const int idx, const int k, const int j,
                   const int i) const {
    return pack_(0, b, idx)(k, j, i);
  }

  KOKKOS_INLINE_FUNCTION
  Real &flux(const int b, const int dir, const int idx, const int k, const int j,
                   const int i) const {
    PARTHENON_DEBUG_REQUIRE(dir > 0 && dir < 4 && with_fluxes_, "Bad input to flux call"); 
    return pack_(dir, b, idx)(k, j, i);
  }

}; 

class SparsePackCache { 
 public: 
  template<class T> 
  bool Add(T* ppack);

  template<class T> 
  bool TryLoad(T* ppack);
  
  // TODO (LFR) : Finish writing this code so that logic for finding cached pack is entirely within SparsePackCache 
  template <class T> 
  SparsePackBase &Get(T* pmd, const PackDescriptor& desc) {
    std::string ident = GetIdentifier(desc); 
    if (pack_map.count(ident) > 0) {
      auto& pack = pack_map[ident];
      if (desc.with_fluxes != pack.with_fluxes_) goto make_new_pack;  
      auto alloc_status_h = SparsePackBase::GetAllocStatus(pmd, desc); 
      if (alloc_status_h.size() != pack.alloc_status_h_.size()) goto make_new_pack;
      for (int i=0; i<alloc_status_h.size(); ++i) {
        if (alloc_status_h(i) != pack.alloc_status_h_(i)) goto make_new_pack;
      } 
      return pack_map[ident]; 
    }

  make_new_pack:
    SparsePackBase pack = SparsePackBase::Build(pmd, desc); 
    pack_map[ident] = pack; 
    return pack;
  }
  
  void clear() { pack_map.clear(); }
 
 protected: 
  std::string GetIdentifier(const PackDescriptor &desc) const {
    std::string identifier("");
    for (const auto& flag : desc.flags) identifier += flag.Name();
    identifier += "____";  
    for (int i=0; i < desc.vars.size(); ++i) identifier += desc.vars[i] + std::to_string(desc.use_regex[i]); 
    return identifier;
  } 

  std::unordered_map<std::string, SparsePackBase> pack_map; 
};

template <class... Ts>
class SparsePack : public SparsePackBase {
 public:
  SparsePack() = default;

  template <class T>
  explicit SparsePack(T *pmd, const std::vector<MetadataFlag> &flags = {}, bool with_fluxes = false) 
    : SparsePackBase(Build(pmd, PackDescriptor{std::vector<std::string>{Ts::name()...}, std::vector<bool>{Ts::regex()...}, flags, with_fluxes})) {}
  
  template <class T> 
  SparsePack(T *pmd, SparsePackCache* pcache, const std::vector<MetadataFlag> &flags = {}, bool with_fluxes = false) {
    PackDescriptor desc{std::vector<std::string>{Ts::name()...}, std::vector<bool>{Ts::regex()...}, flags, with_fluxes};
    auto include_variable = SparsePackBase::GetTestFunction(desc);
    alloc_status_h_ = GetAllocStatus(pmd, desc); 
    with_fluxes_ = with_fluxes;
    if (!pcache->TryLoad(this)) {
      // Ok since there should be no data members of SparsePack itself
      static_cast<SparsePackBase&>(*this) = Build(pmd, desc);
      pcache->Add(this);
    }
  }
  
  explicit SparsePack(const SparsePackBase& spb) : SparsePackBase(spb) {}
  
  template <class T>
  static SparsePack Make(T* pmd, const std::vector<MetadataFlag> &flags = {}) {
    return MakeImpl(pmd, flags, false, static_cast<int>(0)); 
  }

  template <class T>
  static SparsePack Make(T* pmd, SparsePackCache* pcache, const std::vector<MetadataFlag> &flags = {}) {
    return SparsePack(pmd, pcache, flags, false); 
  }

  template <class T>
  static SparsePack MakeWithFluxes(T* pmd, const std::vector<MetadataFlag> &flags = {}) {
    return MakeImpl(pmd, flags, true, static_cast<int>(0)); 
  }
  
  template <class T>
  static SparsePack MakeWithFluxes(T* pmd, SparsePackCache* pcache, const std::vector<MetadataFlag> &flags = {}) {
    return SparsePack(pmd, pcache, flags, true); 
  } 

  std::string GetIdentifier() const {
    const std::vector<std::string> idents{(Ts::name() + std::to_string(Ts::regex()))...}; 
    std::string identifier(""); 
    for (const auto &str : idents) identifier += str;
    return identifier;
  }

  template <class TIn>
  KOKKOS_INLINE_FUNCTION int GetLowerBound(const int b, const TIn &) const {
    const int vidx = GetTypeIdx<TIn, Ts...>::value;
    return bounds_(0, b, vidx);
  }

  template <class TIn>
  KOKKOS_INLINE_FUNCTION int GetUpperBound(const int b, const TIn &) const {
    const int vidx = GetTypeIdx<TIn, Ts...>::value;
    return bounds_(1, b, vidx);
  }
  
  // This has to be defined here since the base class operator is apparently 
  // covered by the template operator below even if std::enable_if fails
  KOKKOS_INLINE_FUNCTION
  Real &operator()(const int b, const int idx, const int k, const int j,
                   const int i) const {
    return pack_(0, b, idx)(k, j, i);
  }

  template <class TIn, class = typename std::enable_if<!std::is_integral<TIn>::value>::type>
  KOKKOS_INLINE_FUNCTION Real &operator()(const int b, const TIn &t, const int k,
                                          const int j, const int i) const {
    const int vidx = GetLowerBound(t, b) + t.idx;
    return pack_(0, b, vidx)(k, j, i);
  }

  KOKKOS_INLINE_FUNCTION
  Real &flux(const int b, const int dir, const int idx, const int k, const int j,
                   const int i) const {
    PARTHENON_DEBUG_REQUIRE(dir > 0 && dir < 4 && with_fluxes_, "Bad input to flux call"); 
    return pack_(dir, b, idx)(k, j, i);
  }
  
  template <class TIn, class = typename std::enable_if<!std::is_integral<TIn>::value>::type>
  KOKKOS_INLINE_FUNCTION Real &flux(const int b, const int dir, const TIn &t, const int k,
                                          const int j, const int i) const {
    PARTHENON_DEBUG_REQUIRE(dir > 0 && dir < 4 && with_fluxes_, "Bad input to flux call"); 
    const int vidx = GetLowerBound(t, b) + t.idx;
    return pack_(dir, b, vidx)(k, j, i);
  }



 protected:
  
  template <class T>
  static auto MakeImpl(T* pmd, const std::vector<MetadataFlag> &flags, bool fluxes, int) 
      -> decltype(T().GetSparsePackCache(), SparsePack()) {
    return SparsePack(pmd, &(pmd->GetSparsePackCache()), flags, fluxes); 
  }

  template <class T>
  static SparsePack MakeImpl(T* pmd, const std::vector<MetadataFlag> &flags, bool fluxes, double) {
    return SparsePack(pmd, flags, fluxes); 
  }

  static auto GetTestFunction(const std::vector<MetadataFlag> &flags = {}) { 
    const std::vector<std::string> names{Ts::name()...};
    const std::vector<std::regex> regexes{std::regex(Ts::name())...};
    const std::vector<bool> use_regex{Ts::regex()...};
    
    // Lambda for testing wether or not we want to include cellvariable pv 
    // in the index range of the type variable with index vidx 
    return [=](int vidx, const std::shared_ptr<CellVariable<Real>> &pv) {
      // TODO(LFR): Check that the shapes agree
      
      if (flags.size() > 0) {
        for (const auto& flag : flags) {
          if (!pv->IsSet(flag)) {
            return false;
          }
        }
      }
      
      if (use_regex[vidx]) {
        if (std::regex_match(std::string(pv->label()), regexes[vidx])) return true;
      } else {
        if (names[vidx] == pv->label()) return true;
      }
      return false;
    };
  }
};

//-----------------------------------------------------------------------------
// Declarations below 
//-----------------------------------------------------------------------------
template <class T> 
SparsePackBase::alloc_t SparsePackBase::GetAllocStatus(T *pmd, const PackDescriptor &desc) {
  
  using mbd_t = decltype(MeshBlockDataTypeHack<T>());

  int nvar = desc.vars.size();
  auto include_variable = GetTestFunction(desc);
  
  std::vector<bool> astat; 
  IterateBlocks(pmd, [&](int b, mbd_t *pmbd) {
    for (int i = 0; i < nvar; ++i) {
      for (auto &pv : pmbd->GetCellVariableVector()) {
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

template <class T> 
SparsePackBase SparsePackBase::Build(T *pmd, const PackDescriptor &desc) {
  
  using mbd_t = decltype(MeshBlockDataTypeHack<T>());
  int nvar = desc.vars.size();
  
  auto include_variable = GetTestFunction(desc);
  SparsePackBase pack; 
  pack.alloc_status_h_ = GetAllocStatus(pmd, desc);
  pack.with_fluxes_ = desc.with_fluxes;

  // Count up the size of the array that is required
  int max_size = 0;
  int nblocks = 0;
  int ndim = 3;
  IterateBlocks(pmd, [&](int b, mbd_t *pmbd) {
    int size = 0;
    nblocks++;
    for (auto &pv : pmbd->GetCellVariableVector()) {
      for (int i = 0; i < nvar; ++i) {
        if (include_variable(i, pv)) {
          if (pv->IsAllocated()) {
            size += pv->GetDim(6) * pv->GetDim(5) * pv->GetDim(4);
            ndim = (pv->GetDim(1) > 1 ? 1 : 0) 
                 + (pv->GetDim(2) > 1 ? 1 : 0) 
                 + (pv->GetDim(3) > 1 ? 1 : 0);
          }
        }
      }
    }
    max_size = std::max(size, max_size);
  });
  pack.nblocks_ = nblocks; 

  int leading_dim = 1; 
  if (desc.with_fluxes) leading_dim += 3;
  pack.pack_ = pack_t("data_ptr", leading_dim, nblocks, max_size);
  auto pack_h = Kokkos::create_mirror_view(pack.pack_);
  
  pack.bounds_ = bounds_t("bounds", 2, nblocks, nvar); 
  auto bounds_h = Kokkos::create_mirror_view(pack.bounds_);
  
  pack.coords_ = coords_t("coords", nblocks); 
  auto coords_h = Kokkos::create_mirror_view(pack.coords_);
  
  IterateBlocks(pmd, [&](int b, mbd_t *pmbd) {
    int idx = 0;
    coords_h(b) = pmbd->GetBlockPointer()->coords_device;

    for (int i = 0; i < nvar; ++i) {
      bounds_h(0, b, i) = idx;
    
      for (auto &pv : pmbd->GetCellVariableVector()) {
        if (include_variable(i, pv)) {
          if (pv->IsAllocated()) {
            for (int t = 0; t < pv->GetDim(6); ++t) {
              for (int u = 0; u < pv->GetDim(5); ++u) {
                for (int v = 0; v < pv->GetDim(4); ++v) {
                  pack_h(0, b, idx) = pv->data.Get(t, u, v);
                  PARTHENON_REQUIRE(pack_h(0, b, idx).size() > 10, "Seems like this pack might not actually be allocated.");
                  if (desc.with_fluxes && pv->IsSet(Metadata::WithFluxes)) {
                    // TODO (LFR): Need to add some checks for flux allocation 
                    pack_h(1, b, idx) = pv->flux[1].Get(t, u, v); 
                    PARTHENON_REQUIRE(pack_h(1, b, idx).size() == pack_h(0, b, idx).size(), "Different size fluxes.");
                    if (ndim > 1) { 
                      pack_h(2, b, idx) = pv->flux[2].Get(t, u, v); 
                      PARTHENON_REQUIRE(pack_h(2, b, idx).size() == pack_h(0, b, idx).size(), "Different size fluxes.");
                    }
                    if (ndim > 2) { 
                      pack_h(3, b, idx) = pv->flux[3].Get(t, u, v); 
                      PARTHENON_REQUIRE(pack_h(3, b, idx).size() == pack_h(0, b, idx).size(), "Different size fluxes.");
                    }
                  }
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
  Kokkos::deep_copy(pack.coords_, coords_h);
  pack.ndim_ = ndim;

  return pack;
}

template<class T> 
bool SparsePackCache::Add(T* ppack) { 
  std::string ident = ppack->GetIdentifier(); 
  bool preexists = pack_map.count(ident); 
  pack_map[ident] = static_cast<SparsePackBase>(*ppack);
  return preexists;
}

template<class T> 
bool SparsePackCache::TryLoad(T* ppack) {
  std::string ident = ppack->GetIdentifier();
  if (pack_map.count(ident) > 0) {
    auto& pack = pack_map[ident];
    if (ppack->with_fluxes_ != pack.with_fluxes_) return false; 
    if (ppack->alloc_status_h_.size() != pack.alloc_status_h_.size()) return false;
    for (int i=0; i<ppack->alloc_status_h_.size(); ++i) {
      if (ppack->alloc_status_h_(i) != pack.alloc_status_h_(i)) return false;
    }

    *ppack = T(pack);
    return true;
  }
  return false;
}

} // namespace parthenon

#endif // INTERFACE_SPARSE_PACK_HPP_
