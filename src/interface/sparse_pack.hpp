//========================================================================================
// (C) (or copyright) 2020-2023. Triad National Security, LLC. All rights reserved.
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
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <regex>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "coordinates/coordinates.hpp"
#include "interface/sparse_pack_base.hpp"
#include "interface/variable.hpp"
#include "utils/concepts_lite.hpp"
#include "utils/utils.hpp"

namespace parthenon {

// Sparse pack index type which allows for relatively simple indexing
// into non-variable name type based SparsePacks (i.e. objects of
// type SparsePack<> which are created with a vector of variable
// names and/or regexes)
class PackIdx {
 public:
  KOKKOS_INLINE_FUNCTION
  explicit PackIdx(std::size_t var_idx) : vidx(var_idx), offset(0) {}
  KOKKOS_INLINE_FUNCTION
  PackIdx(std::size_t var_idx, int off) : vidx(var_idx), offset(off) {}

  KOKKOS_INLINE_FUNCTION
  PackIdx &operator=(std::size_t var_idx) {
    vidx = var_idx;
    offset = 0;
    return *this;
  }

  KOKKOS_INLINE_FUNCTION
  std::size_t VariableIdx() { return vidx; }
  KOKKOS_INLINE_FUNCTION
  int Offset() { return offset; }

 private:
  std::size_t vidx;
  int offset;
};

// Operator overloads to make calls like `my_pack(b, my_pack_idx + 3, k, j, i)` work
template <class T, REQUIRES(std::is_integral<T>::value)>
KOKKOS_INLINE_FUNCTION PackIdx operator+(PackIdx idx, T offset) {
  return PackIdx(idx.VariableIdx(), idx.Offset() + offset);
}

template <class T, REQUIRES(std::is_integral<T>::value)>
KOKKOS_INLINE_FUNCTION PackIdx operator+(T offset, PackIdx idx) {
  return idx + offset;
}

// Namespace in which to put variable name types that are used for
// indexing into SparsePack<[type list of variable name types]> on
// device
namespace variable_names {
// Struct that all variable_name types should inherit from
template <bool REGEX, int... NCOMP>
struct base_t {
  KOKKOS_INLINE_FUNCTION
  base_t() : idx(0) {}

  KOKKOS_INLINE_FUNCTION
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
  
  static std::vector<int> shape() { return {NCOMP...}; }
  const int idx;
};

// An example variable name type that selects all variables available
// on Mesh*Data
struct any : public base_t<true> {
  template <class... Ts>
  KOKKOS_INLINE_FUNCTION any(Ts &&...args) : base_t<true>(std::forward<Ts>(args)...) {}
  static std::string name() { return ".*"; }
};
} // namespace variable_names

template <class... Ts>
class SparsePack : public SparsePackBase {
 public:
  SparsePack() = default;

  explicit SparsePack(const SparsePackBase &spb) : SparsePackBase(spb) {}

  class Descriptor : public impl::PackDescriptor {
   public:
    explicit Descriptor(const impl::PackDescriptor &desc_in)
        : impl::PackDescriptor(desc_in) {}

    // Make a `SparsePack` from variable_name types in the type list Ts..., creating the
    // pack in `pmd->SparsePackCache` if it doesn't already exist. Variables can be
    // accessed on device via instance of types in the type list Ts...
    // The pack will be created and accessible on the device
    template <class T>
    SparsePack GetPack(T *pmd) const {
      return SparsePack(SparsePackBase::GetPack(pmd, *this));
    }

    SparsePackIdxMap GetMap() const {
      PARTHENON_REQUIRE(sizeof...(Ts) == 0,
                        "Should not be getting an IdxMap for a type based pack");
      return SparsePackBase::GetIdxMap(*this);
    }
  };

  // Methods for getting parts of the shape of the pack
  KOKKOS_FORCEINLINE_FUNCTION
  int GetNBlocks() const { return nblocks_; }

  KOKKOS_FORCEINLINE_FUNCTION
  int GetMaxNumberOfVars() const { return pack_.extent_int(2); }

  // Returns the total number of vars/components in pack
  KOKKOS_FORCEINLINE_FUNCTION
  int GetSize() const { return size_; }

  KOKKOS_INLINE_FUNCTION
  const Coordinates_t &GetCoordinates(const int b = 0) const { return coords_(b)(); }

  // Bound overloads
  KOKKOS_INLINE_FUNCTION int GetLowerBound(const int b) const {
    return (flat_ && (b > 0)) ? (bounds_(1, b - 1, nvar_) + 1) : 0;
  }

  KOKKOS_INLINE_FUNCTION int GetUpperBound(const int b) const {
    return bounds_(1, b, nvar_);
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

  // Host Bound overloads
  KOKKOS_INLINE_FUNCTION int GetLowerBoundHost(const int b) const {
    return (flat_ && (b > 0)) ? (bounds_h_(1, b - 1, nvar_) + 1) : 0;
  }

  KOKKOS_INLINE_FUNCTION int GetUpperBoundHost(const int b) const {
    return bounds_h_(1, b, nvar_);
  }

  KOKKOS_INLINE_FUNCTION int GetLowerBoundHost(const int b, PackIdx idx) const {
    static_assert(sizeof...(Ts) == 0);
    return bounds_h_(0, b, idx.VariableIdx());
  }

  KOKKOS_INLINE_FUNCTION int GetUpperBoundHost(const int b, PackIdx idx) const {
    static_assert(sizeof...(Ts) == 0);
    return bounds_h_(1, b, idx.VariableIdx());
  }

  template <class TIn, REQUIRES(IncludesType<TIn, Ts...>::value)>
  KOKKOS_INLINE_FUNCTION int GetLowerBoundHost(const int b, const TIn &) const {
    const int vidx = GetTypeIdx<TIn, Ts...>::value;
    return bounds_h_(0, b, vidx);
  }

  template <class TIn, REQUIRES(IncludesType<TIn, Ts...>::value)>
  KOKKOS_INLINE_FUNCTION int GetUpperBoundHost(const int b, const TIn &) const {
    const int vidx = GetTypeIdx<TIn, Ts...>::value;
    return bounds_h_(1, b, vidx);
  }

  // operator() overloads
  using TE = TopologicalElement;
  KOKKOS_INLINE_FUNCTION auto &operator()(const int b, const TE el, const int idx) const {
    return pack_(static_cast<int>(el) % 3, b, idx);
  }
  KOKKOS_INLINE_FUNCTION auto &operator()(const int b, const int idx) const {
    PARTHENON_DEBUG_REQUIRE(pack_(0, b, idx).topological_type == TopologicalType::Cell,
                            "Suppressed topological element index assumes that this is a "
                            "cell variable, but it isn't");
    return pack_(0, b, idx);
  }

  KOKKOS_INLINE_FUNCTION auto &operator()(const int b, const TE el, PackIdx idx) const {
    static_assert(sizeof...(Ts) == 0);
    PARTHENON_DEBUG_REQUIRE(!flat_, "Accessor cannot be used for flat packs");
    const int n = bounds_(0, b, idx.VariableIdx()) + idx.Offset();
    return pack_(static_cast<int>(el) % 3, b, n);
  }
  KOKKOS_INLINE_FUNCTION auto &operator()(const int b, PackIdx idx) const {
    return (*this)(b, TE::CC, idx);
  }

  template <class TIn, REQUIRES(IncludesType<TIn, Ts...>::value)>
  KOKKOS_INLINE_FUNCTION auto &operator()(const int b, const TE el, const TIn &t) const {
    const int vidx = GetLowerBound(b, t) + t.idx;
    return pack_(static_cast<int>(el) % 3, b, vidx);
  }

  template <class TIn, REQUIRES(IncludesType<TIn, Ts...>::value)>
  KOKKOS_INLINE_FUNCTION auto &operator()(const int b, const TIn &t) const {
    PARTHENON_DEBUG_REQUIRE(!flat_, "Accessor cannot be used for flat packs");
    const int vidx = GetLowerBound(b, t) + t.idx;
    return pack_(0, b, vidx);
  }

  KOKKOS_INLINE_FUNCTION
  Real &operator()(const int b, const int idx, const int k, const int j,
                   const int i) const {
    PARTHENON_DEBUG_REQUIRE(!flat_, "Accessor cannot be used for flat packs");
    return pack_(0, b, idx)(k, j, i);
  }
  KOKKOS_INLINE_FUNCTION
  Real &operator()(int idx, const int k, const int j, const int i) const {
    PARTHENON_DEBUG_REQUIRE(flat_, "Accessor valid only for flat packs");
    return pack_(0, 0, idx)(k, j, i);
  }

  KOKKOS_INLINE_FUNCTION Real &operator()(const int b, const TE el, const int idx,
                                          const int k, const int j, const int i) const {
    return pack_(static_cast<int>(el) % 3, b, idx)(k, j, i);
  }

  KOKKOS_INLINE_FUNCTION Real &operator()(const int b, PackIdx idx, const int k,
                                          const int j, const int i) const {
    static_assert(sizeof...(Ts) == 0, "Cannot create a string/type hybrid pack");
    PARTHENON_DEBUG_REQUIRE(!flat_, "Accessor cannot be used for flat packs");
    const int n = bounds_(0, b, idx.VariableIdx()) + idx.Offset();
    return pack_(0, b, n)(k, j, i);
  }

  KOKKOS_INLINE_FUNCTION Real &operator()(const int b, const TE el, PackIdx idx,
                                          const int k, const int j, const int i) const {
    static_assert(sizeof...(Ts) == 0, "Cannot create a string/type hybrid pack");
    const int n = bounds_(0, b, idx.VariableIdx()) + idx.Offset();
    return pack_(static_cast<int>(el) % 3, b, n)(k, j, i);
  }

  template <class TIn, REQUIRES(IncludesType<TIn, Ts...>::value)>
  KOKKOS_INLINE_FUNCTION Real &operator()(const int b, const TIn &t, const int k,
                                          const int j, const int i) const {
    PARTHENON_DEBUG_REQUIRE(!flat_, "Accessor cannot be used for flat packs");
    const int vidx = GetLowerBound(b, t) + t.idx;
    return pack_(0, b, vidx)(k, j, i);
  }

  template <class TIn, REQUIRES(IncludesType<TIn, Ts...>::value)>
  KOKKOS_INLINE_FUNCTION Real &operator()(const int b, const TE el, const TIn &t,
                                          const int k, const int j, const int i) const {
    const int vidx = GetLowerBound(b, t) + t.idx;
    return pack_(static_cast<int>(el) % 3, b, vidx)(k, j, i);
  }

  // flux() overloads
  KOKKOS_INLINE_FUNCTION
  auto &flux(const int b, const int dir, const int idx) const {
    PARTHENON_DEBUG_REQUIRE(!flat_, "Accessor cannot be used for flat packs");
    PARTHENON_DEBUG_REQUIRE(dir > 0 && dir < 4 && with_fluxes_, "Bad input to flux call");
    return pack_(dir, b, idx);
  }

  KOKKOS_INLINE_FUNCTION
  Real &flux(const int b, const int dir, const int idx, const int k, const int j,
             const int i) const {
    PARTHENON_DEBUG_REQUIRE(!flat_, "Accessor cannot be used for flat packs");
    PARTHENON_DEBUG_REQUIRE(dir > 0 && dir < 4 && with_fluxes_, "Bad input to flux call");
    return pack_(dir, b, idx)(k, j, i);
  }

  KOKKOS_INLINE_FUNCTION
  Real &flux(const int dir, const int idx, const int k, const int j, const int i) const {
    PARTHENON_DEBUG_REQUIRE(flat_, "Accessor must only be used for flat packs");
    PARTHENON_DEBUG_REQUIRE(dir > 0 && dir < 4 && with_fluxes_, "Bad input to flux call");
    return pack_(dir, 0, idx)(k, j, i);
  }

  KOKKOS_INLINE_FUNCTION
  Real &flux(const int b, const int dir, PackIdx idx, const int k, const int j,
             const int i) const {
    static_assert(sizeof...(Ts) == 0, "Cannot create a string/type hybrid pack");
    PARTHENON_DEBUG_REQUIRE(!flat_, "Accessor cannot be used for flat packs");
    PARTHENON_DEBUG_REQUIRE(dir > 0 && dir < 4 && with_fluxes_, "Bad input to flux call");
    const int n = bounds_(0, b, idx.VariableIdx()) + idx.Offset();
    return pack_(dir, b, n)(k, j, i);
  }

  template <class TIn, REQUIRES(IncludesType<TIn, Ts...>::value)>
  KOKKOS_INLINE_FUNCTION Real &flux(const int b, const int dir, const TIn &t, const int k,
                                    const int j, const int i) const {
    PARTHENON_DEBUG_REQUIRE(!flat_, "Accessor cannot be used for flat packs");
    PARTHENON_DEBUG_REQUIRE(dir > 0 && dir < 4 && with_fluxes_, "Bad input to flux call");
    const int vidx = GetLowerBound(b, t) + t.idx;
    return pack_(dir, b, vidx)(k, j, i);
  }

  template <class... VTs>
  KOKKOS_INLINE_FUNCTION auto GetPtrs(const int b, const TE el, int k, int j, int i,
                                      VTs... vts) const {
    return std::make_tuple(&(*this)(b, el, vts, k, j, i)...);
  }
};

inline auto MakePackDescriptor(StateDescriptor *psd, const std::vector<std::string> &vars,
                               const std::vector<bool> &use_regex,
                               const std::vector<MetadataFlag> &flags = {},
                               const std::set<PDOpt> &options = {}) {
  PARTHENON_REQUIRE(vars.size() == use_regex.size(),
                    "Vargroup names and use_regex need to be the same size.");
  auto selector = [&](int vidx, const VarID &id, const Metadata &md) {
    if (flags.size() > 0) {
      for (const auto &flag : flags) {
        if (!md.IsSet(flag)) return false;
      }
    }

    if (use_regex[vidx]) {
      if (std::regex_match(std::string(id.label()), std::regex(vars[vidx]))) return true;
    } else {
      if (vars[vidx] == id.label()) return true;
      if (vars[vidx] == id.base_name && id.sparse_id != InvalidSparseID) return true;
    }
    return false;
  };

  impl::PackDescriptor base_desc(psd, vars, selector, options);
  return typename SparsePack<>::Descriptor(base_desc);
}

template <class... Ts>
inline auto MakePackDescriptor(StateDescriptor *psd,
                               const std::vector<MetadataFlag> &flags = {},
                               const std::set<PDOpt> &options = {}) {
  static_assert(sizeof...(Ts) > 0, "Must have at least one variable type for type pack");

  std::vector<std::string> vars{Ts::name()...};
  std::vector<bool> use_regex{Ts::regex()...};

  return typename SparsePack<Ts...>::Descriptor(static_cast<impl::PackDescriptor>(
      MakePackDescriptor(psd, vars, use_regex, flags, options)));
}

inline auto MakePackDescriptor(StateDescriptor *psd, const std::vector<std::string> &vars,
                               const std::vector<MetadataFlag> &flags = {},
                               const std::set<PDOpt> &options = {}) {
  return MakePackDescriptor(psd, vars, std::vector<bool>(vars.size(), false), flags,
                            options);
}

inline auto MakePackDescriptor(
    StateDescriptor *psd, const std::vector<std::pair<std::string, bool>> &var_regexes,
    const std::vector<MetadataFlag> &flags = {}, const std::set<PDOpt> &options = {}) {
  std::vector<std::string> vars;
  std::vector<bool> use_regex;
  for (const auto &[v, r] : var_regexes) {
    vars.push_back(v);
    use_regex.push_back(r);
  }
  return MakePackDescriptor(psd, vars, use_regex, flags, options);
}

} // namespace parthenon

#endif // INTERFACE_SPARSE_PACK_HPP_
