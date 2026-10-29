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
#ifndef LUKEWARM_LUKEWARM_PACK_HPP_
#define LUKEWARM_LUKEWARM_PACK_HPP_

// This file was made in part with generative AI.

#include <cstdint>
#include <map>
#include <string>
#include <tuple>
#include <type_traits>

#include <Kokkos_Core.hpp>

#include "basic_types.hpp"
#include "interface/mesh_data.hpp"
#include "interface/state_descriptor.hpp"
#include "kokkos_abstraction.hpp"
#include "lukewarm/lukewarm.hpp"
#include "lukewarm/lukewarm_geometry.hpp"
#include "lukewarm/lukewarm_variable.hpp"
#include "pack/default_names.hpp"
#include "pack/sparse_pack/make_pack_descriptor.hpp"
#include "utils/concepts_lite.hpp"
#include "utils/error_checking.hpp"
#include "utils/type_list.hpp"

namespace parthenon {

// Device access to the fields Ts... of one lukewarm on one MeshData, as in
//   auto pack = desc.GetPack(md);
//   par_for(..., 0, pack.GetNParticles() - 1, KOKKOS_LAMBDA(const int n) {
//     pack(lukewarm::x(), n) += dt * pack(vel(0), n);
//   });
// A pack is a snapshot: get a new one after particles are created or removed.
//
// Every particle remembers the block it was last located in. Locate(n) brings that up
// to date after the particle moves, and costs a few comparisons when it has not left
// the block:
//   pack(lukewarm::x(), n) += dt * pack(vel(0), n);
//   const auto loc = pack.Locate(n);
//   if (loc.InMeshData()) {
//     const auto [k, j, i] = pack.GetCellIndices(loc.Block(), n);
//     Kokkos::atomic_add(&field_pack(loc.Block(), field::deposition(), k, j, i), w);
//   }
template <class... Ts>
class LukewarmPack {
  static_assert(sizeof...(Ts) > 0, "A lukewarm pack needs at least one field");
  static_assert((LukewarmTypes::IsIn<typename Ts::data_type>() && ...),
                "Lukewarm fields must be Real, int, or std::uint64_t");

 public:
  class Descriptor {
   public:
    Descriptor() = default;
    Descriptor(const StateDescriptor *psd, const std::string &lukewarm_name)
        : lukewarm_name_(lukewarm_name) {
      PARTHENON_REQUIRE_THROWS(psd->LukewarmPresent(lukewarm_name),
                               "No lukewarm named " + lukewarm_name);
      const auto &fields = psd->AllLukewarmFields(lukewarm_name);
      (Check_<Ts>(fields), ...);
    }

    const std::string &LukewarmName() const { return lukewarm_name_; }

    template <class MD>
    LukewarmPack GetPack(MD *pmd) const {
      return LukewarmPack(pmd->GetLukewarm(lukewarm_name_), pmd->GetLukewarmGeometry());
    }

   private:
    template <class T>
    void Check_(const std::map<std::string, Metadata> &fields) const {
      const auto itr = fields.find(T::name());
      PARTHENON_REQUIRE_THROWS(itr != fields.end(), "Lukewarm " + lukewarm_name_ +
                                                        " has no field " + T::name());
      PARTHENON_REQUIRE_THROWS(
          itr->second.Type() == LukewarmTypeFlag<typename T::data_type>(),
          "Lukewarm field " + T::name() + " is packed with the wrong data type");
    }

    std::string lukewarm_name_;
  };

  LukewarmPack() = default;
  LukewarmPack(const Lukewarm &lukewarm, const LukewarmGeometry &geometry)
      : real_(lukewarm.GetData<Real>()), int_(lukewarm.GetData<int>()),
        uint64_(lukewarm.GetData<std::uint64_t>()), marks_(lukewarm.GetRemovalMarks()),
        geometry_(geometry), nparticles_(lukewarm.NumParticles()) {
    x_ = lukewarm.Get<Real>(lukewarm::x::name()).GetOffset();
    y_ = lukewarm.Get<Real>(lukewarm::y::name()).GetOffset();
    z_ = lukewarm.Get<Real>(lukewarm::z::name()).GetOffset();
    int i = 0;
    ((offsets_[i++] = lukewarm.Get<typename Ts::data_type>(Ts::name()).GetOffset()), ...);
  }

  KOKKOS_FORCEINLINE_FUNCTION
  int GetNParticles() const { return nparticles_; }

  template <class TIn>
    requires(IncludesType<TIn, Ts...>::value)
  KOKKOS_INLINE_FUNCTION auto &operator()(const TIn &t, const int n) const {
    constexpr auto i = TypeList<Ts...>::template GetIdx<TIn>();
    return Data_<typename TIn::data_type>()(offsets_[i] + t.idx, n);
  }

  // Find the block containing particle n, starting from the block it was last located
  // in, and remember it if it is in this MeshData. Wraps the position of the particle
  // across periodic boundaries.
  KOKKOS_INLINE_FUNCTION
  LukewarmLocation Locate(const int n) const {
    // An unlocated particle, with block -1, starts from the block nearest to it
    const auto loc =
        geometry_.Locate(Block_(n), real_(x_, n), real_(y_, n), real_(z_, n));
    if (loc.InMeshData()) Block_(n) = loc.Block();
    return loc;
  }

  // The block particle n was last located in, by Locate(n) or RelocateLukewarm(). It is
  // only current if the particle has not moved since.
  KOKKOS_FORCEINLINE_FUNCTION
  int GetBlockIndex(const int n) const {
    PARTHENON_DEBUG_REQUIRE(
        Block_(n) >= 0 &&
            geometry_.Contains(Block_(n), real_(x_, n), real_(y_, n), real_(z_, n)),
        "Lukewarm particle is not in the block it was located in");
    return Block_(n);
  }

  // Interior cell of block b holding particle n, in the order a field pack takes them,
  // where b is the particle's block from Locate(n).Block() or GetBlockIndex(n)
  KOKKOS_FORCEINLINE_FUNCTION
  auto GetCellIndices(const int b, const int n) const {
    int i, j, k;
    geometry_.Xtoijk(b, real_(x_, n), real_(y_, n), real_(z_, n), i, j, k);
    return std::make_tuple(k, j, i);
  }

  KOKKOS_FORCEINLINE_FUNCTION
  const LukewarmGeometry &GetGeometry() const { return geometry_; }

  // Remove particle n at the next Lukewarm::RemoveMarked()
  KOKKOS_INLINE_FUNCTION
  void MarkForRemoval(const int n) const { marks_(n) = 1; }
  // Storage has no holes, so a marked particle is the only kind a kernel may need to
  // skip: until Lukewarm::RemoveMarked() it is still among [0, GetNParticles())
  KOKKOS_FORCEINLINE_FUNCTION
  bool IsMarkedForRemoval(const int n) const { return marks_(n) != 0; }

 private:
  KOKKOS_FORCEINLINE_FUNCTION
  int &Block_(const int n) const { return int_(Lukewarm::block_row, n); }

  template <class T>
  KOKKOS_FORCEINLINE_FUNCTION const ParArray2DRaw<T> &Data_() const {
    if constexpr (std::is_same_v<T, Real>) {
      return real_;
    } else if constexpr (std::is_same_v<T, int>) {
      return int_;
    } else {
      return uint64_;
    }
  }

  ParArray2DRaw<Real> real_;
  ParArray2DRaw<int> int_;
  ParArray2DRaw<std::uint64_t> uint64_;
  ParArray1DRaw<int> marks_;
  LukewarmGeometry geometry_;
  Kokkos::Array<int, sizeof...(Ts)> offsets_;
  // Offsets of the fields every lukewarm has
  int x_ = 0, y_ = 0, z_ = 0;
  int nparticles_ = 0;
};

// Mirrors MakePackDescriptor<Ts...> for fields
template <class... Ts, class MT>
inline auto MakeLukewarmPackDescriptor(MT *pmd, const std::string &lukewarm_name) {
  return typename LukewarmPack<Ts...>::Descriptor(impl::GetStateDescriptor<MT>(pmd),
                                                  lukewarm_name);
}

} // namespace parthenon

#endif // LUKEWARM_LUKEWARM_PACK_HPP_
