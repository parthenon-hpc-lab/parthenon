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
#ifndef INTERFACE_SWARM_PACK_BASE_HPP_
#define INTERFACE_SWARM_PACK_BASE_HPP_

#include <algorithm>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <regex>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "coordinates/coordinates.hpp"
#include "interface/mesh_data.hpp"
#include "interface/meshblock_data.hpp"
#include "interface/state_descriptor.hpp"
#include "interface/variable.hpp"
#include "utils/utils.hpp"

namespace {
// SFINAE for block iteration so that sparse packs can work for MeshBlockData and MeshData
template <class T, class F>
inline auto ForEachBlock(T *pmd, F func) -> decltype(T().GetBlockData(0), void()) {
  for (int b = 0; b < pmd->NumBlocks(); ++b) {
    auto &pmbd = pmd->GetBlockData(b);
    func(b, pmbd.get());
  }
}

template <class T, class F>
inline auto ForEachBlock(T *pmbd, F func) -> decltype(T().GetBlockPointer(), void()) {
  func(0, pmbd);
}

// Check data types of requested swarm variables
template <typename Head, typename... Tail>
struct GetDataType {
  using value = typename Head::data_type;
  static_assert(std::is_same<value, typename GetDataType<Tail...>::value>::value,
                "Types must all be the same");
};
template <typename T>
struct GetDataType<T> {
  using value = typename T::data_type;
};
} // namespace

namespace parthenon {

namespace impl {
template <typename TYPE>
struct SwarmPackDescriptor {
  SwarmPackDescriptor(const std::string &swarm_name, const std::vector<std::string> &vars)
      : swarm_name(swarm_name), vars(vars), identifier(GetIdentifier()) {}

  // Determining if variable pv should be included in SwarmPack
  bool IncludeVariable(int vidx,
                       const std::shared_ptr<ParticleVariable<TYPE>> &pv) const {
    if (vars[vidx] == pv->label()) return true;
    return false;
  }

  const std::string swarm_name;
  const std::vector<std::string> vars;
  const std::string identifier;

 private:
  std::string GetIdentifier() const {
    std::string ident("");
    for (const auto &var : vars)
      ident += var;
    ident += "|swarm_name:";
    ident += swarm_name;
    return ident;
  }
};
} // namespace impl

using SwarmPackIdxMap = std::unordered_map<std::string, std::size_t>;

template <typename TYPE>
class SwarmPackBase {
 public:
  SwarmPackBase() = default;
  virtual ~SwarmPackBase() = default;

  using pack_t = ParArray3D<ParArray1D<TYPE>>;
  using bounds_t = ParArray3D<int>;
  using contexts_t = ParArray1D<SwarmDeviceContext>;
  using max_active_indices_t = ParArray1D<int>;
  using desc_t = impl::SwarmPackDescriptor<TYPE>;

  // Actually build a `SwarmPackBase` (i.e. create a view of views, fill on host, and
  // deep copy the view of views to device) from the variables specified in desc contained
  // from the blocks contained in pmd (which can either be MeshBlockData/MeshData).
  template <class MBD>
  static SwarmPackBase<TYPE> Build(MBD *pmd, const SwarmPackDescriptor<TYPE> &desc) {
    using mbd_t = MeshBlockData<Real>;
    int nvar = desc.vars.size();

    SwarmPackBase<TYPE> pack;
    pack.nvar_ = desc.vars.size();

    // Count up the size of the array that is required
    int max_size = 0;
    int nblocks = 0;
    int ndim = 3;
    std::vector<int> vardims;
    ForEachBlock(pmd, [&](int b, mbd_t *pmbd) {
      auto swarm = pmbd->GetSwarm(desc.swarm_name);
      int size = 0;
      nblocks++;
      for (auto &pv : swarm->template GetVariableVector<TYPE>()) {
        for (int i = 0; i < nvar; ++i) {
          if (desc.IncludeVariable(i, pv)) {
            size += pv->GetDim(6) * pv->GetDim(5) * pv->GetDim(4) * pv->GetDim(3) *
                    pv->GetDim(2);
            ndim = 1;
            int vardim = (pv->GetDim(6) > 1 ? 1 : 0) + (pv->GetDim(5) > 1 ? 1 : 0) +
                         (pv->GetDim(4) > 1 ? 1 : 0) + (pv->GetDim(3) > 1 ? 1 : 0) +
                         (pv->GetDim(2) > 1 ? 1 : 0);
            vardims.push_back(vardim);
          }
        }
      }
      max_size = std::max(size, max_size);
    });
    pack.nblocks_ = nblocks;

    // Allocate the views
    int leading_dim = 1;
    pack.pack_ = pack_t("data_ptr", leading_dim, nblocks, max_size);
    auto pack_h = Kokkos::create_mirror_view(pack.pack_);

    int bounds_leading_size = 2 + 1 + *max_element(vardims.begin(), vardims.end());
    pack.bounds_ = bounds_t("bounds", bounds_leading_size, nblocks, nvar);
    auto bounds_h = Kokkos::create_mirror_view(pack.bounds_);

    pack.contexts_ = contexts_t("contexts", nblocks);
    auto contexts_h = Kokkos::create_mirror_view(pack.contexts_);

    pack.max_active_indices_ = max_active_indices_t("max_active_indices", nblocks);
    auto max_active_indices_h = Kokkos::create_mirror_view(pack.max_active_indices_);

    // Fill the views
    ForEachBlock(pmd, [&](int b, mbd_t *pmbd) {
      int idx = 0;
      auto swarm = pmbd->GetSwarm(desc.swarm_name);
      contexts_h(b) = swarm->GetDeviceContext();
      max_active_indices_h(b) = swarm->GetMaxActiveIndex();

      for (int i = 0; i < nvar; ++i) {
        bounds_h(0, b, i) = idx;
        for (auto &pv : swarm->template GetVariableVector<TYPE>()) {
          if (desc.IncludeVariable(i, pv)) {
            for (int t = 0; t < pv->GetDim(6); ++t) {
              for (int u = 0; u < pv->GetDim(5); ++u) {
                for (int v = 0; v < pv->GetDim(4); ++v) {
                  for (int l = 0; l < pv->GetDim(3); ++l) {
                    for (int m = 0; m < pv->GetDim(2); ++m) {
                      pack_h(0, b, idx) = pv->data.Get(0, t, u, v, l, m);
                      PARTHENON_REQUIRE(
                          pack_h(0, b, idx).size() > 0,
                          "Seems like this variable might not actually be allocated.");
                      idx++;
                    }
                  }
                }
              }
            }
            bounds_h(2, b, i) = (pv->GetDim(6) > 1 ? 1 : 0) +
                                (pv->GetDim(5) > 1 ? 1 : 0) +
                                (pv->GetDim(4) > 1 ? 1 : 0) +
                                (pv->GetDim(3) > 1 ? 1 : 0) + (pv->GetDim(2) > 1 ? 1 : 0);
            for (int d = 0; d < bounds_h(2, b, i); d++) {
              bounds_h(3 + d, b, i) = pv->GetDim(2 + d);
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
    Kokkos::deep_copy(pack.contexts_, contexts_h);
    Kokkos::deep_copy(pack.max_active_indices_, max_active_indices_h);
    return pack;
  }

  template <class MBD>
  static SwarmPackBase<TYPE> BuildAndAdd(MBD *pmd, const SwarmPackDescriptor<TYPE> &desc,
                                         const std::string &ident) {
    auto &pack_map = pmd->template GetSwarmPackCache<TYPE>().pack_map;
    pack_map[ident] = Build<MBD>(pmd, desc);
    return pack_map[ident];
  }

  template <class MBD>
  static SwarmPackBase<TYPE> Get(MBD *pmd, const impl::SwarmPackDescriptor<TYPE> &desc) {
    return BuildAndAdd<MBD>(pmd, desc, desc.identifier);
  }

  template <class MBD>
  static SwarmPackBase<TYPE> GetPack(MBD *pmd,
                                     const impl::SwarmPackDescriptor<TYPE> &desc) {
    return Get<MBD>(pmd, desc);
  }

  static SwarmPackIdxMap GetIdxMap(const desc_t &desc) {
    SwarmPackIdxMap map;
    std::size_t idx = 0;
    for (const auto &var : desc.vars) {
      map[var] = idx;
      ++idx;
    }
    return map;
  }

 public:
  pack_t pack_;
  bounds_t bounds_;
  contexts_t contexts_;
  max_active_indices_t max_active_indices_;

  int nblocks_;
  int nvar_;
};

template <typename TYPE>
class SwarmPackCache {
 public:
  std::size_t size() const { return pack_map.size(); }

  void clear() { pack_map.clear(); }

  std::unordered_map<std::string, SwarmPackBase<TYPE>> pack_map;
};

} // namespace parthenon

#endif // INTERFACE_SWARM_PACK_BASE_HPP_
