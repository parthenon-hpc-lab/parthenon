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
#ifndef PACK_SWARM_PACK_BASE_HPP_
#define PACK_SWARM_PACK_BASE_HPP_

#include <algorithm>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "interface/state_descriptor.hpp"
#include "interface/swarm_device_context.hpp"
#include "interface/variable.hpp"
#include "kokkos_abstraction.hpp"
#include "pack/pack_utils.hpp"
#include "utils/utils.hpp"

namespace parthenon {

namespace impl {
template <typename TYPE>
struct SwarmPackDescriptor;
}

template <typename TYPE>
class SwarmPackBase {
 public:
  SwarmPackBase() = default;
  virtual ~SwarmPackBase() = default;

  using pack_t = ParArray3DRaw<ParArray1D<TYPE>>;
  using bounds_t = ParArray3D<int>;
  using contexts_t = ParArray1DRaw<SwarmDeviceContext>;
  using contexts_h_t = typename contexts_t::HostMirror;
  using max_active_indices_t = ParArray1D<int>;
  using desc_t = impl::SwarmPackDescriptor<TYPE>;
  using idx_map_t = std::unordered_map<std::string, std::size_t>;

  // Build supplemental entries to SwarmPack that change on a cadence faster than the
  // other pack cache
  template <class T>
  static void BuildSupplemental(T *pmd, const SwarmPackDescriptor<TYPE> &desc,
                                SwarmPackBase<TYPE> &pack) {
    // Fill the views
    auto flat_index_map_h = Kokkos::create_mirror_view(pack.flat_index_map_);
    auto max_active_indices_h = Kokkos::create_mirror_view(pack.max_active_indices_);
    ForEachBlock(pmd, std::vector<bool>{}, [&](int b, auto *pmbd) {
      auto swarm = pmbd->GetSwarm(desc.swarm_name);
      pack.contexts_h_(b) = swarm->GetDeviceContext();
      max_active_indices_h(b) = swarm->GetMaxActiveIndex();
      flat_index_map_h(b) =
          (b == 0 ? 0 : flat_index_map_h(b - 1) + max_active_indices_h(b - 1) + 1);
    });
    flat_index_map_h(pack.nblocks_) =
        flat_index_map_h(pack.nblocks_ - 1) + max_active_indices_h(pack.nblocks_ - 1) + 1;
    // make it an inclusive bound
    pack.max_flat_index_ = flat_index_map_h(pack.nblocks_) - 1;

    Kokkos::deep_copy(pack.contexts_, pack.contexts_h_);
    Kokkos::deep_copy(pack.max_active_indices_, max_active_indices_h);
    Kokkos::deep_copy(pack.flat_index_map_, flat_index_map_h);
  }

  // Actually build a `SwarmPackBase` (i.e. create a view of views, fill on host, and
  // deep copy the view of views to device) from the variables specified in desc contained
  // from the blocks contained in pmd (which can either be MeshBlockData/MeshData).
  template <class T>
  static SwarmPackBase<TYPE> Build(T *pmd, const SwarmPackDescriptor<TYPE> &desc) {
    int nvar = desc.vars.size();

    SwarmPackBase<TYPE> pack;
    pack.nvar_ = desc.vars.size();

    // Count up the size of the array that is required
    int max_size = 0;
    int nblocks = 0;
    std::vector<int> vardims;
    ForEachBlock(pmd, std::vector<bool>{}, [&](int b, auto *pmbd) {
      auto swarm = pmbd->GetSwarm(desc.swarm_name);
      int size = 0;
      nblocks++;
      for (auto &pv : swarm->template GetVariableVector<TYPE>()) {
        for (int i = 0; i < nvar; ++i) {
          if (desc.IncludeVariable(i, pv)) {
            size += pv->GetDim(6) * pv->GetDim(5) * pv->GetDim(4) * pv->GetDim(3) *
                    pv->GetDim(2);
          }
        }
      }
      max_size = std::max(size, max_size);
    });
    pack.nblocks_ = nblocks;

    // Allocate the views
    int leading_dim = 1;
    pack.pack_ = pack_t(ViewOfViewAlloc("data_ptr"), leading_dim, nblocks, max_size);
    auto pack_h = create_view_of_view_mirror(pack.pack_);

    pack.bounds_ = bounds_t("bounds", 2, nblocks, nvar);
    auto bounds_h = Kokkos::create_mirror_view(pack.bounds_);

    // Fill the views
    ForEachBlock(pmd, std::vector<bool>{}, [&](int b, auto *pmbd) {
      int idx = 0;
      auto swarm = pmbd->GetSwarm(desc.swarm_name);
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

    pack.contexts_ = contexts_t(ViewOfViewAlloc("contexts"), nblocks);
    pack.contexts_h_ = create_view_of_view_mirror(pack.contexts_);
    pack.max_active_indices_ = max_active_indices_t("max_active_indices", nblocks);
    pack.flat_index_map_ = max_active_indices_t("flat_index_map", nblocks + 1);
    BuildSupplemental(pmd, desc, pack);

    return pack;
  }

  template <class T>
  static SwarmPackBase<TYPE> BuildAndAdd(T *pmd, const SwarmPackDescriptor<TYPE> &desc) {
    auto &pack_map = pmd->template GetSwarmPackCache<TYPE>().pack_map;
    pack_map[desc.identifier] = Build<T>(pmd, desc);
    return pack_map[desc.identifier];
  }

  template <class T>
  static SwarmPackBase<TYPE> Get(T *pmd, const impl::SwarmPackDescriptor<TYPE> &desc) {
    auto &pack_map = pmd->template GetSwarmPackCache<TYPE>().pack_map;
    if (pack_map.count(desc.identifier) > 0) {
      // Cached version is not stale, so just return a reference to it
      BuildSupplemental(pmd, desc, pack_map[desc.identifier]);
      return pack_map[desc.identifier];
    }
    return BuildAndAdd<T>(pmd, desc);
  }

  template <class T>
  static SwarmPackBase<TYPE> GetPack(T *pmd,
                                     const impl::SwarmPackDescriptor<TYPE> &desc) {
    return Get<T>(pmd, desc);
  }

  static idx_map_t GetIdxMap(const desc_t &desc) {
    idx_map_t map;
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
  contexts_h_t contexts_h_;
  max_active_indices_t max_active_indices_;
  max_active_indices_t flat_index_map_;

  int nblocks_;
  int nvar_;
  int max_flat_index_;
};

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

template <typename TYPE>
class SwarmPackCache {
 public:
  std::size_t size() const { return pack_map.size(); }

  void clear() { pack_map.clear(); }

  std::unordered_map<std::string, SwarmPackBase<TYPE>> pack_map;
};

} // namespace parthenon

#endif // PACK_SWARM_PACK_BASE_HPP_
