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

#include <algorithm>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <regex>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "coordinates/coordinates.hpp"
#include "interface/mesh_data.hpp"
#include "interface/meshblock_data.hpp"
#include "interface/sparse_pack_base.hpp"
#include "interface/variable.hpp"
#include "utils/utils.hpp"

namespace {
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
} // namespace

namespace parthenon {

template <class T>
SparsePackBase::alloc_t SparsePackBase::GetAllocStatus(T *pmd,
                                                       const PackDescriptor &desc) {
  using mbd_t = MeshBlockData<Real>;

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
  for (int i = 0; i < astat.size(); ++i)
    alloc_status_h(i) = astat[i];
  return alloc_status_h;
}

// Specialize for the only two types this should work for
template SparsePackBase::alloc_t
SparsePackBase::GetAllocStatus<MeshBlockData<Real>>(MeshBlockData<Real> *,
                                                    const PackDescriptor &);
template SparsePackBase::alloc_t
SparsePackBase::GetAllocStatus<MeshData<Real>>(MeshData<Real> *, const PackDescriptor &);

template <class T>
SparsePackBase SparsePackBase::Build(T *pmd, const PackDescriptor &desc) {
  using mbd_t = MeshBlockData<Real>;
  int nvar = desc.vars.size();

  auto include_variable = GetTestFunction(desc);
  SparsePackBase pack;
  pack.alloc_status_h_ = GetAllocStatus(pmd, desc);
  pack.with_fluxes_ = desc.with_fluxes;
  pack.coarse_ = desc.coarse;
  pack.nvar_ = desc.vars.size();
  
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
            ndim = (pv->GetDim(1) > 1 ? 1 : 0) + (pv->GetDim(2) > 1 ? 1 : 0) +
                   (pv->GetDim(3) > 1 ? 1 : 0);
          }
        }
      }
    }
    max_size = std::max(size, max_size);
  });
  pack.nblocks_ = nblocks;

  // Allocate the views
  int leading_dim = 1;
  if (desc.with_fluxes) leading_dim += 3;
  pack.pack_ = pack_t("data_ptr", leading_dim, nblocks, max_size);
  auto pack_h = Kokkos::create_mirror_view(pack.pack_);

  pack.bounds_ = bounds_t("bounds", 2, nblocks, nvar);
  auto bounds_h = Kokkos::create_mirror_view(pack.bounds_);

  pack.coords_ = coords_t("coords", nblocks);
  auto coords_h = Kokkos::create_mirror_view(pack.coords_);

  // Fill the views
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
                  if (pack.coarse_) {
                    pack_h(0, b, idx) = pv->coarse_s.Get(t, u, v);
                  } else {
                    pack_h(0, b, idx) = pv->data.Get(t, u, v);
                  }
                  PARTHENON_REQUIRE(
                      pack_h(0, b, idx).size() > 10,
                      "Seems like this pack might not actually be allocated.");
                  if (desc.with_fluxes && pv->IsSet(Metadata::WithFluxes)) {
                    pack_h(1, b, idx) = pv->flux[1].Get(t, u, v);
                    PARTHENON_REQUIRE(pack_h(1, b, idx).size() ==
                                          pack_h(0, b, idx).size(),
                                      "Different size fluxes.");
                    if (ndim > 1) {
                      pack_h(2, b, idx) = pv->flux[2].Get(t, u, v);
                      PARTHENON_REQUIRE(pack_h(2, b, idx).size() ==
                                            pack_h(0, b, idx).size(),
                                        "Different size fluxes.");
                    }
                    if (ndim > 2) {
                      pack_h(3, b, idx) = pv->flux[3].Get(t, u, v);
                      PARTHENON_REQUIRE(pack_h(3, b, idx).size() ==
                                            pack_h(0, b, idx).size(),
                                        "Different size fluxes.");
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
  pack.dims_[1] = pack.nblocks_;
  pack.dims_[2] = -1; // Not allowed to ask for the ragged dimension anyway
  pack.dims_[3] = pack_h(0, 0, 0).extent_int(0);
  pack.dims_[4] = pack_h(0, 0, 0).extent_int(2);
  pack.dims_[5] = pack_h(0, 0, 0).extent_int(3);

  return pack;
}

// Specialize for the only two types this should work for
template SparsePackBase
SparsePackBase::Build<MeshBlockData<Real>>(MeshBlockData<Real> *, const PackDescriptor &);
template SparsePackBase SparsePackBase::Build<MeshData<Real>>(MeshData<Real> *,
                                                              const PackDescriptor &);

inline SparsePackBase::test_func_t
SparsePackBase::GetTestFunction(const PackDescriptor &desc) {
  std::vector<std::regex> regexes;
  for (const auto &var : desc.vars)
    regexes.push_back(std::regex(var));

  // Lambda for testing wether or not we want to include cellvariable pv
  // in the index range of the type variable with index vidx
  return [=](int vidx, const std::shared_ptr<CellVariable<Real>> &pv) {
    // TODO(LFR): Check that the shapes agree

    if (desc.flags.size() > 0) {
      for (const auto &flag : desc.flags) {
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

} // namespace parthenon
