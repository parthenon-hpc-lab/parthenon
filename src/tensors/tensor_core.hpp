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

#ifndef TENSORS_TENSOR_CORE_HPP
#define TENSORS_TENSOR_CORE_HPP

#include "basic_types.hpp"
#include "kokkos_abstraction.hpp"


struct ManagedTag {};
struct UnmanagedTag {};

template <class Device,
          class RealT = Real,
          class Layout = Kokkos::LayoutRight>
struct TensorTraits {
  using device_type = Device;
  using execution_space = typename device_type::execution_space;
  using memory_space = typename device_type::memory_space;
  using layout = Layout;
  using real_t = RealT;
  using scratch_memory_space = Kokkos::ScratchMemorySpace<execution_space>;

  using host_mirror_space =
      typename Kokkos::View<real_t*, layout, memory_space>::host_mirror_space;

  template <class OwnershipTag>
  using memory_traits =
      std::conditional_t<std::is_same_v<OwnershipTag, ManagedTag>,
                         Kokkos::MemoryTraits<0>,
                         Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  template <class DataType, class OwnershipTag>
  using view_t =
      Kokkos::View<DataType, layout, memory_space,
                   memory_traits<OwnershipTag>>;

  template <class DataType, class OwnershipTag>
  using host_view_t =
      Kokkos::View<DataType, layout, host_mirror_space,
                   memory_traits<OwnershipTag>>;
};

template <class TTraits, class OwnershipTag> 
using FiberView = typename TTraits::template view_t<typename TTraits::real_t*, OwnershipTag>;

template <class TTraits>
class TensorCoreDevice {
 public:  
  using real_t = typename TTraits::real_t;
  using fiber_unmanaged_t = FiberView<TTraits, UnmanagedTag>; 
  using device_fibers_view_t = typename TTraits::template view_t<fiber_unmanaged_t**, UnmanagedTag>;
 
  KOKKOS_FUNCTION
  TensorCoreDevice() = default;
  
  KOKKOS_FUNCTION
  TensorCoreDevice(int lr, int dd, int rr, const device_fibers_view_t &fibers)
      : lr(lr), dd(dd), rr(rr), fibers(fibers) {}
  
  KOKKOS_INLINE_FUNCTION int RR() const {return rr;}
  KOKKOS_INLINE_FUNCTION int DD() const {return dd;} 
  KOKKOS_INLINE_FUNCTION int LR() const {return lr;}

  KOKKOS_INLINE_FUNCTION
  real_t &operator()(int alpha, int j, int beta) const { 
    return fibers(alpha, beta)(j);
  }

  KOKKOS_INLINE_FUNCTION
  fiber_unmanaged_t fiber(int alpha, int beta) const {
    return fibers(alpha, beta);
  }
 private:
  int lr{0}, dd{0}, rr{0};
  device_fibers_view_t fibers;
};

template <class TTraits>
class TensorCoreHost {
 public:
  using real_t = typename TTraits::real_t;
  using fiber_unmanaged_t = FiberView<TTraits, UnmanagedTag>; 
  using fiber_managed_t = FiberView<TTraits, ManagedTag>; 
  using host_fibers_view_t = typename TTraits::template host_view_t<fiber_managed_t**, ManagedTag>;
  using device_managed_fibers_view_t = typename TTraits::template view_t<fiber_unmanaged_t**, ManagedTag>; 
  using device_unmanaged_fibers_view_t = typename TTraits::template view_t<fiber_unmanaged_t**, UnmanagedTag>; 
  
  TensorCoreHost() = default;

  TensorCoreHost(int lr_in, int dd, int rr_in) : dd(dd) { 
    RebuildOuterViews(lr_in, rr_in, [dd](int, int){
      // TODO(LFR): Switch to memory pools
      return fiber_managed_t("fiber_m", dd);
    });
  }

  void ReduceSize(int lr_in, int rr_in) { 
    // Reduce the size assuming the fibers in the reduced range are already correct
    PARTHENON_REQUIRE(lr_in <= lr, "Target sizes must be smaller than original sizes.");
    PARTHENON_REQUIRE(rr_in <= rr, "Target sizes must be smaller than original sizes.");
    auto old_host_fibers = host_fibers;
    RebuildOuterViews(lr_in, rr_in, [&](int l, int r){
      return old_host_fibers(l, r);
    }); 
  }

  int RR() const {return rr;}
  int DD() const {return dd;}
  int LR() const {return lr;}
    
  TensorCoreDevice<TTraits> GetTensorCoreDevice() { 
    return TensorCoreDevice<TTraits>(lr, dd, rr, device_unmanaged_fibers);
  }
 
 private:
  int lr{0}, dd{0}, rr{0};
  // Stores managed fibers on host to maintain ownership
  host_fibers_view_t  host_fibers;
  // Managed view of unmanaged fibers to maintain ownership of the fibers view
  device_managed_fibers_view_t  device_managed_fibers;
  // Unmanaged view of unmanaged fibers for use in view of views of views on device
  device_unmanaged_fibers_view_t  device_unmanaged_fibers;
  
  // The only routine that should change the structural state of a TensorCoreHost.
  // It preserves the core invariant that host_fibers, device_managed_fibers, and
  // device_unmanaged_fibers all describe the same (lr, rr) outer hierarchy.
  template <class ManagedFiberGetter>
  void RebuildOuterViews(int lr_new, int rr_new, ManagedFiberGetter &&get_fiber) {
    // Rebuild the outer (left-rank, right-rank) hierarchy while obtaining the
    // managed fibers through get_fiber(l, r).
    lr = lr_new;
    rr = rr_new;
  
    // host_fibers owns the managed fibers; device_managed_fibers owns the
    // device-side outer descriptor; device_unmanaged_fibers is the shallow
    // device wrapper used when constructing TensorCoreDevice.
    host_fibers = host_fibers_view_t("fibers_m", lr, rr);
    device_managed_fibers = device_managed_fibers_view_t("fibers_u", lr, rr);
    device_unmanaged_fibers =
        device_unmanaged_fibers_view_t(device_managed_fibers.data(), lr, rr);
  
    auto device_managed_fibers_h = Kokkos::create_mirror_view(device_managed_fibers);
  
    for (int l = 0; l < lr; ++l) {
      for (int r = 0; r < rr; ++r) {
        host_fibers(l, r) = get_fiber(l, r);
        auto &f = host_fibers(l, r);
        device_managed_fibers_h(l, r) = fiber_unmanaged_t(f.data(), f.extent(0));
      }
    }
  
    // Only the outer descriptor is copied here. The fiber data already live on
    // device and are not copied by this deep_copy.
    Kokkos::deep_copy(device_managed_fibers, device_managed_fibers_h);
  }
};

template <class TTraits>
class TensorTrain {
  TensorTrain(const std::vector<TensorCoreHost<TTraits>> &cores_in) : cores(cores_in) { 
    PARTHENON_REQUIRE(cores.front().LR() == 1, "First core must have left side size one.");
    PARTHENON_REQUIRE(cores.back().RR() == 1, "Last core must have right side size one.");
    for (int c = 1; c < NCores() - 1; ++c) {
      PARTHENON_REQUIRE(cores[c-1].RR() == cores[c].LR(), "Cores must have consistent ranks.");
    }
  }

  TensorTrain(const std::vector<int> &phys_dims, const std::vector<int> &ranks) {
    PARTHENON_REQUIRE(phys_dims.size() - 1 == ranks.size(), "Incompatible number of ranks and dimensions.");
    if (ranks.size() == 0) {
      cores.emplace_back(1, phys_dims[0], 1);
    } else {
      cores.emplace_back(1, phys_dims[0], ranks[0]);
      for (int c = 1; c < phys_dims.size() - 1; ++c)
        cores.emplace_back(ranks[c - 1], phys_dims[c], ranks[c]);
      cores.emplace_back(ranks.back(), phys_dims.back(), 1);
    }
  }

  auto NCores() const {return cores.size();}
  auto &GetCoreHost(int c) {return cores[c];}
  const auto &GetCoreHost(int c) const {return cores[c];}

  auto &operator()(int c) {return cores[c];}
  const auto &operator()(int c) const {return cores[c];}
 
 private:
  std::vector<TensorCoreHost<TTraits>> cores;
};

template <class TTraits>
struct TensorPack {
  using view_t = typename TTraits::template view_t<TensorCoreDevice<TTraits>***, ManagedTag>;
  // view of size (nblocks, nvars, ncores), we just store the cores of the trains rather than
  // explicitly storing the TensorTrain type
  view_t cores;
  int ncores_per_train;

  KOKKOS_INLINE_FUNCTION
  int GetNBlocks() const {return cores.extent_int(0); }
  KOKKOS_INLINE_FUNCTION
  int GetNCores() const {return cores.extent_int(2); }

  TensorPack(std::vector<TensorTrain<TTraits>> &trains) {
    int nvars{1}; // Placeholder
    ncores_per_train = trains[0].NCores();
    cores = view_t("TensorPack", trains.size(), nvars, ncores_per_train);
    auto cores_h = Kokkos::create_mirror_view(cores);
    
    for (int v = 0; v < nvars; ++v) {
      for (int t = 0; t < trains.size(); ++t) {
        PARTHENON_REQUIRE(trains[t].NCores() == ncores_per_train, "All trains must be the same dimension.");
        for (int c = 0; c < ncores_per_train; ++c) {
          cores_h(t, v, c) = trains[t].GetCoreHost(c).GetTensorCoreDevice();
        }
      }
    }
    Kokkos::deep_copy(cores, cores_h);
  }

  KOKKOS_INLINE_FUNCTION
  auto &operator(int b, int v, int c) {return cores(b, v, c);}
};

template <class TTraits>
KOKKOS_INLINE_FUNCTION
void CopyCoreBlock(parthenon::team_mbr_t member,
                   const TensorCoreDevice<TTraits> &src,
                   TensorCoreDevice<TTraits> &dst,
                   int loffset = 0, int roffset = 0) {
  for (int l = 0; l < src.LR(); ++l) {
    for (int r = 0; r < src.RR(); ++r) {
      auto const * const fs = &src(l, 0, r);
      auto *fd = &dst(l + loffset, 0, r + roffset);
      parthenon::par_for_inner(member, 0, src.DD() - 1,
                               [&](const int j) { fd[j] = fs[j]; });
    }
  }
}

template <class TTraits>
KOKKOS_INLINE_FUNCTION
void SetCoreBlock(parthenon::team_mbr_t member,
                  TensorCoreDevice<TTraits> &dst,
                  typename TTraits::real_t value,
                  std::pair<int, int> lrange, 
                  std::pair<int, int> rrange) {
  for (int l = lrange.first; l < lrange.second; ++l) {
    for (int r = rrange.first; r < rrange.second; ++r) {
      auto *fd = &dst(l, 0, r);
      parthenon::par_for_inner(member, 0, dst.DD() - 1,
                               [&](const int j) { fd[j] = value; });
    }
  }
}

template <class TTraits>
KOKKOS_INLINE_FUNCTION
void HadamardCoreBlocks(parthenon::team_mbr_t member,
                        const TensorCoreDevice<TTraits> &core_a,
                        const TensorCoreDevice<TTraits> &core_b,
                        TensorCoreDevice<TTraits> &core_c) {
  for (int la = 0; la < core_a.LR(); ++la) {
    for (int lb = 0; lb < core_b.LR(); ++lb) {
      const int lc = la * core_b.LR() + lb;

      for (int ra = 0; ra < core_a.RR(); ++ra) {
        for (int rb = 0; rb < core_b.RR(); ++rb) {
          const int rc = ra * core_b.RR() + rb;

          auto const * const fa = &core_a(la, 0, ra);
          auto const * const fb = &core_b(lb, 0, rb);
          auto *fc = &core_c(lc, 0, rc);

          parthenon::par_for_inner(member, 0, core_c.DD() - 1,
                                   [&](const int j) { fc[j] = fa[j] * fb[j]; });
        }
      }
    }
  }
}

template <class TTraits>
void SetTTPackToValue(TensorPack<TTraits> &pack, Real value) {
  constexpr int unused_scratch_size = 0;
  constexpr int unused_scratch_level = 1;
  parthenon::par_for_outer(
    PARTHENON_AUTO_LABEL, unused_scratch_size, unused_scratch_level,
    0, pack.GetNBlocks() - 1, 0, pack.GetNcores() - 1,
    KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int c) {
      auto &core = pack.(b, 0, c);
      SetCoreBlock(member, core, value, {0, core.LR()}, {0, core.RR()});
    });
}

template <class TTraits>
std::vector<TensorTrain<TTraits>>
NonDestructiveSum(std::vector<TensorTrain<TTraits>> &TrainsA,
                  std::vector<TensorTrain<TTraits>> &TrainsB) {
  PARTHENON_REQUIRE(TrainsA.size() == TrainsB.size(), "Must be adding the same number of TTs.");

  // First create the memory to store the new train
  std::vector<TensorTrain<TTraits>> TrainsC;
  TrainsC.reserve(TrainsA.size());

  for (int t = 0; t < TrainsA.size(); ++t) {
    const auto &train_A = TrainsA[t];
    const auto &train_B = TrainsB[t];
    std::vector<int> phys_dims, target_ranks;
    PARTHENON_REQUIRE(train_A.NCores() == train_B.NCores(), "Added trains must have the same number of cores.");
    for (int c = 0; c < train_A.NCores(); ++c) { 
      PARTHENON_REQUIRE(train_A(c).DD() == train_B(c).DD(), "Must have equivalent physical dims.");
      phys_dims.push_back(train_A(c).DD());
    }
    for (int c = 0; c < train_A.NCores() - 1; ++c)
      target_ranks.push_back(train_A(c).RR() + train_B(c).RR());
    TrainsC.emplace_back(phys_dims, target_ranks);
  }
  
  // Now make the packs, eventually this may be just one pack
  TensorPack<TTraits> pack_a(TrainsA);
  TensorPack<TTraits> pack_b(TrainsB);
  TensorPack<TTraits> pack_c(TrainsC);
  
  constexpr int unused_scratch_size = 0;
  constexpr int unused_scratch_level = 1;
  parthenon::par_for_outer(
    PARTHENON_AUTO_LABEL, unused_scratch_size, unused_scratch_level,
    0, pack_a.GetNBlocks() - 1, 0, pack_a.GetNcores() - 1,
    KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int c) {
      auto &core_c = pack_c(b, 0, c);

      // First add the a contribution      
      auto &core_a = pack_a(b, 0, c);
      CopyCoreBlock(member, core_a, core_c, 0, 0);

      // Then add the b contribution      
      auto &core_b = pack_b(b, 0, c);
      const int loffset = (c > 0) * core_a.LR(); // Should be zero if the first core
      const int roffset = (c != (pack_a.GetNcores() - 1)) * core_a.RR(); // Should be zero if the last core
      CopyCoreBlock(member, core_b, core_c, loffset, roffset);

      // Zero the off diagonals, probably lots of room to optimize here (e.g. fill with null fibers)
      if (loffset && roffset) {
        SetCoreBlock(member, core_c, 0.0, {0, core_a.LR()}, {core_a.RR(), core_a.RR() + core_b.RR()});
        SetCoreBlock(member, core_c, 0.0, {core_a.LR(), core_a.LR() + core_b.LR()}, {0, core_a.RR()});
      }
    });
  return TrainsC; 
}

template <class TTraits>
std::vector<TensorTrain<TTraits>>
HadamardProduct(std::vector<TensorTrain<TTraits>> &TrainsA,
                std::vector<TensorTrain<TTraits>> &TrainsB) {
  PARTHENON_REQUIRE(TrainsA.size() == TrainsB.size(),
                    "Must be taking the Hadamard product of the same number of TTs.");

  std::vector<TensorTrain<TTraits>> TrainsC;
  TrainsC.reserve(TrainsA.size());

  for (int t = 0; t < TrainsA.size(); ++t) {
    const auto &train_A = TrainsA[t];
    const auto &train_B = TrainsB[t];

    PARTHENON_REQUIRE(train_A.NCores() == train_B.NCores(),
                      "Hadamard product requires the same number of cores.");

    std::vector<int> phys_dims, target_ranks;
    for (int c = 0; c < train_A.NCores(); ++c) {
      PARTHENON_REQUIRE(train_A(c).DD() == train_B(c).DD(),
                        "Hadamard product requires matching physical dimensions.");
      phys_dims.push_back(train_A(c).DD());
    }
    for (int c = 0; c < train_A.NCores() - 1; ++c) {
      target_ranks.push_back(train_A(c).RR() * train_B(c).RR());
    }

    TrainsC.emplace_back(phys_dims, target_ranks);
  }

  TensorPack<TTraits> pack_a(TrainsA);
  TensorPack<TTraits> pack_b(TrainsB);
  TensorPack<TTraits> pack_c(TrainsC);

  constexpr int unused_scratch_size = 0;
  constexpr int unused_scratch_level = 1;
  parthenon::par_for_outer(
      PARTHENON_AUTO_LABEL, unused_scratch_size, unused_scratch_level,
      0, pack_a.GetNBlocks() - 1, 0, pack_a.GetNcores() - 1,
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int c) {
        auto &core_a = pack_a(b, 0, c);
        auto &core_b = pack_b(b, 0, c);
        auto &core_c = pack_c(b, 0, c);
        HadamardCoreBlocks(member, core_a, core_b, core_c);
      });

  return TrainsC;
}

#endif // TENSOR_TENSOR_CORE_HPP