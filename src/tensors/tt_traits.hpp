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

#ifndef TENSORS_TT_TRAITS_HPP
#define TENSORS_TT_TRAITS_HPP

#include "basic_types.hpp"
#include "kokkos_abstraction.hpp"

namespace parthenon {
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
} // namespace parthenon
#endif // TENSOR_TT_TRAITS_HPP