//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020-2023 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2024. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001
// for Los Alamos National Laboratory (LANL), which is operated by Triad
// National Security, LLC for the U.S. Department of Energy/National Nuclear
// Security Administration. All rights in the program are reserved by Triad
// National Security, LLC, and the U.S. Department of Energy/National Nuclear
// Security Administration. The Government is granted for itself and others
// acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license
// in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do
// so.
//========================================================================================

#ifndef KOKKOS_TYPES_HPP_
#define KOKKOS_TYPES_HPP_

#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

#include <Kokkos_Core.hpp>

#include "basic_types.hpp"
#include "config.hpp"
#include "parthenon_array_generic.hpp"
#include "utils/multi_pointer.hpp"
#include "utils/object_pool.hpp"

namespace parthenon {

#ifdef KOKKOS_ENABLE_CUDA_UVM
using DevMemSpace = Kokkos::CudaUVMSpace;
using HostMemSpace = Kokkos::CudaUVMSpace;
using DevExecSpace = Kokkos::Cuda;
#else
using DevMemSpace = Kokkos::DefaultExecutionSpace::memory_space;
using HostMemSpace = Kokkos::HostSpace;
using DevExecSpace = Kokkos::DefaultExecutionSpace;
#endif
using ScratchMemSpace = DevExecSpace::scratch_memory_space;

using HostExecSpace = Kokkos::DefaultHostExecutionSpace;
using LayoutWrapper = Kokkos::LayoutRight;
using MemUnmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;

#if defined(PARTHENON_ENABLE_HOST_COMM_BUFFERS)
#if defined(KOKKOS_ENABLE_CUDA)
using BufMemSpace = Kokkos::CudaHostPinnedSpace::memory_space;
#elif defined(KOKKOS_ENABLE_HIP)
using BufMemSpace = Kokkos::Experimental::HipHostPinnedSpace::memory_space;
#else
#error "Unknow comm buffer space for chose execution space."
#endif
#else
using BufMemSpace = Kokkos::DefaultExecutionSpace::memory_space;
#endif

// MPI communication buffers
template <typename T>
using BufArray1D = Kokkos::View<T *, LayoutWrapper, BufMemSpace>;

// Structures for reusable memory pools and communication
template <typename T>
using buf_pool_t = ObjectPool<BufArray1D<T>>;

template <typename T, typename State = empty_state_t>
using ParArray0D = ParArrayGeneric<Kokkos::View<T, LayoutWrapper, DevMemSpace>, State>;
template <typename T, typename State = empty_state_t>
using ParArray1D = ParArrayGeneric<Kokkos::View<T *, LayoutWrapper, DevMemSpace>, State>;
template <typename T, typename State = empty_state_t>
using ParArray2D = ParArrayGeneric<Kokkos::View<T **, LayoutWrapper, DevMemSpace>, State>;
template <typename T, typename State = empty_state_t>
using ParArray3D =
    ParArrayGeneric<Kokkos::View<T ***, LayoutWrapper, DevMemSpace>, State>;
template <typename T, typename State = empty_state_t>
using ParArray4D =
    ParArrayGeneric<Kokkos::View<T ****, LayoutWrapper, DevMemSpace>, State>;
template <typename T, typename State = empty_state_t>
using ParArray5D =
    ParArrayGeneric<Kokkos::View<T *****, LayoutWrapper, DevMemSpace>, State>;
template <typename T, typename State = empty_state_t>
using ParArray6D =
    ParArrayGeneric<Kokkos::View<T ******, LayoutWrapper, DevMemSpace>, State>;
template <typename T, typename State = empty_state_t>
using ParArray7D =
    ParArrayGeneric<Kokkos::View<T *******, LayoutWrapper, DevMemSpace>, State>;
template <typename T, typename State = empty_state_t>
using ParArray8D =
    ParArrayGeneric<Kokkos::View<T ********, LayoutWrapper, DevMemSpace>, State>;

// Host mirrors
template <typename T>
using HostArray0D = typename ParArray0D<T>::HostMirror;
template <typename T>
using HostArray1D = typename ParArray1D<T>::HostMirror;
template <typename T>
using HostArray2D = typename ParArray2D<T>::HostMirror;
template <typename T>
using HostArray3D = typename ParArray3D<T>::HostMirror;
template <typename T>
using HostArray4D = typename ParArray4D<T>::HostMirror;
template <typename T>
using HostArray5D = typename ParArray5D<T>::HostMirror;
template <typename T>
using HostArray6D = typename ParArray6D<T>::HostMirror;
template <typename T>
using HostArray7D = typename ParArray7D<T>::HostMirror;

using team_policy = Kokkos::TeamPolicy<>;
using team_mbr_t = Kokkos::TeamPolicy<>::member_type;

template <typename T>
using ScratchPad1D = Kokkos::View<T *, LayoutWrapper, ScratchMemSpace, MemUnmanaged>;
template <typename T>
using ScratchPad2D = Kokkos::View<T **, LayoutWrapper, ScratchMemSpace, MemUnmanaged>;
template <typename T>
using ScratchPad3D = Kokkos::View<T ***, LayoutWrapper, ScratchMemSpace, MemUnmanaged>;
template <typename T>
using ScratchPad4D = Kokkos::View<T ****, LayoutWrapper, ScratchMemSpace, MemUnmanaged>;
template <typename T>
using ScratchPad5D = Kokkos::View<T *****, LayoutWrapper, ScratchMemSpace, MemUnmanaged>;
template <typename T>
using ScratchPad6D = Kokkos::View<T ******, LayoutWrapper, ScratchMemSpace, MemUnmanaged>;

// Used for ParArrayND
// TODO(JMM): Should all of parthenon_arrays.hpp
// be moved here? Or should all of the above stuff be moved to
// parthenon_arrays.hpp?
inline constexpr std::size_t MAX_VARIABLE_DIMENSION = 7;
template <typename T, typename Layout = LayoutWrapper>
using device_view_t =
    Kokkos::View<multi_pointer_t<T, MAX_VARIABLE_DIMENSION>, Layout, DevMemSpace>;
template <typename T, typename Layout = LayoutWrapper>
using host_view_t = typename device_view_t<T, Layout>::HostMirror;

} // namespace parthenon

#endif // KOKKOS_TYPES_HPP_
