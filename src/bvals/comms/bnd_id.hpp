//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2024-2025 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2025. Triad National Security, LLC. All rights reserved.
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

#ifndef BVALS_COMMS_BND_ID_HPP_
#define BVALS_COMMS_BND_ID_HPP_

#include <memory>
#include <string>
#include <vector>

#include "basic_types.hpp"
#include "bvals/neighbor_block.hpp"
#include "coordinates/coordinates.hpp"
#include "interface/variable_state.hpp"
#include "mesh/domain.hpp"
#include "mesh/forest/logical_coordinate_transformation.hpp"
#include "utils/communication_buffer.hpp"
#include "utils/indexer.hpp"
#include "utils/object_pool.hpp"

namespace parthenon {

template <typename T>
class Variable;

// Provides the information necessary for identifying a unique variable-boundary
// buffer, identifying the coalesced buffer it is associated with, and its
// position within the coalesced buffer.
struct BndId {
  constexpr static std::size_t NDAT = 10;
  int data[NDAT];

  // Information for identifying the buffer with a communication
  // channel, variable, and the ranks it is communicated across
  KOKKOS_FORCEINLINE_FUNCTION
  int &send_gid() { return data[0]; }
  KOKKOS_FORCEINLINE_FUNCTION
  int &recv_gid() { return data[1]; }
  KOKKOS_FORCEINLINE_FUNCTION
  int &loc_idx() { return data[2]; }
  KOKKOS_FORCEINLINE_FUNCTION
  int &var_id() { return data[3]; }
  KOKKOS_FORCEINLINE_FUNCTION
  int &extra_id() { return data[4]; }
  KOKKOS_FORCEINLINE_FUNCTION
  int &rank_send() { return data[5]; }
  KOKKOS_FORCEINLINE_FUNCTION
  int &rank_recv() { return data[6]; }
  BoundaryType bound_type;

  // MeshData partition id of the *sender*
  // not set by constructors and only necessary for coalesced comms
  KOKKOS_FORCEINLINE_FUNCTION
  int &partition() { return data[7]; }
  KOKKOS_FORCEINLINE_FUNCTION
  int &size() { return data[8]; }
  KOKKOS_FORCEINLINE_FUNCTION
  int &start_idx() { return data[9]; }

  bool buf_allocated{false};
  buf_pool_t<Real>::weak_t buf;   // comm buffer from pool
  BufArray1D<Real> coalesced_buf; // Combined buffer

  void PrintInfo(const std::string &start);

  KOKKOS_DEFAULTED_FUNCTION
  BndId() = default;
  KOKKOS_DEFAULTED_FUNCTION
  BndId(const BndId &) = default;

  KOKKOS_FUNCTION
  ~BndId() {}

  explicit BndId(const int *const data_in) {
    for (int i = 0; i < NDAT; ++i) {
      data[i] = data_in[i];
    }
  }

  void Serialize(int *data_out) {
    for (int i = 0; i < NDAT; ++i) {
      data_out[i] = data[i];
    }
  }

  bool SameBVChannel(const BndId &other) {
    // Don't want to compare start_idx, so -1
    for (int i = 0; i < NDAT - 1; ++i) {
      if (data[i] != other.data[i]) return false;
    }
    return true;
  }

  static BndId GetSend(MeshBlock *pmb, const NeighborBlock &nb,
                       std::shared_ptr<Variable<Real>> v, BoundaryType b_type,
                       int partition, int start_idx);
};
} // namespace parthenon

#endif // BVALS_COMMS_BND_ID_HPP_
