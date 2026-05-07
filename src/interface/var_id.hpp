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
#ifndef INTERFACE_VAR_ID_HPP_
#define INTERFACE_VAR_ID_HPP_

#include <cstddef>
#include <functional>
#include <limits>
#include <string>

#include <Kokkos_Core.hpp>

#include "utils/error_checking.hpp"

namespace parthenon {

inline constexpr int InvalidSparseIDValue = std::numeric_limits<int>::min();

struct SparseID {
  int id_ = InvalidSparseIDValue;
  int id2_ = InvalidSparseIDValue;

  KOKKOS_INLINE_FUNCTION constexpr SparseID() = default;
  KOKKOS_INLINE_FUNCTION constexpr SparseID(int id)
      : id_(id), id2_(InvalidSparseIDValue) {}
  KOKKOS_INLINE_FUNCTION constexpr SparseID(int id, int id2) : id_(id), id2_(id2) {}

  KOKKOS_INLINE_FUNCTION static constexpr SparseID Scalar(const int id) {
    return SparseID{id};
  }
  KOKKOS_INLINE_FUNCTION static constexpr SparseID Pair(const int id, const int id2) {
    return SparseID{id, id2};
  }

  KOKKOS_INLINE_FUNCTION constexpr int operator()() const {
    PARTHENON_DEBUG_REQUIRE(id2_ == InvalidSparseIDValue,
                            "SparseID() only valid for scalar sparse IDs");
    return id_;
  }
  KOKKOS_INLINE_FUNCTION constexpr int operator()(const int idx) const {
    return (idx == 0) ? id_ : id2_;
  }
};

inline constexpr SparseID InvalidSparseID{InvalidSparseIDValue, InvalidSparseIDValue};

KOKKOS_INLINE_FUNCTION constexpr bool operator==(const SparseID lhs,
                                                 const SparseID rhs) {
  return lhs(0) == rhs(0) && lhs(1) == rhs(1);
}

KOKKOS_INLINE_FUNCTION constexpr bool operator!=(const SparseID lhs,
                                                 const SparseID rhs) {
  return !(lhs == rhs);
}

KOKKOS_INLINE_FUNCTION constexpr bool operator<(const SparseID lhs,
                                                const SparseID rhs) {
  return (lhs(0) < rhs(0)) || (lhs(0) == rhs(0) && lhs(1) < rhs(1));
}

KOKKOS_INLINE_FUNCTION constexpr bool operator>(const SparseID lhs,
                                                const SparseID rhs) {
  return rhs < lhs;
}

KOKKOS_INLINE_FUNCTION constexpr bool operator<=(const SparseID lhs,
                                                 const SparseID rhs) {
  return !(rhs < lhs);
}

KOKKOS_INLINE_FUNCTION constexpr bool operator>=(const SparseID lhs,
                                                 const SparseID rhs) {
  return !(lhs < rhs);
}

inline constexpr bool IsValidSparseID(const SparseID sparse_id) {
  return sparse_id(0) != InvalidSparseIDValue;
}

inline std::string MakeVarLabel(const std::string &base_name, const SparseID sparse_id) {
  if (sparse_id == InvalidSparseID) return base_name;
  if (sparse_id(1) == InvalidSparseIDValue) {
    return base_name + "_" + std::to_string(sparse_id(0));
  }
  return base_name + "_" + std::to_string(sparse_id(0)) + "_" +
         std::to_string(sparse_id(1));
}

/// We uniquely identify a variable by its full label, i.e. base name plus sparse ID.
/// However, sometimes we also need to be able to separate the base name from the sparse
/// ID. Instead of relying on the fact that they are separated by a "_", we store them
/// separately in VarID struct. This way we know that a dense variable "foo_3" does not
/// have a sparse ID and a sparse field "foo_3" has base name "foo" and sparse ID 3,
/// however, the two VarIDs representing them are still considered equal, so that we find
/// such duplicates
/// TODO(JMM): Using VarID machinery for prolongation/restriction
/// implies that all vars in a sparse pool have the same custom
/// prolongation/restriction operators.
struct VarID {
  std::string base_name;
  SparseID sparse_id;

  explicit VarID(const std::string base_name, SparseID sparse_id = InvalidSparseID)
      : base_name(base_name), sparse_id(sparse_id) {}

  std::string label() const { return MakeVarLabel(base_name, sparse_id); }

  bool operator==(const VarID &other) const { return (label() == other.label()); }
};

struct VarIDHasher {
  auto operator()(const VarID &vid) const {
    return std::hash<std::string>{}(vid.label());
  }
};

struct SparseIDHasher {
  auto operator()(const SparseID sparse_id) const {
    return std::hash<int>{}(sparse_id(0)) ^ (std::hash<int>{}(sparse_id(1)) << 1);
  }
};

} // namespace parthenon

#endif // INTERFACE_VAR_ID_HPP_
